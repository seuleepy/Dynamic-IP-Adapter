from typing import List
import torch
from PIL import Image

from diffusers.pipelines.controlnet import MultiControlNetModel

from .utils import is_torch2_available, get_generator
from .ip_adapter import IPAdapter

if is_torch2_available():
    from .attention_processor import (
        AttnProcessor2_0 as AttnProcessor,
    )
    from .attention_processor import (
        CNAttnProcessor2_0 as CNAttnProcessor,
    )

    from .attention_processor import (
        IPAttnProcessor2_0 as IPAttnProcessor,
    )
    from .attention_processor_time import (
        TimeIPAttnProcessor2_0 as TimeIPAttnProcessor,
    )
else:
    from .attention_processor import AttnProcessor, CNAttnProcessor, IPAttnProcessor
    from .attention_processor_time import TimeIPAttnProcessor
from .resampler import Resampler


class TimeIPAdapter(IPAdapter):

    def __init__(
        self,
        sd_pipe,
        image_encoder_path,
        ip_ckpt,
        device,
        num_tokens=4,
        time_attention: bool = None,
    ):
        self.time_attention = time_attention
        super().__init__(sd_pipe, image_encoder_path, ip_ckpt, device, num_tokens)

    def set_ip_adapter(self):
        unet = self.pipe.unet
        attn_procs = {}
        for name in unet.attn_processors.keys():
            cross_attention_dim = (
                None
                if name.endswith("attn1.processor")
                else unet.config.cross_attention_dim
            )
            if name.startswith("mid_block"):
                hidden_size = unet.config.block_out_channels[-1]
            elif name.startswith("up_blocks"):
                block_id = int(name[len("up_blocks.")])
                hidden_size = list(reversed(unet.config.block_out_channels))[block_id]
            elif name.startswith("down_blocks"):
                block_id = int(name[len("down_blocks.")])
                hidden_size = unet.config.block_out_channels[block_id]
            if cross_attention_dim is None:
                attn_procs[name] = AttnProcessor()
            else:
                if self.time_attention:
                    self.processor_cls = TimeIPAttnProcessor
                else:
                    self.processor_cls = IPAttnProcessor
                attn_procs[name] = self.processor_cls(
                    hidden_size=hidden_size,
                    cross_attention_dim=cross_attention_dim,
                    scale=1.0,
                    num_tokens=self.num_tokens,
                ).to(self.device, dtype=torch.float16)
        unet.set_attn_processor(attn_procs)
        if hasattr(self.pipe, "controlnet"):
            if isinstance(self.pipe.controlnet, MultiControlNetModel):
                for controlnet in self.pipe.controlnet.nets:
                    controlnet.set_attn_processor(
                        CNAttnProcessor(num_tokens=self.num_tokens)
                    )
            else:
                self.pipe.controlnet.set_attn_processor(
                    CNAttnProcessor(num_tokens=self.num_tokens)
                )

    def set_scale(self, scale):
        for attn_processor in self.pipe.unet.attn_processors.values():
            if isinstance(attn_processor, self.processor_cls):
                attn_processor.scale = scale

    def generate(
        self,
        pil_image=None,
        clip_image_embeds=None,
        prompt=None,
        negative_prompt=None,
        scale=1.0,
        num_samples=4,
        seed=None,
        guidance_scale=7.5,
        num_inference_steps=30,
        TI=None,
        **kwargs,
    ):
        if self.time_attention:
            if TI is None:
                if "step" in kwargs or "ratio" in kwargs:
                    pass
                else:
                    ValueError(
                        "Both 'step' and 'ratio' must be provided in the arguments when time_attention is True and TI is None"
                    )

        self.set_scale(scale)

        if self.time_attention:
            TimeIPAttnProcessor.global_inference_steps = []
            TimeIPAttnProcessor.global_text_weights = []
            TimeIPAttnProcessor.global_ip_weights = []

        if pil_image is not None:
            num_prompts = 1 if isinstance(pil_image, Image.Image) else len(pil_image)
        else:
            num_prompts = clip_image_embeds.size(0)

        if prompt is None:
            prompt = "best quality, high quality"
        if negative_prompt is None:
            negative_prompt = (
                "monochrome, lowres, bad anatomy, worst quality, low quality"
            )

        if not isinstance(prompt, List):
            prompt = [prompt] * num_prompts
        if not isinstance(negative_prompt, List):
            negative_prompt = [negative_prompt] * num_prompts

        image_prompt_embeds, uncond_image_prompt_embeds = self.get_image_embeds(
            pil_image=pil_image, clip_image_embeds=clip_image_embeds
        )
        bs_embed, seq_len, _ = image_prompt_embeds.shape
        image_prompt_embeds = image_prompt_embeds.repeat(1, num_samples, 1)
        image_prompt_embeds = image_prompt_embeds.view(
            bs_embed * num_samples, seq_len, -1
        )
        uncond_image_prompt_embeds = uncond_image_prompt_embeds.repeat(
            1, num_samples, 1
        )
        uncond_image_prompt_embeds = uncond_image_prompt_embeds.view(
            bs_embed * num_samples, seq_len, -1
        )

        with torch.inference_mode():
            prompt_embeds_, negative_prompt_embeds_ = self.pipe.encode_prompt(
                prompt,
                device=self.device,
                num_images_per_prompt=num_samples,
                do_classifier_free_guidance=True,
                negative_prompt=negative_prompt,
            )
            prompt_embeds = torch.cat([prompt_embeds_, image_prompt_embeds], dim=1)
            negative_prompt_embeds = torch.cat(
                [negative_prompt_embeds_, uncond_image_prompt_embeds], dim=1
            )

        generator = get_generator(seed, self.device)

        images = self.pipe(
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            generator=generator,
            TI=TI,
            **kwargs,
        ).images

        return images

    def get_weights(self):
        if self.time_attention:
            inference_steps = TimeIPAttnProcessor.global_inference_steps
            text_weights = TimeIPAttnProcessor.global_text_weights
            ip_weights = TimeIPAttnProcessor.global_ip_weights
            return inference_steps, text_weights, ip_weights
        else:
            return False
