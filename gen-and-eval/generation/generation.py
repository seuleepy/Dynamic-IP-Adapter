from constant import (
    TARGET2CLASS_MAPPING,
    CLASS2OBJECT_MAPPING,
    OBJECT_PROMPT_LIST,
    LIVE_PROMPT_LIST,
)

import torch
from diffusers import (
    TimeStableDiffusionPipeline,
    DDIMScheduler,
    AutoencoderKL,
)
from ip_adapter import TimeIPAdapter

from PIL import Image
import os
import argparse
from tqdm.auto import tqdm
import json


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--base_model_path",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--vae_model_path",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--image_encoder_path",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--ip_ckpt",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--device",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        required=True,
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        required=False,
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--model",
        type=int,
        required=True,
    )
    parser.add_argument("--store_config", action="store_true")
    args = parser.parse_args()

    return args


def create_config_file(args):
    config = {
        "base_model_path": args.base_model_path,
        "vae_model_path": args.vae_model_path,
        "image_encoder_path": args.image_encoder_path,
        "ip_ckpt": args.ip_ckpt,
        "dataset_path": args.dataset_path,
        "num_samples": args.num_samples,
        "seed": args.seed,
    }

    # JSON 파일로 저장
    with open(f"{args.output_dir}/config.json", "w") as f:
        json.dump(config, f, indent=4)

    print(f"Configuration saved to {args.output_dir}/config.json")


def main(args):
    noise_scheduler = DDIMScheduler(
        num_train_timesteps=1000,
        beta_start=0.00085,
        beta_end=0.012,
        beta_schedule="scaled_linear",
        clip_sample=False,
        set_alpha_to_one=False,
        steps_offset=1,
    )
    if args.vae_model_path is not None:
        vae = AutoencoderKL.from_pretrained(args.vae_model_path).to(dtype=torch.float16)
        pipe = TimeStableDiffusionPipeline.from_pretrained(
            args.base_model_path,
            torch_dtype=torch.float16,
            scheduler=noise_scheduler,
            vae=vae,
            feature_extractor=None,
            safety_checker=None,
        )
    else:
        pipe = TimeStableDiffusionPipeline.from_pretrained(
            args.base_model_path,
            torch_dtype=torch.float16,
            scheduler=noise_scheduler,
            feature_extractor=None,
            safety_checker=None,
        )
    ip_model = TimeIPAdapter(
        pipe, args.image_encoder_path, args.ip_ckpt, args.device, time_attention=True
    )

    MODEL_LIST = [
        "constant_05",
        "constant_10",
        "linear",
        "relu_100",
        "relu_500",
        "relu_900",
        "sigmoid",
        "sine",
    ]

    if args.model == 0:
        MODEL_LIST = MODEL_LIST[:4]
    elif args.model == 1:
        MODEL_LIST = MODEL_LIST[4:]

    input_file_path = args.dataset_path + "{0}/00.jpg"
    output_dir = "/{0}/{1}/"
    total_images = (
        len(TARGET2CLASS_MAPPING)
        * len(OBJECT_PROMPT_LIST)
        * len(MODEL_LIST)
        * args.num_samples
    )
    progress_bar = tqdm(total=total_images, desc="Generating Images", initial=0)

    for target, class_type in TARGET2CLASS_MAPPING.items():
        input_target_file_path = input_file_path.format(target)
        image = Image.open(input_target_file_path)

        if CLASS2OBJECT_MAPPING[class_type]:
            prompt_ls = OBJECT_PROMPT_LIST
        else:
            prompt_ls = LIVE_PROMPT_LIST

        for prompt_idx, prompt in enumerate(prompt_ls):
            prompt = prompt.format(class_type)

            for model in MODEL_LIST:
                target_output_dir = args.output_dir + output_dir.format(model, target)
                if not os.path.exists(target_output_dir):
                    os.makedirs(target_output_dir)

                images = ip_model.generate(
                    pil_image=image,
                    num_samples=args.num_samples,
                    num_inference_steps=50,
                    seed=args.seed,
                    prompt=prompt,
                    only_scale=model,
                )

                for image_idx in range(args.num_samples):
                    image_file_name = (
                        f"prompt_{prompt_idx:02d}_image_{image_idx:02d}.jpg"
                    )
                    images[image_idx].save(target_output_dir + image_file_name)
                progress_bar.update(args.num_samples)

    progress_bar.close()


if __name__ == "__main__":
    args = parse_args()
    if args.store_config:
        create_config_file(args)
    main(args)
