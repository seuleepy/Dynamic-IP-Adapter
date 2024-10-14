# modified from https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py
import torch
import torch.nn as nn
import torch.nn.functional as F


class TimeIPAttnProcessor(nn.Module):
    r"""
    Attention processor for IP-Adapater.
    Args:
        hidden_size (`int`):
            The hidden size of the attention layer.
        cross_attention_dim (`int`):
            The number of channels in the `encoder_hidden_states`.
        scale (`float`, defaults to 1.0):
            the weight scale of image prompt.
        num_tokens (`int`, defaults to 4 when do ip_adapter_plus it should be 16):
            The context length of the image features.
    """

    global_inference_steps = []
    global_text_weights = []
    global_ip_weights = []

    def __init__(self, hidden_size, cross_attention_dim=None, scale=1.0, num_tokens=4):
        super().__init__()

        self.hidden_size = hidden_size
        self.cross_attention_dim = cross_attention_dim
        self.scale = scale
        self.num_tokens = num_tokens

        self.to_k_ip = nn.Linear(
            cross_attention_dim or hidden_size, hidden_size, bias=False
        )
        self.to_v_ip = nn.Linear(
            cross_attention_dim or hidden_size, hidden_size, bias=False
        )

    def __call__(
        self,
        attn,
        hidden_states,
        encoder_hidden_states=None,
        attention_mask=None,
        temb=None,
        *args,
        **kwargs,
    ):
        residual = hidden_states

        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim

        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(
                batch_size, channel, height * width
            ).transpose(1, 2)

        batch_size, sequence_length, _ = (
            hidden_states.shape
            if encoder_hidden_states is None
            else encoder_hidden_states.shape
        )
        attention_mask = attn.prepare_attention_mask(
            attention_mask, sequence_length, batch_size
        )

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(
                1, 2
            )

        query = attn.to_q(hidden_states)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        else:
            # get encoder_hidden_states, ip_hidden_states
            end_pos = encoder_hidden_states.shape[1] - self.num_tokens
            encoder_hidden_states, ip_hidden_states = (
                encoder_hidden_states[:, :end_pos, :],
                encoder_hidden_states[:, end_pos:, :],
            )
            if attn.norm_cross:
                encoder_hidden_states = attn.norm_encoder_hidden_states(
                    encoder_hidden_states
                )

        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        query = attn.head_to_batch_dim(query)
        key = attn.head_to_batch_dim(key)
        value = attn.head_to_batch_dim(value)

        attention_probs = attn.get_attention_scores(query, key, attention_mask)
        hidden_states = torch.bmm(attention_probs, value)
        hidden_states = attn.batch_to_head_dim(hidden_states)

        # for ip-adapter
        ip_key = self.to_k_ip(ip_hidden_states)
        ip_value = self.to_v_ip(ip_hidden_states)

        ip_key = attn.head_to_batch_dim(ip_key)
        ip_value = attn.head_to_batch_dim(ip_value)

        ip_attention_probs = attn.get_attention_scores(query, ip_key, None)
        self.attn_map = ip_attention_probs
        ip_hidden_states = torch.bmm(ip_attention_probs, ip_value)
        ip_hidden_states = attn.batch_to_head_dim(ip_hidden_states)

        if "TI" in kwargs:
            inference_time_step = kwargs["inference_time_step"]
            if kwargs["TI"]:
                text_weight = inference_time_step / 1000
                ip_weight = 1 - inference_time_step / 1000
            elif not kwargs["TI"]:
                text_weight = 1 - inference_time_step / 1000
                ip_weight = inference_time_step / 1000
            TimeIPAttnProcessor2_0.global_inference_steps.append(
                inference_time_step.item()
            )
            TimeIPAttnProcessor2_0.global_text_weights.append(text_weight.item())
            TimeIPAttnProcessor2_0.global_ip_weights.append(ip_weight.item())
        elif "step" in kwargs:
            inference_time_step = kwargs["inference_time_step"]
            if kwargs["step"] == 0:
                if 801 <= inference_time_step <= 1000:
                    text_weight = kwargs["ratio"]["text"]
                    ip_weight = kwargs["ratio"]["ip"]
            elif kwargs["step"] == 1:
                if 601 <= inference_time_step <= 800:
                    text_weight = kwargs["ratio"]["text"]
                    ip_weight = kwargs["ratio"]["ip"]
            elif kwargs["step"] == 2:
                if 401 <= inference_time_step <= 600:
                    text_weight = kwargs["ratio"]["text"]
                    ip_weight = kwargs["ratio"]["ip"]
            elif kwargs["step"] == 3:
                if 201 <= inference_time_step <= 400:
                    text_weight = kwargs["ratio"]["text"]
                    ip_weight = kwargs["ratio"]["ip"]
            elif kwargs["step"] == 4:
                if 1 <= inference_time_step <= 200:
                    text_weight = kwargs["ratio"]["text"]
                    ip_weight = kwargs["ratio"]["ip"]
            try:
                text_weight
            except NameError:
                text_weight = 1
                ip_weight = 1
            TimeIPAttnProcessor2_0.global_inference_steps.append(
                inference_time_step.item()
            )
            TimeIPAttnProcessor2_0.global_text_weights.append(text_weight)
            TimeIPAttnProcessor2_0.global_ip_weights.append(ip_weight)
        elif "only_scale" in kwargs:
            inference_time_step = kwargs["inference_time_step"]
            text_weight = 1
            ip_weight = 1
            if kwargs["only_scale"] == "linear":
                self.scale = 1 - inference_time_step / 1000
            elif "relu" in kwargs["only_scale"]:
                step = int(kwargs["only_scale"][5:])
                if inference_time_step > step:
                    self.scale = 0
                else:
                    self.scale = 1 - inference_time_step / step
            elif kwargs["only_scale"] == "sigmoid":
                x = -1 / 100 * inference_time_step + 5
                # x [-5, 5]
                self.scale = 1 / (1 + torch.exp(-x))
            elif kwargs["only_scale"] == "poisson_cdf":
                x = -1 / 100 * inference_time_step + 10
                # x [0, 10]
                self.scale = 1 - torch.exp(-x)
            elif kwargs["only_scale"] == "exponential":
                x = -1 / 100 * inference_time_step
                self.scale = torch.exp(x)
            elif kwargs["only_scale"] == "sine":
                x = torch.pi / 2 * (-1 / 1000 * inference_time_step + 1)
                self.scale = torch.sin(x)
            elif kwargs["only_scale"] == "constant_05":
                self.scale = 0.5
            elif kwargs["only_scale"] == "constant_10":
                self.scale = 1.0
            TimeIPAttnProcessor2_0.global_inference_steps.append(
                inference_time_step.item()
            )
            TimeIPAttnProcessor2_0.global_text_weights.append(text_weight)
            ip_weight_ = ip_weight * self.scale
            if isinstance(ip_weight_, torch.Tensor):
                ip_weight_ = ip_weight_.item()
            TimeIPAttnProcessor2_0.global_ip_weights.append(ip_weight_)
        else:
            text_weight = 1
            ip_weight = 1

        hidden_states = (
            text_weight * hidden_states + ip_weight * self.scale * ip_hidden_states
        )

        hidden_states = (
            text_weight * hidden_states + ip_weight * self.scale * ip_hidden_states
        )

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(
                batch_size, channel, height, width
            )

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor

        return hidden_states


class TimeIPAttnProcessor2_0(torch.nn.Module):
    r"""
    Attention processor for IP-Adapater for PyTorch 2.0.
    Args:
        hidden_size (`int`):
            The hidden size of the attention layer.
        cross_attention_dim (`int`):
            The number of channels in the `encoder_hidden_states`.
        scale (`float`, defaults to 1.0):
            the weight scale of image prompt.
        num_tokens (`int`, defaults to 4 when do ip_adapter_plus it should be 16):
            The context length of the image features.
    """

    global_inference_steps = []
    global_text_weights = []
    global_ip_weights = []

    def __init__(self, hidden_size, cross_attention_dim=None, scale=1.0, num_tokens=4):
        super().__init__()

        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError(
                "AttnProcessor2_0 requires PyTorch 2.0, to use it, please upgrade PyTorch to 2.0."
            )

        self.hidden_size = hidden_size
        self.cross_attention_dim = cross_attention_dim
        self.scale = scale
        self.num_tokens = num_tokens

        self.to_k_ip = nn.Linear(
            cross_attention_dim or hidden_size, hidden_size, bias=False
        )
        self.to_v_ip = nn.Linear(
            cross_attention_dim or hidden_size, hidden_size, bias=False
        )

    def __call__(
        self,
        attn,
        hidden_states,
        encoder_hidden_states=None,
        attention_mask=None,
        temb=None,
        *args,
        **kwargs,
    ):
        residual = hidden_states

        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim

        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(
                batch_size, channel, height * width
            ).transpose(1, 2)

        batch_size, sequence_length, _ = (
            hidden_states.shape
            if encoder_hidden_states is None
            else encoder_hidden_states.shape
        )

        if attention_mask is not None:
            attention_mask = attn.prepare_attention_mask(
                attention_mask, sequence_length, batch_size
            )
            # scaled_dot_product_attention expects attention_mask shape to be
            # (batch, heads, source_length, target_length)
            attention_mask = attention_mask.view(
                batch_size, attn.heads, -1, attention_mask.shape[-1]
            )

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(
                1, 2
            )

        query = attn.to_q(hidden_states)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        else:
            # get encoder_hidden_states, ip_hidden_states
            end_pos = encoder_hidden_states.shape[1] - self.num_tokens
            encoder_hidden_states, ip_hidden_states = (
                encoder_hidden_states[:, :end_pos, :],
                encoder_hidden_states[:, end_pos:, :],
            )
            if attn.norm_cross:
                encoder_hidden_states = attn.norm_encoder_hidden_states(
                    encoder_hidden_states
                )

        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        # the output of sdp = (batch, num_heads, seq_len, head_dim)
        # TODO: add support for attn.scale when we move to Torch 2.1
        hidden_states = F.scaled_dot_product_attention(
            query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
        )

        hidden_states = hidden_states.transpose(1, 2).reshape(
            batch_size, -1, attn.heads * head_dim
        )
        hidden_states = hidden_states.to(query.dtype)

        # for ip-adapter
        ip_key = self.to_k_ip(ip_hidden_states)
        ip_value = self.to_v_ip(ip_hidden_states)

        ip_key = ip_key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        ip_value = ip_value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        # the output of sdp = (batch, num_heads, seq_len, head_dim)
        # TODO: add support for attn.scale when we move to Torch 2.1
        ip_hidden_states = F.scaled_dot_product_attention(
            query, ip_key, ip_value, attn_mask=None, dropout_p=0.0, is_causal=False
        )
        with torch.no_grad():
            self.attn_map = query @ ip_key.transpose(-2, -1).softmax(dim=-1)
            # print(self.attn_map.shape)

        ip_hidden_states = ip_hidden_states.transpose(1, 2).reshape(
            batch_size, -1, attn.heads * head_dim
        )
        ip_hidden_states = ip_hidden_states.to(query.dtype)
        if "TI" in kwargs:
            inference_time_step = kwargs["inference_time_step"]
            if kwargs["TI"]:
                text_weight = inference_time_step / 1000
                ip_weight = 1 - inference_time_step / 1000
            elif not kwargs["TI"]:
                text_weight = 1 - inference_time_step / 1000
                ip_weight = inference_time_step / 1000
            TimeIPAttnProcessor2_0.global_inference_steps.append(
                inference_time_step.item()
            )
            TimeIPAttnProcessor2_0.global_text_weights.append(text_weight.item())
            TimeIPAttnProcessor2_0.global_ip_weights.append(ip_weight.item())
        elif "step" in kwargs:
            inference_time_step = kwargs["inference_time_step"]
            if kwargs["step"] == 0:
                if 801 <= inference_time_step <= 1000:
                    text_weight = kwargs["ratio"]["text"]
                    ip_weight = kwargs["ratio"]["ip"]
            elif kwargs["step"] == 1:
                if 601 <= inference_time_step <= 800:
                    text_weight = kwargs["ratio"]["text"]
                    ip_weight = kwargs["ratio"]["ip"]
            elif kwargs["step"] == 2:
                if 401 <= inference_time_step <= 600:
                    text_weight = kwargs["ratio"]["text"]
                    ip_weight = kwargs["ratio"]["ip"]
            elif kwargs["step"] == 3:
                if 201 <= inference_time_step <= 400:
                    text_weight = kwargs["ratio"]["text"]
                    ip_weight = kwargs["ratio"]["ip"]
            elif kwargs["step"] == 4:
                if 1 <= inference_time_step <= 200:
                    text_weight = kwargs["ratio"]["text"]
                    ip_weight = kwargs["ratio"]["ip"]
            try:
                text_weight
            except NameError:
                text_weight = 1
                ip_weight = 1
            TimeIPAttnProcessor2_0.global_inference_steps.append(
                inference_time_step.item()
            )
            TimeIPAttnProcessor2_0.global_text_weights.append(text_weight)
            TimeIPAttnProcessor2_0.global_ip_weights.append(ip_weight)
        elif "only_scale" in kwargs:
            inference_time_step = kwargs["inference_time_step"]
            text_weight = 1
            ip_weight = 1
            if kwargs["only_scale"] == "linear":
                self.scale = 1 - inference_time_step / 1000
            elif "relu" in kwargs["only_scale"]:
                step = int(kwargs["only_scale"][5:])
                if inference_time_step > step:
                    self.scale = 0
                else:
                    self.scale = 1 - inference_time_step / step
            elif kwargs["only_scale"] == "sigmoid":
                x = -1 / 100 * inference_time_step + 5
                # x [-5, 5]
                self.scale = 1 / (1 + torch.exp(-x))
            elif kwargs["only_scale"] == "poisson_cdf":
                x = -1 / 100 * inference_time_step + 10
                # x [0, 10]
                self.scale = 1 - torch.exp(-x)
            elif kwargs["only_scale"] == "exponential":
                x = -1 / 100 * inference_time_step
                self.scale = torch.exp(x)
            elif kwargs["only_scale"] == "sine":
                x = torch.pi / 2 * (-1 / 1000 * inference_time_step + 1)
                self.scale = torch.sin(x)
            elif kwargs["only_scale"] == "constant_05":
                self.scale = 0.5
            elif kwargs["only_scale"] == "constant_10":
                self.scale = 1.0
            TimeIPAttnProcessor2_0.global_inference_steps.append(
                inference_time_step.item()
            )
            TimeIPAttnProcessor2_0.global_text_weights.append(text_weight)
            ip_weight_ = ip_weight * self.scale
            if isinstance(ip_weight_, torch.Tensor):
                ip_weight_ = ip_weight_.item()
            TimeIPAttnProcessor2_0.global_ip_weights.append(ip_weight_)
        else:
            text_weight = 1
            ip_weight = 1

        hidden_states = (
            text_weight * hidden_states + ip_weight * self.scale * ip_hidden_states
        )

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(
                batch_size, channel, height, width
            )

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor

        return hidden_states
