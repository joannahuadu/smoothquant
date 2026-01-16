import torch
from torch import nn
from functools import partial


@torch.no_grad()
def quantize_weight_per_channel_absmax(w, n_bits=8):
    # w: (out_features, in_features)
    scales = w.abs().max(dim=-1, keepdim=True)[0]
    q_max = 2 ** (n_bits - 1) - 1
    scales.clamp_(min=1e-5).div_(q_max)
    w.div_(scales).round_().mul_(scales)
    return w


@torch.no_grad()
def quantize_weight_per_tensor_absmax(w, n_bits=8):
    # w: (out_features, in_features)
    scales = w.abs().max()
    q_max = 2 ** (n_bits - 1) - 1
    scales.clamp_(min=1e-5).div_(q_max)
    w.div_(scales).round_().mul_(scales)
    return w


@torch.no_grad()
def quantize_activation_per_token_absmax(t, n_bits=8):
    t_shape = t.shape
    t.view(-1, t_shape[-1])
    scales = t.abs().max(dim=-1, keepdim=True)[0]
    q_max = 2 ** (n_bits - 1) - 1
    scales.clamp_(min=1e-5).div_(q_max)
    t.div_(scales).round_().mul_(scales)
    return t


@torch.no_grad()
def quantize_activation_per_tensor_absmax(t, n_bits=8):
    t_shape = t.shape
    t.view(-1, t_shape[-1])
    scales = t.abs().max()
    q_max = 2 ** (n_bits - 1) - 1
    scales.clamp_(min=1e-5).div_(q_max)
    t.div_(scales).round_().mul_(scales)
    return t


@torch.no_grad()
def quantize_activation_per_channel_absmax_static(t, act_scale, n_bits=8):
    t_shape = t.shape
    t.view(-1, t_shape[-1])
    q_max = 2 ** (n_bits - 1) - 1
    scales = act_scale.to(device=t.device, dtype=t.dtype)
    scales = scales.clamp(min=1e-5).div(q_max).view(1, -1)
    t.div_(scales).round_().mul_(scales)
    return t


class W8A8Linear(nn.Module):
    def __init__(
        self,
        in_features,
        out_features,
        bias=True,
        act_quant="per_token",
        quantize_output=False,
        act_scale=None,
        act_sparsity_n=2,
        act_sparsity_m=4,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.act_sparsity_n = act_sparsity_n
        self.act_sparsity_m = act_sparsity_m

        self.register_buffer(
            "weight",
            torch.randn(
                self.out_features,
                self.in_features,
                dtype=torch.float16,
                requires_grad=False,
            ),
        )
        if bias:
            self.register_buffer(
                "bias",
                torch.zeros(
                    (1, self.out_features), dtype=torch.float16, requires_grad=False
                ),
            )
        else:
            self.register_buffer("bias", None)

        if act_quant == "per_token":
            self.act_quant_name = "per_token"
            self.act_quant = partial(quantize_activation_per_token_absmax, n_bits=8)
        elif act_quant == "per_tensor":
            self.act_quant_name = "per_tensor"
            self.act_quant = partial(quantize_activation_per_tensor_absmax, n_bits=8)
        elif act_quant == "per_channel_static":
            if act_scale is None:
                raise ValueError("act_scale is required for per_channel_static")
            self.act_quant_name = "per_channel_static"
            self.act_quant = partial(
                quantize_activation_per_channel_absmax_static,
                act_scale=act_scale,
                n_bits=8,
            )
        else:
            raise ValueError(f"Invalid act_quant: {act_quant}")

        if quantize_output:
            self.output_quant_name = self.act_quant_name
            self.output_quant = self.act_quant
        else:
            self.output_quant_name = "None"
            self.output_quant = lambda x: x

    def to(self, *args, **kwargs):
        super(W8A8Linear, self).to(*args, **kwargs)
        self.weight = self.weight.to(*args, **kwargs)
        if self.bias is not None:
            self.bias = self.bias.to(*args, **kwargs)
        return self

    @torch.no_grad()
    def forward(self, x):
        if self.act_sparsity_n and self.act_sparsity_m:
            x = self.apply_activation_sparsity(x)
        q_x = self.act_quant(x)
        y = torch.functional.F.linear(q_x, self.weight, self.bias)
        q_y = self.output_quant(y)
        return q_y

    @torch.no_grad()
    def apply_activation_sparsity(self, x):
        x_shape = x.shape
        x_2d = x.view(-1, x_shape[-1])
        w_col_norm = self.weight.float().pow(2).sum(dim=0).sqrt()
        min_norm = w_col_norm.min().clamp(min=1e-5)
        scale = (w_col_norm / min_norm).view(1, -1)
        metric = x_2d.abs().float() * scale

        mask = torch.zeros_like(metric, dtype=torch.bool)
        for ii in range(0, metric.shape[1], self.act_sparsity_m):
            group_size = min(self.act_sparsity_m, metric.shape[1] - ii)
            if group_size < self.act_sparsity_n:
                continue
            tmp = metric[:, ii : ii + group_size]
            idx = torch.topk(tmp, self.act_sparsity_n, dim=1, largest=False)[1]
            mask.scatter_(1, ii + idx, True)

        x_2d = x_2d.masked_fill(mask, 0)
        return x_2d.view(x_shape)

    @staticmethod
    def from_float(
        module,
        weight_quant="per_channel",
        act_quant="per_token",
        quantize_output=False,
        act_scale=None,
        act_sparsity_n=2,
        act_sparsity_m=4,
    ):
        assert isinstance(module, torch.nn.Linear)
        new_module = W8A8Linear(
            module.in_features,
            module.out_features,
            module.bias is not None,
            act_quant=act_quant,
            quantize_output=quantize_output,
            act_scale=act_scale,
            act_sparsity_n=act_sparsity_n,
            act_sparsity_m=act_sparsity_m,
        )
        if weight_quant == "per_channel":
            new_module.weight = quantize_weight_per_channel_absmax(
                module.weight, n_bits=8
            )  # use 8-bit integer for weight
        elif weight_quant == "per_tensor":
            new_module.weight = quantize_weight_per_tensor_absmax(
                module.weight, n_bits=8
            )
        else:
            raise ValueError(f"Invalid weight_quant: {weight_quant}")
        new_module.weight_quant_name = weight_quant
        if module.bias is not None:
            new_module.bias = module.bias
        return new_module

    def __repr__(self):
        return f"W8A8Linear({self.in_features}, {self.out_features}, bias={self.bias is not None}, weight_quant={self.weight_quant_name}, act_quant={self.act_quant_name}, output_quant={self.output_quant_name})"


def quantize_opt(
    model,
    weight_quant="per_tensor",
    act_quant="per_tensor",
    quantize_bmm_input=True,
    act_sparsity_n=0,
    act_sparsity_m=0,
):
    from transformers.models.opt.modeling_opt import (
        OPTAttention,
        OPTDecoderLayer,
    )

    for name, m in model.model.named_modules():
        if isinstance(m, OPTDecoderLayer):
            m.fc1 = W8A8Linear.from_float(
                m.fc1,
                weight_quant=weight_quant,
                act_quant=act_quant,
                act_sparsity_n=act_sparsity_n,
                act_sparsity_m=act_sparsity_m,
            )
            m.fc2 = W8A8Linear.from_float(
                m.fc2,
                weight_quant=weight_quant,
                act_quant=act_quant,
                act_sparsity_n=act_sparsity_n,
                act_sparsity_m=act_sparsity_m,
            )
        elif isinstance(m, OPTAttention):
            # Her we simulate quantizing BMM inputs by quantizing the output of q_proj, k_proj, v_proj
            m.q_proj = W8A8Linear.from_float(
                m.q_proj,
                weight_quant=weight_quant,
                act_quant=act_quant,
                quantize_output=quantize_bmm_input,
                act_sparsity_n=act_sparsity_n,
                act_sparsity_m=act_sparsity_m,
            )
            m.k_proj = W8A8Linear.from_float(
                m.k_proj,
                weight_quant=weight_quant,
                act_quant=act_quant,
                quantize_output=quantize_bmm_input,
                act_sparsity_n=act_sparsity_n,
                act_sparsity_m=act_sparsity_m,
            )
            m.v_proj = W8A8Linear.from_float(
                m.v_proj,
                weight_quant=weight_quant,
                act_quant=act_quant,
                quantize_output=quantize_bmm_input,
                act_sparsity_n=act_sparsity_n,
                act_sparsity_m=act_sparsity_m,
            )
            m.out_proj = W8A8Linear.from_float(
                m.out_proj,
                weight_quant=weight_quant,
                act_quant=act_quant,
                act_sparsity_n=act_sparsity_n,
                act_sparsity_m=act_sparsity_m,
            )
    return model


def quantize_llama_like(
    model,
    weight_quant="per_channel",
    act_quant="per_token",
    quantize_bmm_input=False,
    act_scales=None,
    act_sparsity_n=0,
    act_sparsity_m=0,
):
    from transformers.models.llama.modeling_llama import (
        LlamaAttention,
        LlamaMLP,
    )

    from transformers.models.mistral.modeling_mistral import (
        MistralAttention,
        MistralMLP,
    )

    def get_act_scale(module_name):
        if act_scales is None:
            return None
        if module_name in act_scales:
            return act_scales[module_name]
        full_name = f"model.{module_name}"
        if full_name in act_scales:
            return act_scales[full_name]
        return None

    for name, m in model.model.named_modules():
        if isinstance(m, (LlamaMLP, MistralMLP)):
            m.gate_proj = W8A8Linear.from_float(
                m.gate_proj,
                weight_quant=weight_quant,
                act_quant=act_quant,
                act_scale=get_act_scale(f"{name}.gate_proj"),
                act_sparsity_n=act_sparsity_n,
                act_sparsity_m=act_sparsity_m,
            )
            m.up_proj = W8A8Linear.from_float(
                m.up_proj,
                weight_quant=weight_quant,
                act_quant=act_quant,
                act_scale=get_act_scale(f"{name}.up_proj"),
                act_sparsity_n=act_sparsity_n,
                act_sparsity_m=act_sparsity_m,
            )
            m.down_proj = W8A8Linear.from_float(
                m.down_proj,
                weight_quant=weight_quant,
                act_quant=act_quant,
                act_scale=get_act_scale(f"{name}.down_proj"),
                act_sparsity_n=act_sparsity_n,
                act_sparsity_m=act_sparsity_m,
            )
        elif isinstance(m, (LlamaAttention, MistralAttention)):
            # Her we simulate quantizing BMM inputs by quantizing the output of q_proj, k_proj, v_proj
            m.q_proj = W8A8Linear.from_float(
                m.q_proj,
                weight_quant=weight_quant,
                act_quant=act_quant,
                quantize_output=quantize_bmm_input,
                act_scale=get_act_scale(f"{name}.q_proj"),
                act_sparsity_n=act_sparsity_n,
                act_sparsity_m=act_sparsity_m,
            )
            m.k_proj = W8A8Linear.from_float(
                m.k_proj,
                weight_quant=weight_quant,
                act_quant=act_quant,
                quantize_output=quantize_bmm_input,
                act_scale=get_act_scale(f"{name}.k_proj"),
                act_sparsity_n=act_sparsity_n,
                act_sparsity_m=act_sparsity_m,
            )
            m.v_proj = W8A8Linear.from_float(
                m.v_proj,
                weight_quant=weight_quant,
                act_quant=act_quant,
                quantize_output=quantize_bmm_input,
                act_scale=get_act_scale(f"{name}.v_proj"),
                act_sparsity_n=act_sparsity_n,
                act_sparsity_m=act_sparsity_m,
            )
            m.o_proj = W8A8Linear.from_float(
                m.o_proj,
                weight_quant=weight_quant,
                act_quant=act_quant,
                act_scale=get_act_scale(f"{name}.o_proj"),
                act_sparsity_n=act_sparsity_n,
                act_sparsity_m=act_sparsity_m,
            )
    return model


def quantize_mixtral(
    model,
    weight_quant="per_channel",
    act_quant="per_token",
    quantize_bmm_input=False,
    act_sparsity_n=0,
    act_sparsity_m=0,
):
    from transformers.models.mixtral.modeling_mixtral import (
        MixtralAttention,
        MixtralSparseMoeBlock,
        MixtralBLockSparseTop2MLP,
    )

    for name, m in model.model.named_modules():
        if isinstance(m, MixtralBLockSparseTop2MLP):
            m.w1 = W8A8Linear.from_float(
                m.w1,
                weight_quant=weight_quant,
                act_quant=act_quant,
                act_sparsity_n=act_sparsity_n,
                act_sparsity_m=act_sparsity_m,
            )
            m.w2 = W8A8Linear.from_float(
                m.w2,
                weight_quant=weight_quant,
                act_quant=act_quant,
                act_sparsity_n=act_sparsity_n,
                act_sparsity_m=act_sparsity_m,
            )
            m.w3 = W8A8Linear.from_float(
                m.w3,
                weight_quant=weight_quant,
                act_quant=act_quant,
                act_sparsity_n=act_sparsity_n,
                act_sparsity_m=act_sparsity_m,
            )
        elif isinstance(m, MixtralAttention):
            # Her we simulate quantizing BMM inputs by quantizing the output of q_proj, k_proj, v_proj
            m.q_proj = W8A8Linear.from_float(
                m.q_proj,
                weight_quant=weight_quant,
                act_quant=act_quant,
                quantize_output=quantize_bmm_input,
                act_sparsity_n=act_sparsity_n,
                act_sparsity_m=act_sparsity_m,
            )
            m.k_proj = W8A8Linear.from_float(
                m.k_proj,
                weight_quant=weight_quant,
                act_quant=act_quant,
                quantize_output=quantize_bmm_input,
                act_sparsity_n=act_sparsity_n,
                act_sparsity_m=act_sparsity_m,
            )
            m.v_proj = W8A8Linear.from_float(
                m.v_proj,
                weight_quant=weight_quant,
                act_quant=act_quant,
                quantize_output=quantize_bmm_input,
                act_sparsity_n=act_sparsity_n,
                act_sparsity_m=act_sparsity_m,
            )
            m.o_proj = W8A8Linear.from_float(
                m.o_proj,
                weight_quant=weight_quant,
                act_quant=act_quant,
                act_sparsity_n=act_sparsity_n,
                act_sparsity_m=act_sparsity_m,
            )
        elif isinstance(m, MixtralSparseMoeBlock):
            m.gate = W8A8Linear.from_float(
                m.gate,
                weight_quant=weight_quant,
                act_quant=act_quant,
                act_sparsity_n=act_sparsity_n,
                act_sparsity_m=act_sparsity_m,
            )
    return model


def quantize_falcon(
    model,
    weight_quant="per_channel",
    act_quant="per_token",
    quantize_bmm_input=True,
    act_sparsity_n=0,
    act_sparsity_m=0,
):
    from transformers.models.falcon.modeling_falcon import (
        FalconAttention,
        FalconMLP,
    )

    for name, m in model.named_modules():
        if isinstance(m, FalconMLP):
            m.dense_h_to_4h = W8A8Linear.from_float(
                m.dense_h_to_4h,
                weight_quant=weight_quant,
                act_quant=act_quant,
                act_sparsity_n=act_sparsity_n,
                act_sparsity_m=act_sparsity_m,
            )
            m.dense_4h_to_h = W8A8Linear.from_float(
                m.dense_4h_to_h,
                weight_quant=weight_quant,
                act_quant=act_quant,
                act_sparsity_n=act_sparsity_n,
                act_sparsity_m=act_sparsity_m,
            )
        elif isinstance(m, FalconAttention):
            # Her we simulate quantizing BMM inputs by quantizing the output of q_proj, k_proj, v_proj
            m.query_key_value = W8A8Linear.from_float(
                m.query_key_value,
                weight_quant=weight_quant,
                act_quant=act_quant,
                quantize_output=quantize_bmm_input,
                act_sparsity_n=act_sparsity_n,
                act_sparsity_m=act_sparsity_m,
            )
            m.dense = W8A8Linear.from_float(
                m.dense,
                weight_quant=weight_quant,
                act_quant=act_quant,
                act_sparsity_n=act_sparsity_n,
                act_sparsity_m=act_sparsity_m,
            )
    return model


def quantize_model(
    model,
    weight_quant="per_channel",
    act_quant="per_token",
    quantize_bmm_input=False,
    act_sparsity_n=0,
    act_sparsity_m=0,
):
    from transformers.models.opt.modeling_opt import OPTPreTrainedModel
    from transformers.models.llama.modeling_llama import LlamaPreTrainedModel
    from transformers.models.mistral.modeling_mistral import MistralPreTrainedModel
    from transformers.models.mixtral.modeling_mixtral import MixtralPreTrainedModel
    from transformers.models.falcon.modeling_falcon import FalconPreTrainedModel

    if isinstance(model, OPTPreTrainedModel):
        return quantize_opt(
            model,
            weight_quant=weight_quant,
            act_quant=act_quant,
            quantize_bmm_input=quantize_bmm_input,
            act_sparsity_n=act_sparsity_n,
            act_sparsity_m=act_sparsity_m,
        )
    elif isinstance(model, (LlamaPreTrainedModel, MistralPreTrainedModel)):
        return quantize_llama_like(
            model,
            weight_quant=weight_quant,
            act_quant=act_quant,
            quantize_bmm_input=quantize_bmm_input,
            act_sparsity_n=act_sparsity_n,
            act_sparsity_m=act_sparsity_m,
        )
    elif isinstance(model, MixtralPreTrainedModel):
        return quantize_mixtral(
            model,
            weight_quant=weight_quant,
            act_quant=act_quant,
            quantize_bmm_input=quantize_bmm_input,
            act_sparsity_n=act_sparsity_n,
            act_sparsity_m=act_sparsity_m,
        )
    elif isinstance(model, FalconPreTrainedModel):
        return quantize_falcon(
            model,
            weight_quant=weight_quant,
            act_quant=act_quant,
            quantize_bmm_input=quantize_bmm_input,
            act_sparsity_n=act_sparsity_n,
            act_sparsity_m=act_sparsity_m,
        )
    else:
        raise ValueError(f"Unsupported model type: {type(model)}")
