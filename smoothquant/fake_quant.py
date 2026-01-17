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
        w_bits=8,
        a_bits=8,
        act_quant="per_token",
        quantize_output=False,
        act_scale=None,
        act_sparsity_n=2,
        act_sparsity_m=4,
        act_sparsity_location="pre_quant",
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.w_bits = w_bits
        self.a_bits = a_bits
        self.act_sparsity_n = act_sparsity_n
        self.act_sparsity_m = act_sparsity_m
        self.act_sparsity_location = act_sparsity_location
        if self.act_sparsity_location not in ("pre_quant", "post_quant", "pre_smooth"):
            raise ValueError(
                f"Invalid act_sparsity_location: {self.act_sparsity_location}"
            )

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
            self.act_quant = partial(quantize_activation_per_token_absmax, n_bits=a_bits)
        elif act_quant == "per_tensor":
            self.act_quant_name = "per_tensor"
            self.act_quant = partial(quantize_activation_per_tensor_absmax, n_bits=a_bits)
        elif act_quant == "per_channel_static":
            if act_scale is None:
                raise ValueError("act_scale is required for per_channel_static")
            self.act_quant_name = "per_channel_static"
            self.act_quant = partial(
                quantize_activation_per_channel_absmax_static,
                act_scale=act_scale,
                n_bits=a_bits,
            )
        else:
            raise ValueError(f"Invalid act_quant: {act_quant}")

        if quantize_output:
            self.output_quant_name = self.act_quant_name
            self.output_quant = self.act_quant
        else:
            self.output_quant_name = "None"
            self.output_quant = lambda x: x

        self.register_buffer("sparsity_scale", None)

    def to(self, *args, **kwargs):
        super(W8A8Linear, self).to(*args, **kwargs)
        self.weight = self.weight.to(*args, **kwargs)
        if self.bias is not None:
            self.bias = self.bias.to(*args, **kwargs)
        return self

    @torch.no_grad()
    def forward(self, x):
        if (
            self.act_sparsity_n
            and self.act_sparsity_m
            and self.act_sparsity_location == "pre_quant"
        ):
            x = self.apply_activation_sparsity(x)
        q_x = self.act_quant(x)
        if (
            self.act_sparsity_n
            and self.act_sparsity_m
            and self.act_sparsity_location == "post_quant"
        ):
            q_x = self.apply_activation_sparsity(q_x)
        y = torch.functional.F.linear(q_x, self.weight, self.bias)
        q_y = self.output_quant(y)
        return q_y

    @torch.no_grad()
    def apply_activation_sparsity(self, x):
        x_shape = x.shape
        x_2d = x.view(-1, x_shape[-1])
        if self.sparsity_scale is None:
            raise ValueError("sparsity_scale is not set. Ensure from_float is used to create the module.")
        else:
            scale = self.sparsity_scale

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
        w_bits=8,
        a_bits=8,
        act_quant="per_token",
        quantize_output=False,
        act_scale=None,
        act_sparsity_n=2,
        act_sparsity_m=4,
        weight_scoring=True,
        act_sparsity_location="pre_quant",
    ):
        assert isinstance(module, torch.nn.Linear)
        new_module = W8A8Linear(
            module.in_features,
            module.out_features,
            module.bias is not None,
            w_bits=w_bits,
            a_bits=a_bits,
            act_quant=act_quant,
            quantize_output=quantize_output,
            act_scale=act_scale,
            act_sparsity_n=act_sparsity_n,
            act_sparsity_m=act_sparsity_m,
            act_sparsity_location=act_sparsity_location,
        )
        if weight_quant == "per_channel":
            new_module.weight = quantize_weight_per_channel_absmax(
                module.weight, n_bits=w_bits
            )  # use configurable bit width for weight
        elif weight_quant == "per_tensor":
            new_module.weight = quantize_weight_per_tensor_absmax(
                module.weight, n_bits=w_bits
            )
        else:
            raise ValueError(f"Invalid weight_quant: {weight_quant}")
        new_module.weight_quant_name = weight_quant
        if module.bias is not None:
            new_module.bias = module.bias

        with torch.no_grad():
            weight = new_module.weight.float()
            if weight_scoring:
                weight_flat = weight.flatten()
                num_elements = weight_flat.numel()
                if num_elements > 1000000:
                    sample_size = min(100000, num_elements)
                    indices = torch.randperm(num_elements, device=weight.device)[:sample_size]
                    weight_sample = weight_flat[indices]
                    q_low = torch.quantile(weight_sample, 0.005)
                    q_high = torch.quantile(weight_sample, 0.995)
                else:
                    q_low = torch.quantile(weight_flat, 0.005)
                    q_high = torch.quantile(weight_flat, 0.995)
                within_range = (weight >= q_low) & (weight <= q_high)

                if within_range.sum() < 2:
                    weight_processed = weight
                else:
                    w_filtered = weight[within_range]
                    mean = w_filtered.mean()
                    std = w_filtered.std()

                    std = std.clamp(min=1e-8)

                    weight_processed = (weight - mean) / std
                    weight_processed = weight_processed.clamp(
                        min=(q_low - mean) / std,
                        max=(q_high - mean) / std
                    )

                w_col_norm = weight_processed.pow(2).sum(dim=0).sqrt()
            else:
                w_col_norm = weight.pow(2).sum(dim=0).sqrt()
            min_norm = w_col_norm.min().clamp(min=1e-5)
            new_module.sparsity_scale = (w_col_norm / min_norm).view(1, -1)

        return new_module

    def __repr__(self):
        return f"W{self.w_bits}A{self.a_bits}Linear({self.in_features}, {self.out_features}, bias={self.bias is not None}, weight_quant={self.weight_quant_name}, act_quant={self.act_quant_name}, output_quant={self.output_quant_name})"


def quantize_opt(
    model,
    weight_quant="per_tensor",
    w_bits=8,
    a_bits=8,
    act_quant="per_tensor",
    quantize_bmm_input=True,
    act_sparsity_n=0,
    act_sparsity_m=0,
    target_modules=None,
    weight_scoring=True,
    act_sparsity_location="pre_quant",
):
    from transformers.models.opt.modeling_opt import (
        OPTAttention,
        OPTDecoderLayer,
    )

    def should_apply_sparsity(module_name):
        """Check if sparsity should be applied to this module."""
        if target_modules is None:
            return True
        return not any(pattern in module_name for pattern in target_modules)

    for name, m in model.model.named_modules():
        if isinstance(m, OPTDecoderLayer):
            fc1_sparsity = (act_sparsity_n, act_sparsity_m) if should_apply_sparsity(f"{name}.fc1") else (0, 0)
            fc2_sparsity = (act_sparsity_n, act_sparsity_m) if should_apply_sparsity(f"{name}.fc2") else (0, 0)

            m.fc1 = W8A8Linear.from_float(
                m.fc1,
                weight_quant=weight_quant,
                w_bits=w_bits,
                a_bits=a_bits,
                act_quant=act_quant,
                act_sparsity_n=fc1_sparsity[0],
                act_sparsity_m=fc1_sparsity[1],
                weight_scoring=weight_scoring,
                act_sparsity_location=act_sparsity_location,
            )
            m.fc2 = W8A8Linear.from_float(
                m.fc2,
                weight_quant=weight_quant,
                w_bits=w_bits,
                a_bits=a_bits,
                act_quant=act_quant,
                act_sparsity_n=fc2_sparsity[0],
                act_sparsity_m=fc2_sparsity[1],
                weight_scoring=weight_scoring,
                act_sparsity_location=act_sparsity_location,
            )
        elif isinstance(m, OPTAttention):
            # Determine sparsity for each module
            q_sparsity = (act_sparsity_n, act_sparsity_m) if should_apply_sparsity(f"{name}.q_proj") else (0, 0)
            k_sparsity = (act_sparsity_n, act_sparsity_m) if should_apply_sparsity(f"{name}.k_proj") else (0, 0)
            v_sparsity = (act_sparsity_n, act_sparsity_m) if should_apply_sparsity(f"{name}.v_proj") else (0, 0)
            out_sparsity = (act_sparsity_n, act_sparsity_m) if should_apply_sparsity(f"{name}.out_proj") else (0, 0)

            # Her we simulate quantizing BMM inputs by quantizing the output of q_proj, k_proj, v_proj
            m.q_proj = W8A8Linear.from_float(
                m.q_proj,
                weight_quant=weight_quant,
                w_bits=w_bits,
                a_bits=a_bits,
                act_quant=act_quant,
                quantize_output=quantize_bmm_input,
                act_sparsity_n=q_sparsity[0],
                act_sparsity_m=q_sparsity[1],
                weight_scoring=weight_scoring,
                act_sparsity_location=act_sparsity_location,
            )
            m.k_proj = W8A8Linear.from_float(
                m.k_proj,
                weight_quant=weight_quant,
                w_bits=w_bits,
                a_bits=a_bits,
                act_quant=act_quant,
                quantize_output=quantize_bmm_input,
                act_sparsity_n=k_sparsity[0],
                act_sparsity_m=k_sparsity[1],
                weight_scoring=weight_scoring,
                act_sparsity_location=act_sparsity_location,
            )
            m.v_proj = W8A8Linear.from_float(
                m.v_proj,
                weight_quant=weight_quant,
                w_bits=w_bits,
                a_bits=a_bits,
                act_quant=act_quant,
                quantize_output=quantize_bmm_input,
                act_sparsity_n=v_sparsity[0],
                act_sparsity_m=v_sparsity[1],
                weight_scoring=weight_scoring,
                act_sparsity_location=act_sparsity_location,
            )
            m.out_proj = W8A8Linear.from_float(
                m.out_proj,
                weight_quant=weight_quant,
                w_bits=w_bits,
                a_bits=a_bits,
                act_quant=act_quant,
                act_sparsity_n=out_sparsity[0],
                act_sparsity_m=out_sparsity[1],
                weight_scoring=weight_scoring,
                act_sparsity_location=act_sparsity_location,
            )
    return model


def quantize_llama_like(
    model,
    weight_quant="per_channel",
    w_bits=8,
    a_bits=8,
    act_quant="per_token",
    quantize_bmm_input=False,
    act_scales=None,
    act_sparsity_n=0,
    act_sparsity_m=0,
    target_modules=None,
    weight_scoring=True,
    act_sparsity_location="pre_quant",
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

    def should_apply_sparsity(module_name):
        """Check if sparsity should be applied to this module."""
        if target_modules is None:
            return True
        return not any(pattern in module_name for pattern in target_modules)

    for name, m in model.model.named_modules():
        if isinstance(m, (LlamaMLP, MistralMLP)):
            # Determine sparsity for each module
            gate_sparsity = (act_sparsity_n, act_sparsity_m) if should_apply_sparsity(f"{name}.gate_proj") else (0, 0)
            up_sparsity = (act_sparsity_n, act_sparsity_m) if should_apply_sparsity(f"{name}.up_proj") else (0, 0)
            down_sparsity = (act_sparsity_n, act_sparsity_m) if should_apply_sparsity(f"{name}.down_proj") else (0, 0)

            m.gate_proj = W8A8Linear.from_float(
                m.gate_proj,
                weight_quant=weight_quant,
                w_bits=w_bits,
                a_bits=a_bits,
                act_quant=act_quant,
                act_scale=get_act_scale(f"{name}.gate_proj"),
                act_sparsity_n=gate_sparsity[0],
                act_sparsity_m=gate_sparsity[1],
                weight_scoring=weight_scoring,
                act_sparsity_location=act_sparsity_location,
            )
            m.up_proj = W8A8Linear.from_float(
                m.up_proj,
                weight_quant=weight_quant,
                w_bits=w_bits,
                a_bits=a_bits,
                act_quant=act_quant,
                act_scale=get_act_scale(f"{name}.up_proj"),
                act_sparsity_n=up_sparsity[0],
                act_sparsity_m=up_sparsity[1],
                weight_scoring=weight_scoring,
                act_sparsity_location=act_sparsity_location,
            )
            m.down_proj = W8A8Linear.from_float(
                m.down_proj,
                weight_quant=weight_quant,
                w_bits=w_bits,
                a_bits=a_bits,
                act_quant=act_quant,
                act_scale=get_act_scale(f"{name}.down_proj"),
                act_sparsity_n=down_sparsity[0],
                act_sparsity_m=down_sparsity[1],
                weight_scoring=weight_scoring,
                act_sparsity_location=act_sparsity_location,
            )
        elif isinstance(m, (LlamaAttention, MistralAttention)):
            # Determine sparsity for each module
            q_sparsity = (act_sparsity_n, act_sparsity_m) if should_apply_sparsity(f"{name}.q_proj") else (0, 0)
            k_sparsity = (act_sparsity_n, act_sparsity_m) if should_apply_sparsity(f"{name}.k_proj") else (0, 0)
            v_sparsity = (act_sparsity_n, act_sparsity_m) if should_apply_sparsity(f"{name}.v_proj") else (0, 0)
            o_sparsity = (act_sparsity_n, act_sparsity_m) if should_apply_sparsity(f"{name}.o_proj") else (0, 0)

            # Her we simulate quantizing BMM inputs by quantizing the output of q_proj, k_proj, v_proj
            m.q_proj = W8A8Linear.from_float(
                m.q_proj,
                weight_quant=weight_quant,
                w_bits=w_bits,
                a_bits=a_bits,
                act_quant=act_quant,
                quantize_output=quantize_bmm_input,
                act_scale=get_act_scale(f"{name}.q_proj"),
                act_sparsity_n=q_sparsity[0],
                act_sparsity_m=q_sparsity[1],
                weight_scoring=weight_scoring,
                act_sparsity_location=act_sparsity_location,
            )
            m.k_proj = W8A8Linear.from_float(
                m.k_proj,
                weight_quant=weight_quant,
                w_bits=w_bits,
                a_bits=a_bits,
                act_quant=act_quant,
                quantize_output=quantize_bmm_input,
                act_scale=get_act_scale(f"{name}.k_proj"),
                act_sparsity_n=k_sparsity[0],
                act_sparsity_m=k_sparsity[1],
                weight_scoring=weight_scoring,
                act_sparsity_location=act_sparsity_location,
            )
            m.v_proj = W8A8Linear.from_float(
                m.v_proj,
                weight_quant=weight_quant,
                w_bits=w_bits,
                a_bits=a_bits,
                act_quant=act_quant,
                quantize_output=quantize_bmm_input,
                act_scale=get_act_scale(f"{name}.v_proj"),
                act_sparsity_n=v_sparsity[0],
                act_sparsity_m=v_sparsity[1],
                weight_scoring=weight_scoring,
                act_sparsity_location=act_sparsity_location,
            )
            m.o_proj = W8A8Linear.from_float(
                m.o_proj,
                weight_quant=weight_quant,
                w_bits=w_bits,
                a_bits=a_bits,
                act_quant=act_quant,
                act_scale=get_act_scale(f"{name}.o_proj"),
                act_sparsity_n=o_sparsity[0],
                act_sparsity_m=o_sparsity[1],
                weight_scoring=weight_scoring,
                act_sparsity_location=act_sparsity_location,
            )
    return model


def quantize_mixtral(
    model,
    weight_quant="per_channel",
    w_bits=8,
    a_bits=8,
    act_quant="per_token",
    quantize_bmm_input=False,
    act_sparsity_n=0,
    act_sparsity_m=0,
    target_modules=None,
    weight_scoring=True,
    act_sparsity_location="pre_quant",
):
    from transformers.models.mixtral.modeling_mixtral import (
        MixtralAttention,
        MixtralSparseMoeBlock,
        MixtralBLockSparseTop2MLP,
    )

    def should_apply_sparsity(module_name):
        """Check if sparsity should be applied to this module."""
        if target_modules is None:
            return True
        return not any(pattern in module_name for pattern in target_modules)

    for name, m in model.model.named_modules():
        if isinstance(m, MixtralBLockSparseTop2MLP):
            w1_sparsity = (act_sparsity_n, act_sparsity_m) if should_apply_sparsity(f"{name}.w1") else (0, 0)
            w2_sparsity = (act_sparsity_n, act_sparsity_m) if should_apply_sparsity(f"{name}.w2") else (0, 0)
            w3_sparsity = (act_sparsity_n, act_sparsity_m) if should_apply_sparsity(f"{name}.w3") else (0, 0)

            m.w1 = W8A8Linear.from_float(
                m.w1,
                weight_quant=weight_quant,
                w_bits=w_bits,
                a_bits=a_bits,
                act_quant=act_quant,
                act_sparsity_n=w1_sparsity[0],
                act_sparsity_m=w1_sparsity[1],
                weight_scoring=weight_scoring,
                act_sparsity_location=act_sparsity_location,
            )
            m.w2 = W8A8Linear.from_float(
                m.w2,
                weight_quant=weight_quant,
                w_bits=w_bits,
                a_bits=a_bits,
                act_quant=act_quant,
                act_sparsity_n=w2_sparsity[0],
                act_sparsity_m=w2_sparsity[1],
                weight_scoring=weight_scoring,
                act_sparsity_location=act_sparsity_location,
            )
            m.w3 = W8A8Linear.from_float(
                m.w3,
                weight_quant=weight_quant,
                w_bits=w_bits,
                a_bits=a_bits,
                act_quant=act_quant,
                act_sparsity_n=w3_sparsity[0],
                act_sparsity_m=w3_sparsity[1],
                weight_scoring=weight_scoring,
                act_sparsity_location=act_sparsity_location,
            )
        elif isinstance(m, MixtralAttention):
            # Determine sparsity for each module
            q_sparsity = (act_sparsity_n, act_sparsity_m) if should_apply_sparsity(f"{name}.q_proj") else (0, 0)
            k_sparsity = (act_sparsity_n, act_sparsity_m) if should_apply_sparsity(f"{name}.k_proj") else (0, 0)
            v_sparsity = (act_sparsity_n, act_sparsity_m) if should_apply_sparsity(f"{name}.v_proj") else (0, 0)
            o_sparsity = (act_sparsity_n, act_sparsity_m) if should_apply_sparsity(f"{name}.o_proj") else (0, 0)

            # Her we simulate quantizing BMM inputs by quantizing the output of q_proj, k_proj, v_proj
            m.q_proj = W8A8Linear.from_float(
                m.q_proj,
                weight_quant=weight_quant,
                w_bits=w_bits,
                a_bits=a_bits,
                act_quant=act_quant,
                quantize_output=quantize_bmm_input,
                act_sparsity_n=q_sparsity[0],
                act_sparsity_m=q_sparsity[1],
                weight_scoring=weight_scoring,
                act_sparsity_location=act_sparsity_location,
            )
            m.k_proj = W8A8Linear.from_float(
                m.k_proj,
                weight_quant=weight_quant,
                w_bits=w_bits,
                a_bits=a_bits,
                act_quant=act_quant,
                quantize_output=quantize_bmm_input,
                act_sparsity_n=k_sparsity[0],
                act_sparsity_m=k_sparsity[1],
                weight_scoring=weight_scoring,
                act_sparsity_location=act_sparsity_location,
            )
            m.v_proj = W8A8Linear.from_float(
                m.v_proj,
                weight_quant=weight_quant,
                w_bits=w_bits,
                a_bits=a_bits,
                act_quant=act_quant,
                quantize_output=quantize_bmm_input,
                act_sparsity_n=v_sparsity[0],
                act_sparsity_m=v_sparsity[1],
                weight_scoring=weight_scoring,
                act_sparsity_location=act_sparsity_location,
            )
            m.o_proj = W8A8Linear.from_float(
                m.o_proj,
                weight_quant=weight_quant,
                w_bits=w_bits,
                a_bits=a_bits,
                act_quant=act_quant,
                act_sparsity_n=o_sparsity[0],
                act_sparsity_m=o_sparsity[1],
                weight_scoring=weight_scoring,
                act_sparsity_location=act_sparsity_location,
            )
        elif isinstance(m, MixtralSparseMoeBlock):
            gate_sparsity = (act_sparsity_n, act_sparsity_m) if should_apply_sparsity(f"{name}.gate") else (0, 0)
            m.gate = W8A8Linear.from_float(
                m.gate,
                weight_quant=weight_quant,
                w_bits=w_bits,
                a_bits=a_bits,
                act_quant=act_quant,
                act_sparsity_n=gate_sparsity[0],
                act_sparsity_m=gate_sparsity[1],
                weight_scoring=weight_scoring,
                act_sparsity_location=act_sparsity_location,
            )
    return model


def quantize_falcon(
    model,
    weight_quant="per_channel",
    w_bits=8,
    a_bits=8,
    act_quant="per_token",
    quantize_bmm_input=True,
    act_sparsity_n=0,
    act_sparsity_m=0,
    target_modules=None,
    weight_scoring=True,
    act_sparsity_location="pre_quant",
):
    from transformers.models.falcon.modeling_falcon import (
        FalconAttention,
        FalconMLP,
    )

    def should_apply_sparsity(module_name):
        """Check if sparsity should be applied to this module."""
        if target_modules is None:
            return True
        return not any(pattern in module_name for pattern in target_modules)

    for name, m in model.named_modules():
        if isinstance(m, FalconMLP):
            dense_h_to_4h_sparsity = (act_sparsity_n, act_sparsity_m) if should_apply_sparsity(f"{name}.dense_h_to_4h") else (0, 0)
            dense_4h_to_h_sparsity = (act_sparsity_n, act_sparsity_m) if should_apply_sparsity(f"{name}.dense_4h_to_h") else (0, 0)

            m.dense_h_to_4h = W8A8Linear.from_float(
                m.dense_h_to_4h,
                weight_quant=weight_quant,
                w_bits=w_bits,
                a_bits=a_bits,
                act_quant=act_quant,
                act_sparsity_n=dense_h_to_4h_sparsity[0],
                act_sparsity_m=dense_h_to_4h_sparsity[1],
                weight_scoring=weight_scoring,
                act_sparsity_location=act_sparsity_location,
            )
            m.dense_4h_to_h = W8A8Linear.from_float(
                m.dense_4h_to_h,
                weight_quant=weight_quant,
                w_bits=w_bits,
                a_bits=a_bits,
                act_quant=act_quant,
                act_sparsity_n=dense_4h_to_h_sparsity[0],
                act_sparsity_m=dense_4h_to_h_sparsity[1],
                weight_scoring=weight_scoring,
                act_sparsity_location=act_sparsity_location,
            )
        elif isinstance(m, FalconAttention):
            # Determine sparsity for each module
            qkv_sparsity = (act_sparsity_n, act_sparsity_m) if should_apply_sparsity(f"{name}.query_key_value") else (0, 0)
            dense_sparsity = (act_sparsity_n, act_sparsity_m) if should_apply_sparsity(f"{name}.dense") else (0, 0)

            # Her we simulate quantizing BMM inputs by quantizing the output of q_proj, k_proj, v_proj
            m.query_key_value = W8A8Linear.from_float(
                m.query_key_value,
                weight_quant=weight_quant,
                w_bits=w_bits,
                a_bits=a_bits,
                act_quant=act_quant,
                quantize_output=quantize_bmm_input,
                act_sparsity_n=qkv_sparsity[0],
                act_sparsity_m=qkv_sparsity[1],
                weight_scoring=weight_scoring,
                act_sparsity_location=act_sparsity_location,
            )
            m.dense = W8A8Linear.from_float(
                m.dense,
                weight_quant=weight_quant,
                w_bits=w_bits,
                a_bits=a_bits,
                act_quant=act_quant,
                act_sparsity_n=dense_sparsity[0],
                act_sparsity_m=dense_sparsity[1],
                weight_scoring=weight_scoring,
                act_sparsity_location=act_sparsity_location,
            )
    return model


def quantize_model(
    model,
    weight_quant="per_channel",
    w_bits=8,
    a_bits=8,
    act_quant="per_token",
    quantize_bmm_input=False,
    act_sparsity_n=0,
    act_sparsity_m=0,
    target_modules=None,
    weight_scoring=True,
    act_sparsity_location="pre_quant",
):
    from transformers.models.opt.modeling_opt import OPTPreTrainedModel
    from transformers.models.llama.modeling_llama import LlamaPreTrainedModel
    from transformers.models.mistral.modeling_mistral import MistralPreTrainedModel
    from transformers.models.mixtral.modeling_mixtral import MixtralPreTrainedModel
    from transformers.models.falcon.modeling_falcon import FalconPreTrainedModel

    if act_sparsity_location == "pre_smooth" and act_sparsity_n and act_sparsity_m:
        from smoothquant.smooth import hook_ln_sparsity

        ln_sparsity_hooks = hook_ln_sparsity(
            model,
            act_sparsity_n=act_sparsity_n,
            act_sparsity_m=act_sparsity_m,
            target_modules=target_modules,
            weight_scoring=weight_scoring,
        )
        setattr(model, "_ln_sparsity_hooks", ln_sparsity_hooks)

    if isinstance(model, OPTPreTrainedModel):
        return quantize_opt(
            model,
            weight_quant=weight_quant,
            w_bits=w_bits,
            a_bits=a_bits,
            act_quant=act_quant,
            quantize_bmm_input=quantize_bmm_input,
            act_sparsity_n=act_sparsity_n,
            act_sparsity_m=act_sparsity_m,
            target_modules=target_modules,
            weight_scoring=weight_scoring,
            act_sparsity_location=act_sparsity_location,
        )
    elif isinstance(model, (LlamaPreTrainedModel, MistralPreTrainedModel)):
        return quantize_llama_like(
            model,
            weight_quant=weight_quant,
            w_bits=w_bits,
            a_bits=a_bits,
            act_quant=act_quant,
            quantize_bmm_input=quantize_bmm_input,
            act_sparsity_n=act_sparsity_n,
            act_sparsity_m=act_sparsity_m,
            target_modules=target_modules,
            weight_scoring=weight_scoring,
            act_sparsity_location=act_sparsity_location,
        )
    elif isinstance(model, MixtralPreTrainedModel):
        return quantize_mixtral(
            model,
            weight_quant=weight_quant,
            w_bits=w_bits,
            a_bits=a_bits,
            act_quant=act_quant,
            quantize_bmm_input=quantize_bmm_input,
            act_sparsity_n=act_sparsity_n,
            act_sparsity_m=act_sparsity_m,
            target_modules=target_modules,
            weight_scoring=weight_scoring,
            act_sparsity_location=act_sparsity_location,
        )
    elif isinstance(model, FalconPreTrainedModel):
        return quantize_falcon(
            model,
            weight_quant=weight_quant,
            w_bits=w_bits,
            a_bits=a_bits,
            act_quant=act_quant,
            quantize_bmm_input=quantize_bmm_input,
            act_sparsity_n=act_sparsity_n,
            act_sparsity_m=act_sparsity_m,
            target_modules=target_modules,
            weight_scoring=weight_scoring,
            act_sparsity_location=act_sparsity_location,
        )
    else:
        raise ValueError(f"Unsupported model type: {type(model)}")
