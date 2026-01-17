import torch
import torch.nn as nn

from transformers.models.opt.modeling_opt import OPTDecoderLayer
from transformers.models.bloom.modeling_bloom import BloomBlock
from transformers.models.llama.modeling_llama import LlamaDecoderLayer, LlamaRMSNorm
from transformers.models.mistral.modeling_mistral import (
    MistralDecoderLayer,
    MistralRMSNorm,
)
from transformers.models.mixtral.modeling_mixtral import (
    MixtralDecoderLayer,
    MixtralRMSNorm,
)
from transformers.models.falcon.modeling_falcon import FalconDecoderLayer


@torch.no_grad()
def smooth_ln_fcs(ln, fcs, act_scales, alpha=0.5, invert_scales=False):
    if not isinstance(fcs, list):
        fcs = [fcs]
    assert isinstance(ln, nn.LayerNorm)
    for fc in fcs:
        assert isinstance(fc, nn.Linear)
        assert ln.weight.numel() == fc.in_features == act_scales.numel()

    device, dtype = fcs[0].weight.device, fcs[0].weight.dtype
    act_scales = act_scales.to(device=device, dtype=dtype)
    weight_scales = torch.cat(
        [fc.weight.abs().max(dim=0, keepdim=True)[0] for fc in fcs], dim=0
    )
    weight_scales = weight_scales.max(dim=0)[0].clamp(min=1e-5)

    scales = (
        (act_scales.pow(alpha) / weight_scales.pow(1 - alpha))
        .clamp(min=1e-5)
        .to(device)
        .to(dtype)
    )
    if invert_scales:
        scales = 1.0 / scales

    ln.weight.div_(scales)
    ln.bias.div_(scales)

    for fc in fcs:
        fc.weight.mul_(scales.view(1, -1))


@torch.no_grad()
def smooth_ln_fcs_llama_like(ln, fcs, act_scales, alpha=0.5, invert_scales=False):
    if not isinstance(fcs, list):
        fcs = [fcs]
    assert isinstance(ln, (LlamaRMSNorm, MistralRMSNorm, MixtralRMSNorm))
    for fc in fcs:
        assert isinstance(fc, nn.Linear)
        assert ln.weight.numel() == fc.in_features == act_scales.numel()
    device, dtype = fcs[0].weight.device, fcs[0].weight.dtype
    act_scales = act_scales.to(device=device, dtype=dtype)
    weight_scales = torch.cat(
        [fc.weight.abs().max(dim=0, keepdim=True)[0] for fc in fcs], dim=0
    )
    weight_scales = weight_scales.max(dim=0)[0].clamp(min=1e-5)
    scales = (
        (act_scales.pow(alpha) / weight_scales.pow(1 - alpha))
        .clamp(min=1e-5)
        .to(device)
        .to(dtype)
    )
    if invert_scales:
        scales = 1.0 / scales

    ln.weight.div_(scales)
    for fc in fcs:
        fc.weight.mul_(scales.view(1, -1))


@torch.no_grad()
def smooth_lm(model, scales, alpha=0.5, invert_scales=False):
    for name, module in model.named_modules():
        if isinstance(module, OPTDecoderLayer):
            attn_ln = module.self_attn_layer_norm
            qkv = [
                module.self_attn.q_proj,
                module.self_attn.k_proj,
                module.self_attn.v_proj,
            ]
            qkv_input_scales = scales[name + ".self_attn.q_proj"]
            smooth_ln_fcs(attn_ln, qkv, qkv_input_scales, alpha, invert_scales)

            ffn_ln = module.final_layer_norm
            fc1 = module.fc1
            fc1_input_scales = scales[name + ".fc1"]
            smooth_ln_fcs(ffn_ln, fc1, fc1_input_scales, alpha, invert_scales)
        elif isinstance(module, BloomBlock):
            attn_ln = module.input_layernorm
            qkv = module.self_attention.query_key_value
            qkv_input_scales = scales[name + ".self_attention.query_key_value"]
            smooth_ln_fcs(attn_ln, qkv, qkv_input_scales, alpha, invert_scales)

            ffn_ln = module.post_attention_layernorm
            fc1 = module.mlp.dense_h_to_4h
            fc1_input_scales = scales[name + ".mlp.dense_h_to_4h"]
            smooth_ln_fcs(ffn_ln, fc1, fc1_input_scales, alpha, invert_scales)
        elif isinstance(module, FalconDecoderLayer):
            qkv = module.self_attention.query_key_value
            qkv_input_scales = scales[name + ".self_attention.query_key_value"]
            fc1_input_scales = scales[name + ".mlp.dense_h_to_4h"]
            fc1 = module.mlp.dense_h_to_4h

            if (
                not module.config.new_decoder_architecture
                and module.config.parallel_attn
            ):
                attn_ln = module.input_layernorm
                smooth_ln_fcs(attn_ln, [qkv, fc1], qkv_input_scales, alpha, invert_scales)
            else:
                attn_ln = (
                    module.ln_attn
                    if module.config.new_decoder_architecture
                    else module.input_layernorm
                )
                ffn_ln = (
                    module.ln_mlp
                    if module.config.new_decoder_architecture
                    else module.post_attention_layernorm
                )
                smooth_ln_fcs(attn_ln, qkv, qkv_input_scales, alpha, invert_scales)
                smooth_ln_fcs(ffn_ln, fc1, fc1_input_scales, alpha, invert_scales)
        elif isinstance(module, (LlamaDecoderLayer, MistralDecoderLayer)):
            attn_ln = module.input_layernorm  # attention forward norm
            qkv = [
                module.self_attn.q_proj,
                module.self_attn.k_proj,
                module.self_attn.v_proj,
            ]

            qkv_input_scales = scales[name + ".self_attn.q_proj"]
            smooth_ln_fcs_llama_like(attn_ln, qkv, qkv_input_scales, alpha, invert_scales)

            ffn_ln = module.post_attention_layernorm  # feed forward norm
            fcs = [module.mlp.gate_proj, module.mlp.up_proj]
            fcs_input_scales = scales[name + ".mlp.gate_proj"]

            smooth_ln_fcs_llama_like(ffn_ln, fcs, fcs_input_scales, alpha, invert_scales)
        elif isinstance(module, MixtralDecoderLayer):
            attn_ln = module.input_layernorm  # attention forward norm
            qkv = [
                module.self_attn.q_proj,
                module.self_attn.k_proj,
                module.self_attn.v_proj,
            ]

            qkv_input_scales = scales[name + ".self_attn.q_proj"]
            smooth_ln_fcs_llama_like(attn_ln, qkv, qkv_input_scales, alpha, invert_scales)

            ffn_ln = module.post_attention_layernorm  # feed forward norm
            fcs = [module.block_sparse_moe.gate]
            for expert in module.block_sparse_moe.experts:
                fcs.append(expert.w1)
                fcs.append(expert.w3)
            fcs_input_scales = scales[name + ".block_sparse_moe.gate"]

            smooth_ln_fcs_llama_like(ffn_ln, fcs, fcs_input_scales, alpha, invert_scales)

class LNInputSparsityHook:
    def __init__(self, act_sparsity_n, act_sparsity_m, scale):
        self.act_sparsity_n = act_sparsity_n
        self.act_sparsity_m = act_sparsity_m
        self.scale = scale

    @torch.no_grad()
    def __call__(self, module, input):
        x = input[0]
        return (self.apply_activation_sparsity(x),)

    @torch.no_grad()
    def apply_activation_sparsity(self, x):
        if self.act_sparsity_n == 0 or self.act_sparsity_m == 0:
            return x

        x_shape = x.shape
        x_2d = x.view(-1, x_shape[-1])
        scale = self.scale.to(device=x.device, dtype=torch.float32)
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


@torch.no_grad()
def _compute_weight_col_norm(weight, weight_scoring):
    weight = weight.float()
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
                max=(q_high - mean) / std,
            )
    else:
        weight_processed = weight

    return weight_processed.pow(2).sum(dim=0).sqrt()


@torch.no_grad()
def _compute_sparsity_scale_from_fcs(fcs, weight_scoring=True):
    col_norms = []
    for fc in fcs:
        col_norms.append(_compute_weight_col_norm(fc.weight, weight_scoring))
    col_norms = torch.stack(col_norms, dim=0)
    w_col_norm = col_norms.max(dim=0)[0]
    min_norm = w_col_norm.min().clamp(min=1e-5)
    return (w_col_norm / min_norm).view(1, -1)


def hook_ln_sparsity(
    model,
    act_sparsity_n=2,
    act_sparsity_m=4,
    target_modules=None,
    weight_scoring=True,
):
    hooks = []

    if act_sparsity_n == 0 or act_sparsity_m == 0:
        return {"hooks": hooks, "num_hooks": 0}

    def should_apply_to_names(names):
        if target_modules is None:
            return True
        return all(not any(pattern in name for pattern in target_modules) for name in names)

    def register_ln_hook(ln, fcs, fc_names, ln_name):
        if not should_apply_to_names(fc_names):
            return
        scale = _compute_sparsity_scale_from_fcs(fcs, weight_scoring)
        hook = LNInputSparsityHook(act_sparsity_n, act_sparsity_m, scale)
        handle = ln.register_forward_pre_hook(hook)
        hooks.append(handle)
        print(f"Registered LN input sparsity hook ({act_sparsity_n}:{act_sparsity_m}) on: {ln_name}")

    for name, module in model.named_modules():
        if isinstance(module, OPTDecoderLayer):
            attn_ln = module.self_attn_layer_norm
            qkv = [
                module.self_attn.q_proj,
                module.self_attn.k_proj,
                module.self_attn.v_proj,
            ]
            qkv_names = [
                f"{name}.self_attn.q_proj",
                f"{name}.self_attn.k_proj",
                f"{name}.self_attn.v_proj",
            ]
            register_ln_hook(attn_ln, qkv, qkv_names, f"{name}.self_attn_layer_norm")

            ffn_ln = module.final_layer_norm
            fc1 = [module.fc1]
            fc1_names = [f"{name}.fc1"]
            register_ln_hook(ffn_ln, fc1, fc1_names, f"{name}.final_layer_norm")
        elif isinstance(module, BloomBlock):
            attn_ln = module.input_layernorm
            qkv = [module.self_attention.query_key_value]
            qkv_names = [f"{name}.self_attention.query_key_value"]
            register_ln_hook(attn_ln, qkv, qkv_names, f"{name}.input_layernorm")

            ffn_ln = module.post_attention_layernorm
            fc1 = [module.mlp.dense_h_to_4h]
            fc1_names = [f"{name}.mlp.dense_h_to_4h"]
            register_ln_hook(ffn_ln, fc1, fc1_names, f"{name}.post_attention_layernorm")
        elif isinstance(module, FalconDecoderLayer):
            qkv = [module.self_attention.query_key_value]
            qkv_names = [f"{name}.self_attention.query_key_value"]
            fc1 = [module.mlp.dense_h_to_4h]
            fc1_names = [f"{name}.mlp.dense_h_to_4h"]

            if (
                not module.config.new_decoder_architecture
                and module.config.parallel_attn
            ):
                attn_ln = module.input_layernorm
                register_ln_hook(attn_ln, qkv + fc1, qkv_names + fc1_names, f"{name}.input_layernorm")
            else:
                attn_ln = (
                    module.ln_attn
                    if module.config.new_decoder_architecture
                    else module.input_layernorm
                )
                ffn_ln = (
                    module.ln_mlp
                    if module.config.new_decoder_architecture
                    else module.post_attention_layernorm
                )
                register_ln_hook(attn_ln, qkv, qkv_names, f"{name}.ln_attn")
                register_ln_hook(ffn_ln, fc1, fc1_names, f"{name}.ln_mlp")
        elif isinstance(module, (LlamaDecoderLayer, MistralDecoderLayer)):
            attn_ln = module.input_layernorm
            qkv = [
                module.self_attn.q_proj,
                module.self_attn.k_proj,
                module.self_attn.v_proj,
            ]
            qkv_names = [
                f"{name}.self_attn.q_proj",
                f"{name}.self_attn.k_proj",
                f"{name}.self_attn.v_proj",
            ]
            register_ln_hook(attn_ln, qkv, qkv_names, f"{name}.input_layernorm")

            ffn_ln = module.post_attention_layernorm
            fcs = [module.mlp.gate_proj, module.mlp.up_proj]
            fcs_names = [f"{name}.mlp.gate_proj", f"{name}.mlp.up_proj"]
            register_ln_hook(ffn_ln, fcs, fcs_names, f"{name}.post_attention_layernorm")
        elif isinstance(module, MixtralDecoderLayer):
            attn_ln = module.input_layernorm
            qkv = [
                module.self_attn.q_proj,
                module.self_attn.k_proj,
                module.self_attn.v_proj,
            ]
            qkv_names = [
                f"{name}.self_attn.q_proj",
                f"{name}.self_attn.k_proj",
                f"{name}.self_attn.v_proj",
            ]
            register_ln_hook(attn_ln, qkv, qkv_names, f"{name}.input_layernorm")

            ffn_ln = module.post_attention_layernorm
            fcs = [module.block_sparse_moe.gate]
            fcs_names = [f"{name}.block_sparse_moe.gate"]
            for expert_idx, expert in enumerate(module.block_sparse_moe.experts):
                fcs.append(expert.w1)
                fcs.append(expert.w3)
                fcs_names.append(f"{name}.block_sparse_moe.experts.{expert_idx}.w1")
                fcs_names.append(f"{name}.block_sparse_moe.experts.{expert_idx}.w3")
            register_ln_hook(ffn_ln, fcs, fcs_names, f"{name}.post_attention_layernorm")

    return {"hooks": hooks, "num_hooks": len(hooks)}
