from __future__ import annotations

from typing import Any, Dict, Iterable, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn


class TokenResidualMLP(nn.Module):
    def __init__(self, hidden: int = 0, chunk_size: int = 1024):
        super().__init__()
        self.hidden = int(hidden)
        self.chunk_size = int(chunk_size) if chunk_size is not None else 0

        in_dim = 8
        if self.hidden <= 0:
            self.weight = nn.Parameter(torch.zeros((in_dim,), dtype=torch.float32))
            self.bias = nn.Parameter(torch.zeros((), dtype=torch.float32))
            nn.init.trunc_normal_(self.weight, std=0.02)
        else:
            self.fc1 = nn.Linear(in_dim, self.hidden, bias=True, dtype=torch.float32)
            self.fc2 = nn.Linear(self.hidden, 1, bias=True, dtype=torch.float32)
            nn.init.trunc_normal_(self.fc1.weight, std=0.02)
            nn.init.zeros_(self.fc1.bias)
            nn.init.trunc_normal_(self.fc2.weight, std=0.02)
            nn.init.zeros_(self.fc2.bias)

    def forward(self, xg: torch.Tensor) -> torch.Tensor:
        if xg.numel() == 0:
            return xg.new_zeros(*xg.shape[:-1], dtype=torch.float32)

        xg_flat = xg.reshape(-1, xg.shape[-2], 4)
        n, g = xg_flat.shape[0], xg_flat.shape[1]
        chunk = self.chunk_size if self.chunk_size and self.chunk_size > 0 else n
        out = xg_flat.new_empty((n, g), dtype=torch.float32)

        if self.hidden <= 0:
            w_x = self.weight[:4]
            w_abs = self.weight[4:]
            for i in range(0, n, chunk):
                xc = xg_flat[i : i + chunk].float()
                out[i : i + chunk] = (
                    (xc * w_x).sum(dim=-1) + (xc.abs() * w_abs).sum(dim=-1) + self.bias
                )
        else:
            for i in range(0, n, chunk):
                xc = xg_flat[i : i + chunk].float()
                feat = torch.cat([xc, xc.abs()], dim=-1)
                out[i : i + chunk] = self.fc2(F.gelu(self.fc1(feat))).squeeze(-1)

        return out.view(*xg.shape[:-1])


class XMaskSwitchTop2Hard(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        *,
        group_size: int = 4,
        topk: int = 2,
        x_mask_tau: float = 1.0,
        x_mask_alpha: float = 1.0,
        x_mask_r_thr: Optional[float] = None,
    ):
        super().__init__()
        if hidden_dim <= 0 or hidden_dim % group_size != 0:
            raise ValueError(f"hidden_dim ({hidden_dim}) must be divisible by group_size ({group_size})")

        self.use_x_mask = True
        self.group_size = int(group_size)
        self.topk = int(topk)
        self.x_mask_tau = float(x_mask_tau)
        self.x_mask_alpha = float(x_mask_alpha)
        self.x_mask_r_thr = x_mask_r_thr
        self._eval_mode = False

        num_groups = hidden_dim // group_size
        self.x_mask_gate_logits = nn.Parameter(torch.zeros((num_groups,), dtype=torch.float32))
        self.x_mask_gate_mean_requires_grad = False

        self.x_mask_token_gate_enabled = False
        self.x_mask_token_mlp: Optional[TokenResidualMLP] = None
        self.x_mask_token_mlp_hidden = 0
        self.x_mask_token_mlp_chunk_size = 1024
        self.x_mask_token_use_layer_scale = True
        self.x_mask_token_scale = nn.Parameter(torch.ones((), dtype=torch.float32), requires_grad=False)

        self._last_x_mask_gate_mean = None
        self._last_x_mask_gate_mean_grad = None
        self._last_x_mask_gate_std = None
        self._last_x_mask_gate_frac_low = None
        self._last_x_mask_gate_frac_high = None
        self._last_x_mask_gate_delta_l2 = None

    def _ensure_x_mask_token_mlp(self) -> TokenResidualMLP:
        if self.x_mask_token_mlp is None:
            self.x_mask_token_mlp = TokenResidualMLP(
                hidden=int(self.x_mask_token_mlp_hidden),
                chunk_size=int(self.x_mask_token_mlp_chunk_size),
            )
            self.x_mask_token_mlp = self.x_mask_token_mlp.to(device=self.x_mask_gate_logits.device)
        return self.x_mask_token_mlp

    def _compute_x_mask_token_delta(self, reshaped: torch.Tensor) -> Optional[torch.Tensor]:
        if not getattr(self, "x_mask_token_gate_enabled", False):
            self._last_x_mask_gate_delta_l2 = None
            return None
        mlp = self._ensure_x_mask_token_mlp()
        delta = mlp(reshaped)
        self._last_x_mask_gate_delta_l2 = delta.pow(2).mean()
        return delta

    def _compute_gate(self, reshaped: torch.Tensor) -> torch.Tensor:
        base = self.x_mask_gate_logits.to(device=reshaped.device, dtype=torch.float32)
        base = base.view(*([1] * (reshaped.dim() - 2)), -1).expand(*reshaped.shape[:-1])
        delta = self._compute_x_mask_token_delta(reshaped)
        if delta is not None:
            delta = delta - delta.mean(dim=-1, keepdim=True)
            delta = torch.tanh(delta)
            delta = delta * self.x_mask_token_scale.to(delta)
            base = base + delta
        r_fp32 = torch.sigmoid(base)
        self._last_x_mask_gate_mean_grad = r_fp32.mean() if self.x_mask_gate_mean_requires_grad else None
        with torch.no_grad():
            stats = r_fp32.detach().float().cpu()
            self._last_x_mask_gate_mean = stats.mean()
            self._last_x_mask_gate_std = stats.std(unbiased=False)
            self._last_x_mask_gate_frac_low = (stats < 0.05).float().mean()
            self._last_x_mask_gate_frac_high = (stats > 0.95).float().mean()
        return r_fp32.to(dtype=reshaped.dtype)

    def to_eval_mode(self) -> None:
        self._eval_mode = True

    def to_train_mode(self) -> None:
        self._eval_mode = False

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        if not self.use_x_mask or tensor.shape[-1] % self.group_size != 0:
            return tensor

        alpha = float(self.x_mask_alpha)
        if alpha <= 0.0:
            return tensor

        reshaped = tensor.view(*tensor.shape[:-1], -1, self.group_size)
        scores = reshaped.abs()
        tau = float(self.x_mask_tau)
        gate_soft = None if tau <= 0.0 else float(self.topk) * torch.softmax(scores / tau, dim=-1)

        idx = scores.topk(self.topk, dim=-1).indices
        gate_hard = torch.zeros_like(reshaped)
        gate_hard.scatter_(-1, idx, 1.0)
        gate_raw = gate_hard if gate_soft is None else gate_hard - gate_soft.detach() + gate_soft
        sparse = reshaped * gate_raw

        r = self._compute_gate(reshaped).unsqueeze(-1)
        mixed = r * reshaped + (1.0 - r) * sparse

        if self._eval_mode and self.x_mask_r_thr is not None:
            hard_sel = (r.squeeze(-1).float() < float(self.x_mask_r_thr)).to(dtype=mixed.dtype).unsqueeze(-1)
            mixed = mixed * (1.0 - hard_sel + hard_sel * gate_raw)

        if alpha < 1.0:
            mixed = (1.0 - alpha) * reshaped + alpha * mixed
        return mixed.view_as(tensor)


def _configure_x_mask_module(
    module: Optional[XMaskSwitchTop2Hard],
    *,
    token_gate_mode: str,
    token_mlp_hidden: int,
    token_mlp_chunk_size: int,
    token_use_layer_scale: bool,
) -> None:
    if module is None:
        return
    if token_gate_mode == "token_all":
        module.x_mask_token_gate_enabled = True
        module.x_mask_token_mlp_hidden = int(token_mlp_hidden)
        module.x_mask_token_mlp_chunk_size = int(token_mlp_chunk_size)
        module.x_mask_token_use_layer_scale = bool(token_use_layer_scale)


def _make_input_mask_hook(owner: nn.Module, mask_name: str):
    def _hook(module: nn.Module, inputs: Tuple[torch.Tensor, ...]):
        if not getattr(owner, "joint_plus_enabled", True):
            return None
        mask = getattr(owner, mask_name, None)
        if mask is None or not inputs:
            return None
        return (mask(inputs[0]),) + tuple(inputs[1:])

    return _hook


def _make_q_proj_output_hook(attn: nn.Module):
    def _hook(module: nn.Module, inputs: Tuple[torch.Tensor, ...], output: torch.Tensor):
        if not getattr(attn, "joint_plus_enabled", True):
            return output
        alpha = getattr(attn, "softmax_alpha", None)
        if alpha is None:
            return output
        scale = alpha.to(device=output.device, dtype=output.dtype).repeat_interleave(attn.head_dim)
        return output * scale.view(*([1] * (output.dim() - 1)), -1)

    return _hook


def _make_output_scale_hook(attn: nn.Module):
    def _hook(module: nn.Module, inputs: Tuple[torch.Tensor, ...], output: torch.Tensor):
        if not getattr(attn, "joint_plus_enabled", True):
            return output
        output_scale = getattr(attn, "output_scale", None)
        if output_scale is None:
            return output
        scale = output_scale.to(device=output.device, dtype=output.dtype)
        return output * scale.view(*([1] * (output.dim() - 1)), -1)

    return _hook


def _attach_joint_plus_to_attention(
    attn: nn.Module,
    *,
    use_x_mask: bool,
    x_mask_tau: float,
    x_mask_alpha: float,
    x_mask_r_thr: Optional[float],
    token_gate_mode: str,
    token_mlp_hidden: int,
    token_mlp_chunk_size: int,
    token_use_layer_scale: bool,
) -> None:
    ref_device = attn.q_proj.weight.device
    attn.x_mask_in = XMaskSwitchTop2Hard(
        attn.hidden_size,
        x_mask_tau=x_mask_tau,
        x_mask_alpha=x_mask_alpha,
        x_mask_r_thr=x_mask_r_thr,
    ).to(device=ref_device) if use_x_mask else None
    attn.x_mask_out = XMaskSwitchTop2Hard(
        attn.hidden_size,
        x_mask_tau=x_mask_tau,
        x_mask_alpha=x_mask_alpha,
        x_mask_r_thr=x_mask_r_thr,
    ).to(device=ref_device) if use_x_mask else None
    for module in (attn.x_mask_in, attn.x_mask_out):
        _configure_x_mask_module(
            module,
            token_gate_mode=token_gate_mode,
            token_mlp_hidden=token_mlp_hidden,
            token_mlp_chunk_size=token_mlp_chunk_size,
            token_use_layer_scale=token_use_layer_scale,
        )

    attn.softmax_alpha = nn.Parameter(torch.ones(attn.num_heads, dtype=torch.float32, device=ref_device))
    attn.output_scale = nn.Parameter(torch.ones(attn.hidden_size, dtype=torch.float32, device=ref_device))
    attn.joint_plus_enabled = True
    attn._joint_plus_hook_handles = [
        attn.q_proj.register_forward_pre_hook(_make_input_mask_hook(attn, "x_mask_in")),
        attn.k_proj.register_forward_pre_hook(_make_input_mask_hook(attn, "x_mask_in")),
        attn.v_proj.register_forward_pre_hook(_make_input_mask_hook(attn, "x_mask_in")),
        attn.o_proj.register_forward_pre_hook(_make_input_mask_hook(attn, "x_mask_out")),
        attn.q_proj.register_forward_hook(_make_q_proj_output_hook(attn)),
        attn.o_proj.register_forward_hook(_make_output_scale_hook(attn)),
    ]


def _attach_joint_plus_to_mlp(
    mlp: nn.Module,
    *,
    use_x_mask: bool,
    x_mask_tau: float,
    x_mask_alpha: float,
    x_mask_r_thr: Optional[float],
    token_gate_mode: str,
    token_mlp_hidden: int,
    token_mlp_chunk_size: int,
    token_use_layer_scale: bool,
) -> None:
    ref_device = mlp.gate_proj.weight.device
    mlp.x_mask_up = XMaskSwitchTop2Hard(
        mlp.hidden_size,
        x_mask_tau=x_mask_tau,
        x_mask_alpha=x_mask_alpha,
        x_mask_r_thr=x_mask_r_thr,
    ).to(device=ref_device) if use_x_mask else None
    mlp.x_mask_down = XMaskSwitchTop2Hard(
        mlp.intermediate_size,
        x_mask_tau=x_mask_tau,
        x_mask_alpha=x_mask_alpha,
        x_mask_r_thr=x_mask_r_thr,
    ).to(device=ref_device) if use_x_mask else None
    for module in (mlp.x_mask_up, mlp.x_mask_down):
        _configure_x_mask_module(
            module,
            token_gate_mode=token_gate_mode,
            token_mlp_hidden=token_mlp_hidden,
            token_mlp_chunk_size=token_mlp_chunk_size,
            token_use_layer_scale=token_use_layer_scale,
        )

    mlp.joint_plus_enabled = True
    mlp._joint_plus_hook_handles = [
        mlp.gate_proj.register_forward_pre_hook(_make_input_mask_hook(mlp, "x_mask_up")),
        mlp.up_proj.register_forward_pre_hook(_make_input_mask_hook(mlp, "x_mask_up")),
        mlp.down_proj.register_forward_pre_hook(_make_input_mask_hook(mlp, "x_mask_down")),
    ]


def apply_joint_plus_llama(
    model,
    *,
    use_x_mask: bool = True,
    x_mask_tau: float = 1.0,
    x_mask_alpha: float = 1.0,
    x_mask_r_thr: Optional[float] = None,
    token_gate_mode: str = "token_all",
    token_mlp_hidden: int = 0,
    token_mlp_chunk_size: int = 1024,
    token_use_layer_scale: bool = True,
):
    for layer in model.model.layers:
        if getattr(layer.self_attn, "_joint_plus_applied", False) and getattr(layer.mlp, "_joint_plus_applied", False):
            continue
        if getattr(layer.self_attn.config, "pretraining_tp", 1) > 1 or getattr(layer.mlp.config, "pretraining_tp", 1) > 1:
            raise NotImplementedError(
                "joint_plus hook mode requires pretraining_tp == 1; hook registration does not intercept the "
                "F.linear fast path used when pretraining_tp > 1"
            )
        _attach_joint_plus_to_attention(
            layer.self_attn,
            use_x_mask=use_x_mask,
            x_mask_tau=x_mask_tau,
            x_mask_alpha=x_mask_alpha,
            x_mask_r_thr=x_mask_r_thr,
            token_gate_mode=token_gate_mode,
            token_mlp_hidden=token_mlp_hidden,
            token_mlp_chunk_size=token_mlp_chunk_size,
            token_use_layer_scale=token_use_layer_scale,
        )
        layer.self_attn._joint_plus_applied = True
        _attach_joint_plus_to_mlp(
            layer.mlp,
            use_x_mask=use_x_mask,
            x_mask_tau=x_mask_tau,
            x_mask_alpha=x_mask_alpha,
            x_mask_r_thr=x_mask_r_thr,
            token_gate_mode=token_gate_mode,
            token_mlp_hidden=token_mlp_hidden,
            token_mlp_chunk_size=token_mlp_chunk_size,
            token_use_layer_scale=token_use_layer_scale,
        )
        layer.mlp._joint_plus_applied = True
    return model


def iter_layer_x_mask_modules(layer) -> Iterable[nn.Module]:
    for module in (getattr(layer.self_attn, "x_mask_in", None), getattr(layer.self_attn, "x_mask_out", None),
                   getattr(layer.mlp, "x_mask_up", None), getattr(layer.mlp, "x_mask_down", None)):
        if module is not None:
            yield module


def set_layer_joint_plus_enabled(layer, enabled: bool) -> None:
    if hasattr(layer.self_attn, "joint_plus_enabled"):
        layer.self_attn.joint_plus_enabled = bool(enabled)
    if hasattr(layer.mlp, "joint_plus_enabled"):
        layer.mlp.joint_plus_enabled = bool(enabled)


def set_layer_x_mask_alpha(layer, alpha: float) -> None:
    for module in iter_layer_x_mask_modules(layer):
        module.x_mask_alpha = float(alpha)


def set_layer_x_mask_eval_mode(layer, enable: bool) -> None:
    for module in iter_layer_x_mask_modules(layer):
        if enable:
            module.to_eval_mode()
        else:
            module.to_train_mode()


def parse_layer_spec(spec: str) -> set[int]:
    if not spec:
        return set()
    out = set()
    for part in str(spec).split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start, end = part.split("-", 1)
            out.update(range(int(start), int(end) + 1))
        else:
            out.add(int(part))
    return out


def split_joint_plus_params(layer) -> Tuple[list[nn.Parameter], list[nn.Parameter]]:
    gate_params: list[nn.Parameter] = []
    alpha_params: list[nn.Parameter] = []

    for module in iter_layer_x_mask_modules(layer):
        if hasattr(module, "x_mask_gate_logits"):
            module.x_mask_gate_logits.requires_grad_(True)
            gate_params.append(module.x_mask_gate_logits)
        if getattr(module, "x_mask_token_gate_enabled", False):
            mlp = module._ensure_x_mask_token_mlp()
            gate_params.extend(p for p in mlp.parameters())
            if module.x_mask_token_scale.requires_grad:
                gate_params.append(module.x_mask_token_scale)

    if hasattr(layer.self_attn, "softmax_alpha"):
        layer.self_attn.softmax_alpha.requires_grad_(True)
        alpha_params.append(layer.self_attn.softmax_alpha)
    if hasattr(layer.self_attn, "output_scale"):
        layer.self_attn.output_scale.requires_grad_(True)
        alpha_params.append(layer.self_attn.output_scale)
    return gate_params, alpha_params


def joint_plus_regularization(layer, gate_cost: float, alpha_reg: float) -> Tuple[torch.Tensor, torch.Tensor]:
    device = next(layer.parameters()).device
    gate_loss = torch.zeros((), device=device)
    alpha_loss = torch.zeros((), device=device)

    if gate_cost > 0:
        for module in iter_layer_x_mask_modules(layer):
            gate_loss = gate_loss + gate_cost * torch.sigmoid(module.x_mask_gate_logits.float()).mean()
            delta_l2 = getattr(module, "_last_x_mask_gate_delta_l2", None)
            if delta_l2 is not None:
                gate_loss = gate_loss + gate_cost * 0.1 * delta_l2

    if alpha_reg > 0 and hasattr(layer.self_attn, "softmax_alpha"):
        alpha_loss = alpha_loss + alpha_reg * (layer.self_attn.softmax_alpha.float() - 1.0).pow(2).mean()
    if alpha_reg > 0 and hasattr(layer.self_attn, "output_scale"):
        alpha_loss = alpha_loss + alpha_reg * (layer.self_attn.output_scale.float() - 1.0).pow(2).mean()
    return gate_loss, alpha_loss


def save_joint_plus_checkpoint(model, path: str, meta: Optional[Dict[str, Any]] = None) -> None:
    ckpt_layers: Dict[int, Dict[str, torch.Tensor]] = {}
    alpha_by_layer = []
    output_scale_by_layer = []

    for idx, layer in enumerate(model.model.layers):
        keep = {}
        for key, value in layer.state_dict().items():
            if "x_mask" in key or key.endswith("softmax_alpha") or key.endswith("output_scale"):
                keep[key] = value.detach().cpu()
        ckpt_layers[idx] = keep
        alpha_by_layer.append(layer.self_attn.softmax_alpha.detach().cpu().clone())
        output_scale_by_layer.append(layer.self_attn.output_scale.detach().cpu().clone())

    torch.save(
        {
            "meta": meta or {},
            "layers": ckpt_layers,
            "softmax_alpha": torch.stack(alpha_by_layer),
            "output_scale": torch.stack(output_scale_by_layer),
        },
        path,
    )


def load_joint_plus_checkpoint(model, path: str, *, skip_layers: Optional[Iterable[int]] = None) -> Dict[str, Any]:
    ckpt = torch.load(path, map_location="cpu")
    skip = set(skip_layers or [])
    for idx, layer in enumerate(model.model.layers):
        if idx in skip:
            continue
        layer_state = ckpt.get("layers", {}).get(idx)
        if layer_state:
            layer.load_state_dict(layer_state, strict=False)

    if "softmax_alpha" in ckpt:
        alpha = ckpt["softmax_alpha"]
        for idx, layer in enumerate(model.model.layers):
            if idx not in skip:
                layer.self_attn.softmax_alpha.data.copy_(alpha[idx].to(layer.self_attn.softmax_alpha))
    if "output_scale" in ckpt:
        output_scale = ckpt["output_scale"]
        for idx, layer in enumerate(model.model.layers):
            if idx not in skip:
                layer.self_attn.output_scale.data.copy_(output_scale[idx].to(layer.self_attn.output_scale))
    return ckpt.get("meta", {})
