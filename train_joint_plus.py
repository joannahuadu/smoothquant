#!/usr/bin/env python3

from __future__ import annotations

import argparse
import logging
import os
from datetime import datetime
from termcolor import colored
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from smoothquant.datautils import get_calibration_batches
from smoothquant.fake_quant import quantize_model
from smoothquant.joint_plus import (
    apply_joint_plus_llama,
    joint_plus_regularization,
    save_joint_plus_checkpoint,
    set_layer_joint_plus_enabled,
    set_layer_x_mask_eval_mode,
    split_joint_plus_params,
)
from smoothquant.smooth import smooth_lm
import pprint

logger = logging.getLogger(__name__)


class _StopForward(RuntimeError):
    pass

def create_logger(exp_dir, dist_rank=0, name=''):
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.propagate = False

    fmt = '[%(asctime)s %(name)s] (%(filename)s %(lineno)d): %(levelname)s %(message)s'
    color_fmt = colored('[%(asctime)s %(name)s]', 'green') + \
                colored('(%(filename)s %(lineno)d)', 'yellow') + ': %(levelname)s %(message)s'

    if dist_rank == 0:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.DEBUG)
        console_handler.setFormatter(
            logging.Formatter(fmt=color_fmt, datefmt='%Y-%m-%d %H:%M:%S'))
        logger.addHandler(console_handler)

    log_file = os.path.join(exp_dir, f'log_rank{dist_rank}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.txt')
    file_handler = logging.FileHandler(log_file, mode='a')
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter(fmt=fmt, datefmt='%Y-%m-%d %H:%M:%S'))
    logger.addHandler(file_handler)

    return logger


def normalize_act_scales(act_scales):
    if not isinstance(act_scales, dict):
        return act_scales
    normalized = dict(act_scales)
    for key, value in list(act_scales.items()):
        mapped = key
        if mapped.endswith(".input"):
            mapped = mapped[: -len(".input")]
        elif mapped.endswith(".output"):
            continue
        if mapped.startswith("layers."):
            mapped = "model." + mapped
        normalized.setdefault(mapped, value)
    return normalized


def parse_args():
    parser = argparse.ArgumentParser(description="Train SmoothQuant joint_plus calibration for Llama.")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--tokenizer_path", type=str, default=None)
    parser.add_argument("--act_scales_path", type=str, default=None)
    parser.add_argument("--dataset", type=str, default="wikitext2", choices=["wikitext2", "c4"])
    parser.add_argument("--alpha", type=float, default=0.85)
    parser.add_argument("--w_bits", type=int, default=8)
    parser.add_argument("--a_bits", type=int, default=8)
    parser.add_argument("--nsamples", type=int, default=128)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--seq_len", type=int, default=2048)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=5e-3)
    parser.add_argument("--alpha_lr", type=float, default=1e-3)
    parser.add_argument("--gate_cost", type=float, default=1e-4)
    parser.add_argument("--alpha_reg", type=float, default=0.01)
    parser.add_argument("--x_mask_tau", type=float, default=1.0)
    parser.add_argument("--x_mask_alpha", type=float, default=1.0)
    parser.add_argument("--x_mask_r_thr", type=float, default=-1.0)
    parser.add_argument("--token_gate_mode", type=str, default="token_all", choices=["token_all", "static_all"])
    parser.add_argument("--token_mlp_hidden", type=int, default=0)
    parser.add_argument("--token_mlp_chunk_size", type=int, default=1024)
    parser.add_argument("--output_dir", type=str, default="outputs")
    parser.add_argument("--exp_name", type=str, default="joint_plus")
    parser.add_argument("--torch_dtype", type=str, default="bfloat16", choices=["float16", "bfloat16", "float32"])
    return parser.parse_args()


def _clone_tree_detached(obj):
    if torch.is_tensor(obj):
        return obj.detach()
    if isinstance(obj, tuple):
        return tuple(_clone_tree_detached(x) for x in obj)
    if isinstance(obj, list):
        return [_clone_tree_detached(x) for x in obj]
    if isinstance(obj, dict):
        return {k: _clone_tree_detached(v) for k, v in obj.items()}
    return obj


def capture_layer_inputs(model, layer_idx, input_ids):
    target_layer = model.model.layers[layer_idx]
    captured = {}

    def _hook(module, args, kwargs):
        captured["args"] = _clone_tree_detached(args)
        captured["kwargs"] = _clone_tree_detached(kwargs)
        raise _StopForward

    handle = target_layer.register_forward_pre_hook(_hook, with_kwargs=True)
    try:
        model(input_ids=input_ids, use_cache=False)
    except _StopForward:
        pass
    finally:
        handle.remove()
    if not captured:
        raise RuntimeError(f"failed to capture inputs for layer {layer_idx}")
    return captured["args"], captured["kwargs"]


def build_model_and_tokenizer(args):
    dtype_map = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path or args.model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=dtype_map[args.torch_dtype],
        trust_remote_code=True,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    return model, tokenizer, device


def main():
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = parse_args()
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    model, tokenizer, device = build_model_and_tokenizer(args)
    if args.act_scales_path:
        act_scales = normalize_act_scales(torch.load(args.act_scales_path))
        smooth_lm(model, act_scales, args.alpha)

    model = quantize_model(
        model,
        weight_quant="per_channel",
        w_bits=args.w_bits,
        a_bits=args.a_bits,
        act_quant="per_token",
        quantize_bmm_input=True,
        act_sparsity_n=0,
        act_sparsity_m=0,
    )
    model = apply_joint_plus_llama(
        model,
        use_x_mask=True,
        x_mask_tau=args.x_mask_tau,
        x_mask_alpha=args.x_mask_alpha,
        x_mask_r_thr=None if args.x_mask_r_thr < 0 else args.x_mask_r_thr,
        token_gate_mode=args.token_gate_mode,
        token_mlp_hidden=args.token_mlp_hidden,
        token_mlp_chunk_size=args.token_mlp_chunk_size,
    )
    model.eval()

    trainloader = get_calibration_batches(
        args.dataset,
        tokenizer,
        nsamples=args.nsamples,
        seed=args.seed,
        seqlen=args.seq_len,
    )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_dir = os.path.join(args.output_dir, f"{args.exp_name}_{timestamp}")
    os.makedirs(exp_dir, exist_ok=True)
    ckpt_path = os.path.join(exp_dir, f"joint_plus_{args.dataset}_w8a8.pt")
    logger = create_logger(exp_dir)
    logger.info('Arguments: ')
    logger.info(pprint.pformat(vars(args)))
    logger.info('--' * 30)
    logger.info(f"Saving outputs to: {exp_dir}")
    logger.info(f"Calibrating {len(model.model.layers)} layers on {len(trainloader)} samples")

    for layer_idx, layer in enumerate(model.model.layers):
        gate_params, alpha_params = split_joint_plus_params(layer)
        if not gate_params and not alpha_params:
            continue

        for param in model.parameters():
            param.requires_grad_(False)
        for param in gate_params + alpha_params:
            param.requires_grad_(True)

        optim_groups = []
        if gate_params:
            optim_groups.append({"params": gate_params, "lr": args.lr})
        if alpha_params:
            optim_groups.append({"params": alpha_params, "lr": args.alpha_lr})
        optimizer = torch.optim.AdamW(optim_groups)

        logger.info(f"\nLayer {layer_idx}: gate_params={len(gate_params)} alpha_params={len(alpha_params)}")
        for epoch in range(args.epochs):
            epoch_total = 0.0
            epoch_distill = 0.0
            epoch_gate = 0.0
            epoch_alpha = 0.0
            for step, (input_ids, _) in enumerate(trainloader):
                input_ids = input_ids.to(device)
                set_layer_joint_plus_enabled(layer, True)
                set_layer_x_mask_eval_mode(layer, False)
                layer_args, layer_kwargs = capture_layer_inputs(model, layer_idx, input_ids)

                with torch.no_grad():
                    set_layer_joint_plus_enabled(layer, False)
                    teacher_out = layer(*layer_args, **layer_kwargs)[0].detach()

                set_layer_joint_plus_enabled(layer, True)
                student_out = layer(*layer_args, **layer_kwargs)[0]
                distill_loss = F.mse_loss(student_out.float(), teacher_out.float())
                gate_loss, alpha_loss = joint_plus_regularization(layer, args.gate_cost, args.alpha_reg)
                loss = distill_loss + gate_loss + alpha_loss
                epoch_total += float(loss.detach().cpu())
                epoch_distill += float(distill_loss.detach().cpu())
                epoch_gate += float(gate_loss.detach().cpu())
                epoch_alpha += float(alpha_loss.detach().cpu())

                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

            cur_lr = optimizer.param_groups[0]["lr"] if optimizer.param_groups else float("nan")
            logger.info(
                f"layer {layer_idx} iter {epoch}, lr {cur_lr:.8f}, "
                f"loss={epoch_total / len(trainloader):.8f}, "
                f"distill={epoch_distill / len(trainloader):.8f}, "
                f"gate={epoch_gate / len(trainloader):.8f}, "
                f"alpha={epoch_alpha / len(trainloader):.8f}"
            )

            stats_parts = []
            for name, mask in (
                ("self_attn.x_mask_in", getattr(layer.self_attn, "x_mask_in", None)),
                ("self_attn.x_mask_out", getattr(layer.self_attn, "x_mask_out", None)),
                ("mlp.x_mask_up", getattr(layer.mlp, "x_mask_up", None)),
                ("mlp.x_mask_down", getattr(layer.mlp, "x_mask_down", None)),
            ):
                if mask is None or not getattr(mask, "use_x_mask", False):
                    continue
                mean = getattr(mask, "_last_x_mask_gate_mean", None)
                if mean is None:
                    continue
                std = getattr(mask, "_last_x_mask_gate_std", None)
                frac_low = getattr(mask, "_last_x_mask_gate_frac_low", None)
                frac_high = getattr(mask, "_last_x_mask_gate_frac_high", None)
                stats_parts.append(
                    f"{name}: mean={float(mean):.3f} std={float(std) if std is not None else float('nan'):.3f} "
                    f"low={float(frac_low) if frac_low is not None else float('nan'):.3f} "
                    f"high={float(frac_high) if frac_high is not None else float('nan'):.3f}"
                )
            if stats_parts:
                logger.info("gate_stats: " + " | ".join(stats_parts))

            if hasattr(layer.self_attn, "softmax_alpha"):
                alpha = layer.self_attn.softmax_alpha.detach().float()
                logger.info(
                    f"softmax_alpha: mean={alpha.mean():.4f} min={alpha.min():.4f} max={alpha.max():.4f} "
                    f"std={alpha.std():.4f} |a-1|_mean={((alpha - 1).abs()).mean():.4f}"
                )
            if hasattr(layer.self_attn, "output_scale"):
                scale = layer.self_attn.output_scale.detach().float()
                logger.info(
                    f"output_scale: mean={scale.mean():.4f} min={scale.min():.4f} max={scale.max():.4f} "
                    f"std={scale.std():.4f} |s-1|_mean={((scale - 1).abs()).mean():.4f}"
                )

        set_layer_joint_plus_enabled(layer, True)
        set_layer_x_mask_eval_mode(layer, True)

    save_joint_plus_checkpoint(
        model,
        ckpt_path,
        meta={
            "model_path": args.model_path,
            "dataset": args.dataset,
            "alpha": args.alpha,
            "w_bits": args.w_bits,
            "a_bits": args.a_bits,
            "nsamples": args.nsamples,
            "seq_len": args.seq_len,
            "epochs": args.epochs,
            "lr": args.lr,
            "alpha_lr": args.alpha_lr,
            "gate_cost": args.gate_cost,
            "alpha_reg": args.alpha_reg,
            "token_gate_mode": args.token_gate_mode,
            "token_mlp_hidden": args.token_mlp_hidden,
        },
    )
    logger.info(f"Saved joint_plus checkpoint to: {ckpt_path}")


if __name__ == "__main__":
    main()
