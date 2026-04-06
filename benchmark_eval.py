#!/usr/bin/env python3
"""
Benchmark Evaluation Script for SmoothQuant Models using lm_eval

This script evaluates SmoothQuant quantized models on various benchmarks
such as MMLU using the EleutherAI lm-evaluation-harness.

Usage:
    # Evaluate MMLU with w8a8 quantization
    python benchmark_eval.py --model_path meta-llama/Llama-3.1-8B --alpha 0.85 --w_bits 8 --a_bits 8 --tasks mmlu

    # Evaluate MMLU with w4a4 quantization
    python benchmark_eval.py --model_path meta-llama/Llama-3.1-8B --alpha 0.85 --w_bits 4 --a_bits 4 --tasks mmlu

    # Evaluate with pre-computed activation scales
    python benchmark_eval.py --model_path ./model --act_scales_path ./act_scales/llama-2-7b.pt --w_bits 8 --a_bits 8 --tasks mmlu

    # Evaluate without quantization (baseline)
    python benchmark_eval.py --model_path meta-llama/Llama-3.1-8B --tasks mmlu
"""

import argparse
import json
import os
import torch
import torch.nn as nn
import tqdm
from pathlib import Path

from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from lm_eval import simple_evaluate
from lm_eval import utils as lm_eval_utils
from lm_eval.models.huggingface import HFLM
from lm_eval.tasks import TaskManager

from smoothquant.smooth import smooth_lm
from smoothquant.fake_quant import quantize_model
from smoothquant.joint_plus import apply_joint_plus_llama, load_joint_plus_checkpoint, parse_layer_spec
from smoothquant.sparse import apply_activation_sparsity_to_model, remove_activation_sparsity_hooks


def normalize_act_scales(act_scales):
    if not isinstance(act_scales, dict):
        return act_scales

    normalized = dict(act_scales)

    # ARCQuant stores activation scales as `layers.X...input/output`; SmoothQuant
    # expects `model.layers.X...` keyed by the input scales.
    for key, value in list(act_scales.items()):
        mapped_key = key
        if mapped_key.endswith(".input"):
            mapped_key = mapped_key[: -len(".input")]
        elif mapped_key.endswith(".output"):
            continue

        if mapped_key.startswith("layers."):
            mapped_key = "model." + mapped_key

        normalized.setdefault(mapped_key, value)

    return normalized


class PerplexityEvaluator:
    def __init__(self, dataset, tokenizer, device, n_samples=None, seq_len=2048):
        self.device = device
        self.seq_len = seq_len
        self.dataset = tokenizer(
            "\n\n".join(dataset["text"]), return_tensors="pt"
        ).input_ids.to(device)
        self.n_samples = n_samples

    @torch.no_grad()
    def evaluate(self, model):
        model.eval()
        nlls = []
        n_samples = self.n_samples if self.n_samples else self.dataset.size(1) // self.seq_len
        for i in tqdm.tqdm(range(n_samples), desc="Evaluating perplexity"):
            start = i * self.seq_len
            end = (i + 1) * self.seq_len
            batch = self.dataset[:, start:end].to(model.device)
            lm_logits = model(batch).logits
            shift_logits = lm_logits[:, :-1, :].contiguous().float()
            shift_labels = self.dataset[:, start:end][:, 1:]
            loss = nn.CrossEntropyLoss()(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
            )
            nlls.append(loss.float() * self.seq_len)

        return torch.exp(torch.stack(nlls).sum() / (n_samples * self.seq_len)).item()


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate SmoothQuant models with lm_eval")

    # Model arguments
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to the model or model identifier from huggingface.co/models",
    )
    parser.add_argument(
        "--tokenizer_path",
        type=str,
        default=None,
        help="Path to the tokenizer (defaults to model_path if not specified)",
    )

    # SmoothQuant arguments (matching ppl_eval.py)
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.5,
        help="Smoothing factor for SmoothQuant (default: 0.5)",
    )
    parser.add_argument(
        "--act_scales_path",
        type=str,
        default=None,
        help="Path to activation scales file for SmoothQuant",
    )
    parser.add_argument(
        "--invert_scales",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Invert SmoothQuant scales (scales = 1/scales)",
    )
    parser.add_argument(
        "--w_bits",
        type=int,
        default=8,
        help="Weight quantization bit width (default: 8)",
    )
    parser.add_argument(
        "--a_bits",
        type=int,
        default=8,
        help="Activation quantization bit width (default: 8)",
    )
    parser.add_argument(
        "--act_sparsity",
        type=str,
        default="",
        help="Enable activation N:M sparsity, format '2:4'. Empty disables.",
    )
    parser.add_argument(
        "--target_modules",
        type=str,
        default=None,
        help="Target modules for sparsity (e.g., 'q_proj,k_proj,v_proj')",
    )
    parser.add_argument(
        "--quantize",
        action="store_true",
        help="Enable quantization (must also set w_bits and a_bits)",
    )
    parser.add_argument(
        "--act_sparsity_location",
        type=str,
        default="pre_quant",
        choices=["pre_quant", "post_quant", "pre_smooth"],
        help="Where to apply activation sparsity when quantizing (default: pre_quant). Use pre_smooth to hook LN inputs before SmoothQuant",
    )
    parser.add_argument(
        "--weight_scoring",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable weight scoring for sparsity scaling (default: True)",
    )
    parser.add_argument(
        "--joint_plus_ckpt",
        type=str,
        default=None,
        help="Optional checkpoint produced by train_joint_plus.py.",
    )
    parser.add_argument(
        "--joint_plus_skip_layers",
        type=str,
        default="",
        help="Comma/range list of layer ids to skip when loading joint_plus ckpt.",
    )

    # Evaluation arguments
    parser.add_argument(
        "--tasks",
        type=str,
        default="mmlu",
        help="Comma-separated list of tasks to evaluate (e.g., mmlu,hellaswag,winogrande)",
    )
    parser.add_argument(
        "--num_fewshot",
        type=int,
        default=0,
        help="Number of few-shot examples (default: 0, zero-shot)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Batch size for evaluation (default: 8)",
    )
    parser.add_argument(
        "--batch_size_overrides",
        type=str,
        default='{"mmlu": 1, "ceval-valid": 1}',
        help="JSON dict of per-task batch size overrides, matching ARCQuant eval behavior.",
    )
    parser.add_argument(
        "--fewshot_overrides",
        type=str,
        default='{"mmlu": 5, "ceval-valid": 5}',
        help="JSON dict of per-task few-shot overrides, matching ARCQuant eval behavior.",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default=None,
        help="Path to save evaluation results as JSON",
    )
    parser.add_argument(
        "--eval_ppl",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Run Wikitext-2 perplexity evaluation in addition to lm_eval tasks.",
    )
    parser.add_argument(
        "--ppl_n_samples",
        type=int,
        default=None,
        help="Optional number of 2048-token windows for perplexity evaluation.",
    )
    parser.add_argument(
        "--torch_dtype",
        type=str,
        default="bfloat16",
        choices=["float32", "float16", "bfloat16"],
        help="Torch dtype for model loading (default: float16)",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    # Set tokenizer path
    tokenizer_path = args.tokenizer_path or args.model_path

    # Parse tasks
    tasks_list = [t.strip() for t in args.tasks.split(",") if t.strip()]
    task_aliases = {"ceval": "ceval-valid"}
    tasks_list = [task_aliases.get(task, task) for task in tasks_list]
    task_manager = TaskManager()
    tasks_list = sorted(set(lm_eval_utils.pattern_match(tasks_list, task_manager.all_tasks)))
    fewshot_overrides = json.loads(args.fewshot_overrides) if args.fewshot_overrides else {}
    batch_size_overrides = json.loads(args.batch_size_overrides) if args.batch_size_overrides else {}

    # Parse activation sparsity
    act_sparsity_n, act_sparsity_m = 0, 0
    if args.act_sparsity:
        act_sparsity_n, act_sparsity_m = map(int, args.act_sparsity.split(":"))
        print(f"Enabling activation sparsity {act_sparsity_n}:{act_sparsity_m}")
    target_modules = args.target_modules.split(",") if args.target_modules else None
    if target_modules:
        print(f"Target modules: {target_modules}")

    # Set torch dtype
    dtype_map = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    torch_dtype = dtype_map[args.torch_dtype]

    # Determine if quantization is enabled
    # Quantize if --quantize flag is set OR if w_bits/a_bits differ from 8
    quantize_enabled = args.quantize or args.w_bits != 8 or args.a_bits != 8

    print("="*60)
    print("SmoothQuant Model Evaluation with lm_eval")
    print("="*60)
    print(f"Model: {args.model_path}")
    print(f"Tokenizer: {tokenizer_path}")
    print(f"Tasks: {tasks_list}")
    print(f"Num fewshot: {args.num_fewshot}")
    print(f"Batch size: {args.batch_size}")
    print(f"Alpha: {args.alpha}")
    print(f"Eval PPL: {args.eval_ppl}")
    if args.act_scales_path:
        print(f"Act scales: {args.act_scales_path}")
    if quantize_enabled:
        print(f"Quantization: w{args.w_bits}a{args.a_bits}")
    print("="*60)

    # Load tokenizer
    print("\nLoading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_path,
        trust_remote_code=True,
    )

    # Load model
    print(f"Loading model from {args.model_path}...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch_dtype,
        trust_remote_code=True,
    )
    model_device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(model_device)
    print(f"Model loaded on {next(model.parameters()).device}")

    # Apply SmoothQuant smoothing if activation scales are provided
    if args.act_scales_path:
        if os.path.exists(args.act_scales_path):
            print(f"\nLoading activation scales from {args.act_scales_path}")
            act_scales = normalize_act_scales(torch.load(args.act_scales_path))
            print(f"Applying SmoothQuant smoothing with alpha={args.alpha}, invert={args.invert_scales}...")
            smooth_lm(model, act_scales, args.alpha, invert_scales=args.invert_scales)
            print("Smoothing applied successfully")
        else:
            print(f"Warning: Activation scales file not found: {args.act_scales_path}")
            print("Proceeding without SmoothQuant smoothing...")

    # Track sparsity hooks for cleanup
    sparsity_hooks = None

    # Apply quantization if specified
    if quantize_enabled:
        print(f"\nApplying w{args.w_bits}a{args.a_bits} quantization...")
        print(f"target_modules: {args.target_modules}")
        print(f"N:M: {act_sparsity_n}:{act_sparsity_m}")
        print(f"act_sparsity_location: {args.act_sparsity_location}")
        model = quantize_model(
            model,
            weight_quant="per_channel",
            w_bits=args.w_bits,
            a_bits=args.a_bits,
            act_quant="per_token",
            quantize_bmm_input=True,
            act_sparsity_n=act_sparsity_n,
            act_sparsity_m=act_sparsity_m,
            weight_scoring=args.weight_scoring,
            act_sparsity_location=args.act_sparsity_location,
            target_modules=target_modules,
        )
        print("Quantization applied successfully")
    elif act_sparsity_n and act_sparsity_m:
        # Apply only sparsity without quantization
        print(f"\nApplying activation sparsity ({act_sparsity_n}:{act_sparsity_m}) without quantization...")
        sparsity_hooks = apply_activation_sparsity_to_model(
            model,
            act_sparsity_n=act_sparsity_n,
            act_sparsity_m=act_sparsity_m,
            target_modules=target_modules,
            weight_scoring=args.weight_scoring,
        )
        print(f"Registered {sparsity_hooks['num_hooks']} sparsity hooks")

    if args.joint_plus_ckpt:
        print(f"\nLoading joint_plus checkpoint from {args.joint_plus_ckpt}")
        model = apply_joint_plus_llama(model)
        meta = load_joint_plus_checkpoint(
            model,
            args.joint_plus_ckpt,
            skip_layers=parse_layer_spec(args.joint_plus_skip_layers),
        )
        for layer in model.model.layers:
            if hasattr(layer.self_attn, "x_mask_in") and layer.self_attn.x_mask_in is not None:
                layer.self_attn.x_mask_in.to_eval_mode()
            if hasattr(layer.self_attn, "x_mask_out") and layer.self_attn.x_mask_out is not None:
                layer.self_attn.x_mask_out.to_eval_mode()
            if hasattr(layer.mlp, "x_mask_up") and layer.mlp.x_mask_up is not None:
                layer.mlp.x_mask_up.to_eval_mode()
            if hasattr(layer.mlp, "x_mask_down") and layer.mlp.x_mask_down is not None:
                layer.mlp.x_mask_down.to_eval_mode()
        if meta:
            print(f"Loaded joint_plus meta: {meta}")

    ppl_result = None
    if args.eval_ppl:
        print("\nEvaluating perplexity on wikitext-2-raw-v1 test split...")
        ppl_dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
        ppl_evaluator = PerplexityEvaluator(
            ppl_dataset,
            tokenizer,
            model_device,
            n_samples=args.ppl_n_samples,
        )
        ppl_result = ppl_evaluator.evaluate(model)
        print(f"Perplexity: {ppl_result}")

    # Evaluate task-by-task to support ARCQuant-style per-task overrides.
    print(f"\nEvaluating on {tasks_list}...")
    results_by_task = {}
    for task_name in tasks_list:
        task_batch_size = next(
            (
                value
                for key, value in batch_size_overrides.items()
                if task_name == key or task_name.startswith(key + "_")
            ),
            args.batch_size,
        )
        task_fewshot = fewshot_overrides.get(task_name, args.num_fewshot)
        print(
            f"\nInitializing lm_eval HFLM for {task_name} "
            f"(batch_size={task_batch_size}, num_fewshot={task_fewshot})..."
        )
        lm = HFLM(pretrained=model, tokenizer=tokenizer, batch_size=task_batch_size)
        task_result = simple_evaluate(
            lm,
            tasks=[task_name],
            num_fewshot=task_fewshot,
            batch_size=task_batch_size,
        )
        results_by_task[task_name] = task_result.get("results", {}).get(task_name, {})

    # Clean up sparsity hooks if they were applied
    if sparsity_hooks is not None:
        remove_activation_sparsity_hooks(sparsity_hooks)
    if hasattr(model, "_ln_sparsity_hooks"):
        remove_activation_sparsity_hooks(model._ln_sparsity_hooks)
        delattr(model, "_ln_sparsity_hooks")

    # Print results
    print("\n" + "="*60)
    print("Evaluation Results")
    print("="*60)
    if ppl_result is not None:
        print(f"\nppl:")
        print(f"  wikitext2: {ppl_result}")
    for task, metrics in results_by_task.items():
        print(f"\n{task}:")
        for k, v in metrics.items():
            if "stderr" not in k:
                print(f"  {k}: {v}")

    # Save results if output file is specified
    if args.output_file:
        output_path = Path(args.output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_payload = dict(results_by_task)
        if ppl_result is not None:
            output_payload["_ppl"] = {"wikitext2": ppl_result}
        with open(output_path, "w") as f:
            json.dump(output_payload, f, indent=2)
        print(f"\nResults saved to {args.output_file}")

    # Print summary
    print("\n" + "="*60)
    print("Summary")
    print("="*60)
    quant_str = f"w{args.w_bits}a{args.a_bits}" if quantize_enabled else "none"
    print(f"Config: model={args.model_path}, alpha={args.alpha}, quant={quant_str}")
    if ppl_result is not None:
        print(f"ppl wikitext2: {ppl_result}")
    for task, metrics in results_by_task.items():
        # Try to find the main metric (usually ends with /acc or similar)
        for k, v in metrics.items():
            if "stderr" not in k and (k.endswith("/acc") or "acc" in k.lower()):
                print(f"{task} {k}: {v}")

    print("="*60)


if __name__ == "__main__":
    main()
