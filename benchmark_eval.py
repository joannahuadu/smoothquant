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
from pathlib import Path

from transformers import AutoTokenizer, AutoModelForCausalLM
from lm_eval import simple_evaluate
from lm_eval.models.huggingface import HFLM

from smoothquant.smooth import smooth_lm
from smoothquant.fake_quant import quantize_model
from smoothquant.sparse import apply_activation_sparsity_to_model, remove_activation_sparsity_hooks


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
        "--output_file",
        type=str,
        default=None,
        help="Path to save evaluation results as JSON",
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
    tasks_list = [t.strip() for t in args.tasks.split(",")]

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
        device_map="auto",
        trust_remote_code=True,
    )
    print(f"Model loaded on {model.device}")

    # Apply SmoothQuant smoothing if activation scales are provided
    if args.act_scales_path:
        if os.path.exists(args.act_scales_path):
            print(f"\nLoading activation scales from {args.act_scales_path}")
            act_scales = torch.load(args.act_scales_path)
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

    # Create HFLM evaluator
    print("\nInitializing lm_eval HFLM...")
    lm = HFLM(pretrained=model, tokenizer=tokenizer)

    # Run evaluation
    print(f"\nEvaluating on {tasks_list}...")
    results = simple_evaluate(
        lm,
        tasks=tasks_list,
        num_fewshot=args.num_fewshot,
        batch_size=args.batch_size,
    )

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
    for task, metrics in results.get("results", {}).items():
        print(f"\n{task}:")
        for k, v in metrics.items():
            if "stderr" not in k:
                print(f"  {k}: {v}")

    # Save results if output file is specified
    if args.output_file:
        output_path = Path(args.output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(results["results"], f, indent=2)
        print(f"\nResults saved to {args.output_file}")

    # Print summary
    print("\n" + "="*60)
    print("Summary")
    print("="*60)
    quant_str = f"w{args.w_bits}a{args.a_bits}" if quantize_enabled else "none"
    print(f"Config: model={args.model_path}, alpha={args.alpha}, quant={quant_str}")
    for task, metrics in results.get("results", {}).items():
        # Try to find the main metric (usually ends with /acc or similar)
        for k, v in metrics.items():
            if "stderr" not in k and (k.endswith("/acc") or "acc" in k.lower()):
                print(f"{task} {k}: {v}")

    print("="*60)


if __name__ == "__main__":
    main()
