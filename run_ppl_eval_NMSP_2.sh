#!/usr/bin/env bash

export CUDA_VISIBLE_DEVICES=2

ALPHAS=(0.0 0.15 0.25 0.5 0.75 0.85 0.99)

for alpha in "${ALPHAS[@]}"; do
    echo "Running with alpha=${alpha}"
    python smoothquant/ppl_eval.py \
        --alpha ${alpha} \
        --model_path /gemini/code/checkpoints/models--meta-llama--Llama-3.1-8B/snapshots/d04e592bb4f6aa9cfee91e2e20afa771667e1d4b \
        --act_scales_path act_scales/llama-3.1-8B.pt \
        --smooth \
        --quantize \
        --act_sparsity 2:4 
done
