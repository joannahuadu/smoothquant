#!/usr/bin/env bash

export CUDA_VISIBLE_DEVICES=3

MODEL_PATH="/gemini/code/checkpoints/models--meta-llama--Llama-3.1-8B/snapshots/d04e592bb4f6aa9cfee91e2e20afa771667e1d4b"
ACT_SCALES_PATH="act_scales/llama-3.1-8B.pt"

ALPHAS=(0.0 0.15 0.25 0.5 0.75 0.85 0.99)

for alpha in "${ALPHAS[@]}"; do
    echo "Running with alpha=${alpha}"
    python benchmark_eval.py \
        --alpha ${alpha} \
        --model_path ${MODEL_PATH} \
        --act_scales_path ${ACT_SCALES_PATH} \
        --w_bits 8 \
        --a_bits 8 \
        --quantize \
        --tasks arc_challenge \
        --num_fewshot 0 \
        --batch_size 32 \
        --output_file /gemini/code/NMSparsity/smoothquant/results/models--meta-llama--Llama-3.1-8B_SQw8a8_alpha${alpha}_AC.json

done