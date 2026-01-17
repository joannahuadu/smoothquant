#!/usr/bin/env bash

export CUDA_VISIBLE_DEVICES=3

MODEL_PATH="/gemini/code/checkpoints/models--meta-llama--Llama-3.1-8B/snapshots/d04e592bb4f6aa9cfee91e2e20afa771667e1d4b"

# Define different target_modules combinations
declare -a target_modules_configs=(
    "lm_head"
    "lm_head,o_proj"
    "lm_head,q_proj"
    "lm_head,k_proj"
    "lm_head,v_proj"
)

# Run evaluation for each configuration
for config in "${target_modules_configs[@]}"; do
    echo "========================================"
    echo "Evaluating with target_modules: ${config}"
    echo "========================================"

    python smoothquant/ppl_eval.py \
        --model_path ${MODEL_PATH} \
        --act_sparsity 2:4 \
        --target_modules "${config}"

    echo ""
    echo "Finished evaluation for: ${config}"
    echo ""
done

echo "All evaluations completed!"
