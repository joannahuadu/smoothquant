#!/usr/bin/env bash

export CUDA_VISIBLE_DEVICES=3

# Base target modules
base_targets=("lm_head" "up_proj" "k_proj" "v_proj" "o_proj")

# Only include q_proj and gate_proj for these layers
target_layers=(19 21 28 30 31)

config=$(IFS=,; echo "${base_targets[*]}")
for i in "${target_layers[@]}"; do
    config+=",layers.${i}.self_attn.q_proj,layers.${i}.mlp.gate_proj"
done

# Define target_modules configurations
# q_proj/gate_proj are layer-specific patterns to skip sensitive layers
declare -a target_modules_configs=(
    "${config}"
)

ALPHAS=(0.75 0.85 0.99)

for config in "${target_modules_configs[@]}"; do
    echo "========================================"
    echo "Evaluating with target_modules: ${config}"
    echo "========================================"
    for alpha in "${ALPHAS[@]}"; do
        echo "Running with alpha=${alpha}"
        python smoothquant/ppl_eval.py \
            --alpha ${alpha} \
            --model_path /gemini/code/checkpoints/models--meta-llama--Llama-3.1-8B/snapshots/d04e592bb4f6aa9cfee91e2e20afa771667e1d4b \
            --act_scales_path act_scales/llama-3.1-8B.pt \
            --smooth \
            --quantize \
            --act_sparsity 2:4 \
            --target_modules "${config}" \
            --no-weight_scoring \
            --act_sparsity_location post_quant
            
    done
    echo ""
    echo "Finished evaluation for: ${config}"
    echo ""
done
echo "All evaluations completed!"