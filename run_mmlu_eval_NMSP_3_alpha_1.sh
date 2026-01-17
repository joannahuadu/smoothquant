#!/usr/bin/env bash

export CUDA_VISIBLE_DEVICES=1

MODEL_PATH="/gemini/code/checkpoints/models--meta-llama--Llama-3.1-8B/snapshots/d04e592bb4f6aa9cfee91e2e20afa771667e1d4b"
ACT_SCALES_PATH="act_scales/llama-3.1-8B.pt"

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
        python benchmark_eval.py \
            --alpha ${alpha} \
            --model_path ${MODEL_PATH} \
            --act_scales_path ${ACT_SCALES_PATH} \
            --w_bits 8 \
            --a_bits 8 \
            --quantize \
            --act_sparsity 2:4 \
            --target_modules "${config}" \
            --no-weight_scoring \
            --tasks mmlu \
            --num_fewshot 0 \
            --output_file /gemini/code/NMSparsity/smoothquant/results/models--meta-llama--Llama-3.1-8B_SQw8a8_alpha${alpha}_MMLU_NMSP_3.json \
            --act_sparsity_location post_quant \
            --invert_scales

    done
    echo ""
    echo "Finished evaluation for: ${config}"
    echo ""
done
echo "All evaluations completed!"
