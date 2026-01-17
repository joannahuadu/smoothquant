#!/usr/bin/env bash

# export CUDA_VISIBLE_DEVICES=2

# MODEL_PATH="/gemini/code/checkpoints/models--meta-llama--Llama-3.1-8B/snapshots/d04e592bb4f6aa9cfee91e2e20afa771667e1d4b"

# # Define different target_modules combinations
# declare -a target_modules_configs=(
#     # "lm_head,q_proj,k_proj,v_proj"
#     # "lm_head,gate_proj"
#     # "lm_head,up_proj"
#     # "lm_head,down_proj"
#     "lm_head,gate_proj,up_proj,k_proj,v_proj,o_proj"
# )

# # Run evaluation for each configuration
# for config in "${target_modules_configs[@]}"; do
#     echo "========================================"
#     echo "Evaluating with target_modules: ${config}"
#     echo "========================================"

#     python smoothquant/ppl_eval.py \
#         --model_path ${MODEL_PATH} \
#         --act_sparsity 2:4 \
#         --target_modules "${config}"

#     echo ""
#     echo "Finished evaluation for: ${config}"
#     echo ""
# done

# echo "All evaluations completed!"

#!/usr/bin/env bash

export CUDA_VISIBLE_DEVICES=2

MODEL_PATH="/gemini/code/checkpoints/models--meta-llama--Llama-3.1-8B/snapshots/d04e592bb4f6aa9cfee91e2e20afa771667e1d4b"

# Base target modules
base_targets=("lm_head" "up_proj" "k_proj" "v_proj" "o_proj" "q_proj" "gate_proj")

# Only include q_proj and gate_proj for these layers
# target_layers=(19 21 28 30 31)
target_layers=()

config=$(IFS=,; echo "${base_targets[*]}")
for i in "${target_layers[@]}"; do
    config+=",layers.${i}.self_attn.q_proj,layers.${i}.mlp.gate_proj"
done

# Define target_modules configurations
# q_proj/gate_proj are layer-specific patterns to skip sensitive layers
declare -a target_modules_configs=(
    "${config}"
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
