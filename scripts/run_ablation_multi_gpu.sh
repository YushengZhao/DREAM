#!/bin/bash

DATASETS="cora citeseer pubmed"
METHOD="dream"
NOISE_TYPE="uniform"
NOISE_RATE="0.3"
RUNS=10
BASE_SEED=3000
GPU_IDS="0 1"
PROCS_PER_GPU=2

ABLATION_FLAGS=(
    "--ablation_disable_topology_aware_anchors"
    "--ablation_disable_proximity_aware_anchors"
    "--ablation_no_temp_scale"
    "--ablation_combine_topk"
    "--ablation_topk_from_all"
    ""
)

ABLATION_NAMES=(
    "V1"
    "V2"
    "V3"
    "V4"
    "V5"
    "ours (baseline)"
)

echo "Starting ablation study for 30% Uniform noise (Multi-GPU)..."
echo "=================================================="

for i in "${!ABLATION_FLAGS[@]}"; do
    ABLATION_FLAG="${ABLATION_FLAGS[$i]}"
    ABLATION_NAME="${ABLATION_NAMES[$i]}"

    echo ""
    echo "--- Running Ablation (Multi-GPU): $ABLATION_NAME ---"
    echo "Command: python run_multi_gpu.py --methods $METHOD --datasets $DATASETS --noise_type $NOISE_TYPE --noise_rate $NOISE_RATE --runs $RUNS --seed $BASE_SEED --gpu_ids $GPU_IDS --procs_per_gpu $PROCS_PER_GPU $ABLATION_FLAG"
    echo "---"

    python run_multi_gpu.py \
        --methods $METHOD \
        --datasets $DATASETS \
        --noise_type $NOISE_TYPE \
        --noise_rate $NOISE_RATE \
        --runs $RUNS \
        --seed $BASE_SEED \
        --gpu_ids $GPU_IDS \
        --procs_per_gpu $PROCS_PER_GPU \
        $ABLATION_FLAG

    echo "--- Finished Ablation (Multi-GPU): $ABLATION_NAME ---"
    echo "=================================================="
    sleep 5
done

echo "Ablation study (Multi-GPU) finished."