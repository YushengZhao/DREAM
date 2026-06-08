#!/bin/bash

DATASETS="cora citeseer pubmed"
METHOD="dream"
NOISE_TYPE="uniform"
NOISE_RATE="0.3"
RUNS=10
DEVICE="cuda:0"
BASE_SEED=3000

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

echo "Starting ablation study for 30% Uniform noise..."
echo "=================================================="

for i in "${!ABLATION_FLAGS[@]}"; do
    ABLATION_FLAG="${ABLATION_FLAGS[$i]}"
    ABLATION_NAME="${ABLATION_NAMES[$i]}"

    echo ""
    echo "--- Running Ablation: $ABLATION_NAME ---"
    echo "Command: python total_exp.py --methods $METHOD --datasets $DATASETS --noise_type $NOISE_TYPE --noise_rate $NOISE_RATE --runs $RUNS --device $DEVICE --seed $BASE_SEED $ABLATION_FLAG"
    echo "---"

    python total_exp.py \
        --methods $METHOD \
        --datasets $DATASETS \
        --noise_type $NOISE_TYPE \
        --noise_rate $NOISE_RATE \
        --runs $RUNS \
        --device $DEVICE \
        --seed $BASE_SEED \
        $ABLATION_FLAG

    echo "--- Finished Ablation: $ABLATION_NAME ---"
    echo "=================================================="
    sleep 2
done

echo "Ablation study finished."