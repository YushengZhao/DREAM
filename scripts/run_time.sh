#!/bin/bash

METHOD="nrgnn rtgnn unionnet pignn dream"
DATASETS="cora citeseer pubmed dblp amazonpho flickr"
NOISE_TYPES="uniform"
NOISE_RATES="0.3"
NUM_RUNS=10
BASE_SEED=3000
GPU_IDS="0 1 2 3 4"
PROCS_PER_GPU=2

echo "Running specified methods with 30% noise on ${DATASETS} ..."

python run_multi_gpu.py \
    --methods $METHOD \
    --datasets $DATASETS \
    --noise_type $NOISE_TYPES \
    --noise_rate $NOISE_RATES \
    --runs $NUM_RUNS \
    --seed $BASE_SEED \
    --gpu_ids $GPU_IDS \
    --procs_per_gpu $PROCS_PER_GPU

echo "Finished."