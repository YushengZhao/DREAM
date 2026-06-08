#!/bin/bash

METHOD="dream"
DATASETS="cora citeseer pubmed dblp amazonpho flickr"
NOISE_TYPES="pair uniform random"
NOISE_RATES="0.3"
NUM_RUNS=10
BASE_SEED=3000
DEVICE="cuda:0"

echo "Running DREAM with 30% noise on specified datasets (total_exp.py)..."

python total_exp.py \
    --methods $METHOD \
    --datasets $DATASETS \
    --noise_type $NOISE_TYPES \
    --noise_rate $NOISE_RATES \
    --runs $NUM_RUNS \
    --seed $BASE_SEED \
    --device $DEVICE

echo "Finished."