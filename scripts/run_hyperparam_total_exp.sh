#!/bin/bash

DATASET="cora citeseer pubmed"
METHOD="dream"
NOISE_RATE="0.3"
NOISE_TYPES_FOR_RUN="pair uniform random"

NUM_RUNS=10
BASE_SEED=3000
DEVICE="cuda:0"

PYTHON_SCRIPT_PATH="./total_exp.py"
DEFAULT_K_TOPOLOGY=10
DEFAULT_K_PROXIMITY=15

K_TOPOLOGY_SWEEP_VALUES=(1 5 10 20 30 40)
K_PROXIMITY_SWEEP_VALUES=(1 5 10 15 20 30)

echo "Starting hyperparameter sweep for ${METHOD} on ${DATASET}..."
echo "Noise Rate: ${NOISE_RATE}, Noise Types: ${NOISE_TYPES_FOR_RUN}"
echo "Python script: ${PYTHON_SCRIPT_PATH}"
echo "Number of statistical runs per point: ${NUM_RUNS}"
echo "-----------------------------------------------------"

echo ""
echo "====================================================="
echo "Part 1: Sweeping k_topology (k_proximity fixed at ${DEFAULT_K_PROXIMITY})"
echo "====================================================="
for k_l_val in "${K_TOPOLOGY_SWEEP_VALUES[@]}"; do
  echo ""
  echo "==> k_topology = ${k_l_val}, k_proximity = ${DEFAULT_K_PROXIMITY}"
  echo "Command: python run_multi_gpu.py --methods $METHOD --datasets $DATASET --noise_type $NOISE_TYPE --noise_rate $NOISE_RATE --k_topology_override $k_l_val --k_proximity_override $DEFAULT_K_PROXIMITY --runs $RUNS --seed $BASE_SEED --device $DEVICE"
  echo "-----------------"

  python ${PYTHON_MULTI_GPU_SCRIPT_PATH} \
    --datasets ${DATASET} \
    --methods ${METHOD} \
    --noise_type ${NOISE_TYPES_FOR_RUN} \
    --noise_rate ${NOISE_RATE} \
    --k_topology_override ${k_l_val} \
    --k_proximity_override ${DEFAULT_K_PROXIMITY} \
    --runs ${NUM_RUNS} \
    --seed ${BASE_SEED} \
    --device ${DEVICE}

  echo "-----------------"
  echo "==> Finished for k_topology = ${k_l_val}"
done

echo ""
echo "====================================================="
echo "Part 2: Sweeping k_proximity (k_topology fixed at ${DEFAULT_K_TOPOLOGY})"
echo "====================================================="
for k_s_c_val in "${K_PROXIMITY_SWEEP_VALUES[@]}"; do
  echo ""
  echo "==> k_topology = ${DEFAULT_K_TOPOLOGY}, k_proximity = ${k_s_c_val}"
  echo "Command: python run_multi_gpu.py --methods $METHOD --datasets $DATASET --noise_type $NOISE_TYPE --noise_rate $NOISE_RATE --k_topology_override $DEFAULT_K_TOPOLOGY --k_proximity_override $k_s_c_val --runs $RUNS --seed $BASE_SEED --device $DEVICE"
  echo "-----------------"

  python ${PYTHON_MULTI_GPU_SCRIPT_PATH} \
    --datasets ${DATASET} \
    --methods ${METHOD} \
    --noise_type ${NOISE_TYPES_FOR_RUN} \
    --noise_rate ${NOISE_RATE} \
    --k_topology_override ${DEFAULT_K_TOPOLOGY} \
    --k_proximity_override ${k_s_c_val} \
    --runs ${NUM_RUNS} \
    --seed ${BASE_SEED} \
    --device ${DEVICE}

  echo "-----------------"
  echo "==> Finished for k_proximity = ${k_s_c_val}"
done

echo ""
echo "====================================================="
echo "Hyperparameter sweep finished."
echo "Check the output files from ResultLogger to plot the accuracy curves."
echo "====================================================="