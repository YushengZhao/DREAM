# Official Implementation for "DREAM: Dual-Standard Semantic Homogeneity with Dynamic Optimization for Graph Learning with Label Noise"

This repository contains the official PyTorch implementation for the paper: **"DREAM: Dual-Standard Semantic Homogeneity with Dynamic Optimization for Graph Learning with Label Noise"**.

Our work introduces **DREAM (Dual-Standard Semantic Homogeneity with Dynamic Optimization for Graph Learning with Label Noise)**, a novel framework designed for robust graph neural network training on graphs with noisy labels. This implementation provides the code to reproduce all experiments and results presented in the paper, including main results, ablation studies, and hyperparameter sensitivity analysis.

## Project Structure

The repository is organized as follows to ensure clarity and ease of navigation:

```
DREAM/
├── config/             # YAML configuration files for datasets and models.
├── data/               # Directory for storing graph datasets.
├── log/                # Directory for saving experimental logs and results.
├── plots/              # Directory for saving generated plots.
├── predictor/          # Implementations of our DREAM model and all baseline models.
│   ├── DREAM_Predictor.py
│   └── ... (other predictors)
├── scripts/            # Shell scripts for easily reproducing all experiments.
│   ├── run_total_exp.sh
│   ├── run_multi_gpu.sh
│   ├── run_ablation_total_exp.sh
│   └── ...
├── utils/              # Utility modules for dataloading, logging, etc.
├── requirements.txt    # Python package dependencies.
├── total_exp.py        # Main script for running experiments on a single GPU.
├── run_multi_gpu.py    # Main script for running experiments on multiple GPUs.
└── README.md
```

## 1. Setup Environment

### Requirements

- Python >= 3.8
- PyTorch >= 2.0.0
- PyTorch Geometric

### Hardware

The experiments in the paper were conducted on an **NVIDIA H800 GPU**.

### Installation Steps

We recommend using `conda` or `venv` to create an isolated Python environment.

1. **Download the Code:**

   Click the Download Repository button at the top of this page to download the source code as a ZIP file. Then, unzip it.

   ```bash
   # After unzipping, navigate into the project directory
   cd DREAM-main  # The folder name might vary, adjust if necessary
   ```

2. **Create a virtual environment (optional but recommended):**

   ```bash
   # Using conda
   conda create -n dream_env python=3.12
   conda activate dream_env
   
   # Or using venv
   # python -m venv venv
   # source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install PyTorch and PyTorch Geometric (Crucial Step):**

   The performance and correctness of this project heavily depend on the correct installation of PyTorch and its ecosystem tailored to your CUDA version. **Please do not simply `pip install torch`**.

   Visit the official PyTorch website to get the correct installation command for your system:
   [**https://pytorch.org/get-started/locally/**](https://pytorch.org/get-started/locally/)

   For example, for CUDA 12.1, the command might be:

   ```bash
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
   ```

   After installing PyTorch, install the corresponding PyTorch Geometric libraries.

   ```bash
   pip install torch_geometric
   # Optional, for full compatibility and speed:
   pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.5.0+cu121.html # Adjust the torch version and CUDA here
   ```

4. **Install remaining dependencies:**

   Once PyTorch is correctly installed, install the other required packages using the provided `requirements.txt` file.

   ```bash
   pip install -r requirements.txt
   ```

## 2. Datasets

The datasets used in our paper (e.g., **cora, citeseer, pubmed, dblp, a-photo, flickr**) will be automatically downloaded and processed when the experiment scripts are run for the first time. They will be stored in the `./data/` directory. No manual download is required.

## 3. Reproducing Experiments

We provide comprehensive shell scripts in the `scripts/` directory to reproduce all experimental results from our paper easily.

### A. Reproduce Main Results

To reproduce the main performance comparison results (as shown in **Table 1**), you can run the `total_exp.py` script, which iterates through all specified datasets and methods.

**For Single-GPU Execution:**
The following script will run our proposed `dream` model on the main datasets with various noise types.

```bash
bash scripts/run_total_exp.sh
```

This script calls `total_exp.py`. You can modify the script to change datasets, methods, or noise configurations.

**For Multi-GPU Execution (Faster):**
If you have multiple GPUs, you can significantly speed up the experiments by using `run_multi_gpu.py`.

```bash
bash scripts/run_multi_gpu.sh
```

Please modify the `GPU_IDS` and `PROCS_PER_GPU` variables inside the script to match your hardware configuration.

### B. Reproduce Ablation Studies

To reproduce the results for the ablation study (as reported in **Ablation Studies section**), run the corresponding ablation script.

**For Single-GPU Execution:**

```bash
bash scripts/run_ablation_total_exp.sh
```

This script will test different variants of our DREAM model by disabling specific components one by one.

**For Multi-GPU Execution:**

```bash
bash scripts/run_ablation_multi_gpu.sh
```

### C. Reproduce Hyperparameter Sensitivity Analysis

To reproduce the hyperparameter sensitivity analysis on `k_topology` and `k_proximity` (as shown in **Figure 2**), use the hyperparameter sweep scripts.

**For Single-GPU Execution:**

```bash
bash scripts/run_hyperparam_total_exp.sh
```

**For Multi-GPU Execution:**

```bash
bash scripts/run_hyperparam_multi_gpu.sh
```

### D. Running Custom Experiments Manually

Beyond the provided shell scripts, you can run experiments with custom configurations directly from the command line. This is useful for debugging, quick tests, or exploring settings not covered in the main scripts.

#### Single-GPU Experiment Loops (`total_exp.py`)

The `total_exp.py` script is designed to run a complete experimental loop over specified methods, datasets, and noise settings on a **single GPU**.

**Example:** To run our `dream` model and the `gcn` baseline on the `cora` dataset with 30% uniform noise for 5 statistical runs:

```bash
python total_exp.py \
    --methods dream gcn \
    --datasets cora \
    --noise_type uniform \
    --noise_rate 0.3 \
    --runs 5 \
    --device cuda:0
```

To see all available options, run:

```bash
python total_exp.py --help
```

#### Multi-GPU Experiment Loops (`run_multi_gpu.py`)

The `run_multi_gpu.py` script accelerates the process by distributing the experiment loops across **multiple GPUs** and processes.

**Example:** To run the same experiment as above but distributed across GPUs 0 and 1, with 2 processes per GPU:

```bash
python run_multi_gpu.py \
    --methods dream gcn \
    --datasets cora \
    --noise_type uniform \
    --noise_rate 0.3 \
    --runs 5 \
    --gpu_ids 0 1 \
    --procs_per_gpu 2
```

To see all available options, run:

```bash
python run_multi_gpu.py --help
```

#### Single Trial & Hyperparameter Tuning (`single_exp.py`)

The `single_exp.py` script is designed for two main purposes:

1.  Running a **single trial** of one specific experiment configuration, which is ideal for fine-grained debugging.
2.  Serving as the entry point for automated hyperparameter tuning using **Microsoft's NNI (Neural Network Intelligence)**.

**Example (Standalone Single Trial):**

```bash
python single_exp.py \
    --method dream \
    --dataset cora \
    --noise_type uniform \
    --noise_rate 0.3 \
    --device cuda:0 \
    --seed 3000
```

This script is also integrated with NNI. When used in an NNI experiment, it will automatically receive hyperparameters from the NNI tuner. To see all standalone options, run `python single_exp.py --help`.

### E. Experiment Outputs

All experimental results are systematically logged into the `./log/` directory. For each complete experimental execution (e.g., running `run_total_exp.sh`), a new set of log files is created with a timestamp prefix (e.g., `2024-05-21_14-30-00_...`).

You will find three types of files for each execution:

- **`[timestamp].txt` (Raw Log):** A human-readable text file that contains a running log of results for each experiment combination as it completes. This file is useful for monitoring the progress of long-running experiments.

- **`[timestamp].xlsx` (Detailed Excel Report):** A comprehensive Excel workbook. This is the **recommended file for detailed analysis**. It contains multiple sheets, each dedicated to a specific performance metric (e.g., `test_acc_main`, `aclt`, `time`), showing the mean and standard deviation across all runs. This file is updated and overwritten as new results become available.

- **`[timestamp].tex` (LaTeX Table):** A LaTeX-formatted table containing the main test accuracies (`mean ± std`). This file can be directly included in a LaTeX document and is also updated and overwritten as the experiment progresses.

---

Should you encounter any issues while trying to reproduce our results, please feel free to contact us through the official review communication channel.