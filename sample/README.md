# DDP Sample for Multi-GPU Training

This repository contains a sample implementation of Distributed Data Parallel (DDP) using PyTorch in a multi-GPU environment. The script `ddp_sample.py` demonstrates how to set up and train a simple neural network model using multiple GPUs to speed up the training process.

## Table of Contents

- [Overview](#overview)
- [Requirements](#requirements)
- [Setup](#setup)
- [Script Explanation](#script-explanation)
- [Notes](#notes)
- [License](#license)

## Overview

This script demonstrates:
1. Loading data using PyTorch.
2. Defining a simple neural network model.
3. Setting up Distributed Data Parallel (DDP) for multi-GPU training.
4. Training and validating the model.

## Requirements

- Python 3.7 or higher
- PyTorch 1.9.0 or higher
- CUDA-compatible GPU(s)
- `numpy`

To install the required Python packages, you can use the following command:

```bash
pip install torch numpy
```

## Setup

1. **Prepare Data:**
   Ensure you have the following data files in the `/mnt/ddp/sample/` directory:
   - `X_train.npy`
   - `y_train.npy`
   - `X_val.npy`
   - `y_val.npy`

   These files should be in `.npy` format containing your training and validation data.

2. **Environment Variables:**
   Set the following environment variables for DDP:
   - `RANK`: The global rank of the process.
   - `LOCAL_RANK`: The local rank of the process on the node.
   - `WORLD_SIZE`: Total number of processes (i.e., number of GPUs).

   These can be set using a script or within your job scheduler if running on a cluster.

3. **Run the Script:**
   Execute the script using a distributed launch utility. For example, with `torchrun`, use:

   ```bash
   torchrun --standalone --nproc-per-node=4 --nnodes=1 ddp_sample.py
   ```

   Replace `nproc-per-node` with the number of GPUs you want to use.

In this example, the script will use 4 GPUs on a single node.

## Script Explanation

### Data Loading

The script loads training and validation data from `.npy` files and uses PyTorch's `Dataset` and `DataLoader` for batching. 

### Model Definition

A simple neural network model (`SimpleModel`) with one linear layer followed by a softmax activation is defined.

### Distributed Setup

- The script checks if it's running in a distributed environment by checking the `RANK` environment variable.
- Initializes the process group for DDP with `nccl` backend.
- Sets the GPU device based on the local rank.

### Training Loop

- The model is trained for 100 epochs.
- During training, the loss is computed, gradients are backpropagated, and the optimizer updates the model parameters.
- Validation is performed at the end of each epoch, and the validation loss is computed.

### DDP Configuration

- The `DistributedDataParallel` wrapper is used to parallelize the model training across GPUs.

### Cleanup

- The process group is destroyed at the end of the script to clean up resources.

## Notes

- Ensure you have set up your GPU environment correctly with CUDA installed.
- The script assumes the use of NVIDIA GPUs and the `nccl` backend for distributed training.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

Feel free to customize the `ddp_sample.py` script to fit your specific use case and data. If you encounter any issues or have suggestions for improvement, please open an issue or submit a pull request.