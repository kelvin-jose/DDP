# DDP for Multi-GPU Training

This repository contains a more complex implementation of Distributed Data Parallel (DDP) using PyTorch in a multi-GPU environment. 

## Overview
This repository contains a PyTorch implementation for training a small Llama model using Distributed Data Parallel (DDP). The model is designed to efficiently process and learn from a toy dataset, leveraging multiple GPUs for accelerated training. The code utilizes the litgpt modules for defining the model architecture and configuration, along with a custom DistributedDataLoader that facilitates data loading in a distributed environment. The implementation focuses on key aspects of training neural networks, including model configuration, data loading, optimization, and evaluation. It is suitable for researchers and practitioners interested in experimenting with transformer-based models in a distributed setting.

## Features
- Distributed Training: Seamlessly integrates DDP to utilize multiple GPUs for faster training.
- Customizable Model Configuration: Users can easily modify hyperparameters such as the number of layers, attention heads, embedding dimensions, and more.
- Efficient Data Loading: Implements a DistributedDataLoader to handle data loading across multiple processes.
- Training and Validation Loops: Provides structured loops for both training and validation phases, including loss computation and logging.
- Compatibility: Designed to work with PyTorch and CUDA-enabled GPUs.

To run,
```bash
   torchrun --standalone --nproc-per-node=4 --nnodes=1 train.py --train_file "/mnt/train/train.pkl" --valid_file "/mnt/val/val.pkl"
   ```

## Notes

- Ensure you have set up your GPU environment correctly with CUDA installed.
- The script assumes the use of NVIDIA GPUs and the `nccl` backend for distributed training.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

Feel free to customize the `train.py` script to fit your specific use case and data. If you encounter any issues or have suggestions for improvement, please open an issue or submit a pull request.