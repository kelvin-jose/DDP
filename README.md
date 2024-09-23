# Distributed Data Parallel (DDP)

Distributed Data Parallel (DDP) is a feature in PyTorch designed to facilitate efficient training of deep learning models across multiple GPUs and machines. It implements data parallelism at the module level, allowing for the distribution of model training tasks over multiple processes, which can significantly speed up the training process.

## Key Features of DDP

- **Multi-Process Training**: 
  - DDP operates by spawning multiple processes, each handling a separate GPU. This allows for both single-node and multi-node training setups.

- **Gradient Synchronization**: 
  - During the backward pass, DDP synchronizes gradients across all processes, ensuring that all model replicas are updated consistently.

- **Performance Benefits**: 
  - Compared to the older `DataParallel` method, DDP is generally faster and more efficient because it avoids the Global Interpreter Lock (GIL) contention found in multi-threaded setups.