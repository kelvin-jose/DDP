# Distributed Data Parallel (DDP)

Distributed Data Parallel (DDP) is a feature in PyTorch designed to facilitate efficient training of deep learning models across multiple GPUs and machines. It implements data parallelism at the module level, allowing for the distribution of model training tasks over multiple processes, which can significantly speed up the training process.

## Key Features of DDP

- **Multi-Process Training**: 
  - DDP operates by spawning multiple processes, each handling a separate GPU. This allows for both single-node and multi-node training setups.

- **Gradient Synchronization**: 
  - During the backward pass, DDP synchronizes gradients across all processes, ensuring that all model replicas are updated consistently.

- **Performance Benefits**: 
  - Compared to the older `DataParallel` method, DDP is generally faster and more efficient because it avoids the Global Interpreter Lock (GIL) contention found in multi-threaded setups.

## Goals of this project

This project was intended to teach myself how to work with DDP systems in general. For this, I tried implementing two projects one is a small naive neural network with added DDP features which you can find inside ```sample/``` directory. Inside the directory, you will find a single python file that contains all the necessary code to load, train and test a simple network. Inside ```main/``` directory, you will find a slightly complicated example of mini Llama training scripts built with help of [litgpt](https://github.com/Lightning-AI/litgpt) modules. Enjoy reading them.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.