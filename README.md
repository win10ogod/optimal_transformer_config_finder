# Optimal Transformer Configuration Finder

Welcome to the **Optimal Transformer Configuration Finder**! 🎉

This repository helps you find the optimal configuration for a Transformer model based on your GPU specifications, context length, batch size, and other parameters. By selecting the best configuration, you can maximize the performance of your deep learning tasks while efficiently utilizing your GPU resources.

## Features

- 📊 **GPU Detection**: Automatically detects GPU specifications via `nvidia-smi`.
- ⚙️ **Flexible Configuration**: Allows you to specify batch size, reserved VRAM, vocabulary size, and context length.
- 🚀 **Performance-Oriented**: Finds an optimal Transformer configuration that maximizes performance while fitting within your GPU memory.
- 🛠️ **Customizable**: Easily modify the parameters for different training needs.

## How It Works

1. **Detect GPU Specs**: Use `nvidia-smi` to detect GPU name, total memory, and clock speed.
2. **Calculate Config**: Iterate through possible configurations to find the optimal one that best utilizes GPU resources.
3. **Output Results**: Print the optimal configuration (number of layers, dimension size, etc.) for your Transformer model.

## Usage

### Prerequisites

- Python 3.6+
- NVIDIA GPU with `nvidia-smi` installed
- `subprocess` (built-in Python module)

### Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/yourusername/optimal-transformer-config-finder.git
    ```
2. Navigate to the project directory:
    ```bash
    cd optimal-transformer-config-finder
    ```

### Running the Script


2. Execute the script:
    ```bash
    python optimal_transformer_config_finder.py
    ```
3. The output will be similar to:
    ```
    Detected GPU: NVIDIA GeForce RTX 3090
    Total Memory: 24576.0 MB
    Max SM Clock: 1.695 GHz
    Tensor Cores: 64
    Optimal Transformer Configuration: dim=1024, n_layers=12, n_heads=16, context_length=512, vocab_size=32000, batch_size=32
    ```

### Customizing Parameters

You can customize the parameters directly in `optimal_transformer_config_finder.py` or modify the function arguments.

#### Script Parameters

- **Context Length**: The maximum sequence length. Default is `512`.
- **Vocabulary Size**: Size of the vocabulary. Default is `32000`.
- **Reserved VRAM**: Amount of GPU memory (in bytes) to reserve. Default is `1GB`.
- **Batch Size**: The size of each training batch. Default is `32`.
- **Training Factor**: A multiplier to simulate training memory usage. Default is `1.0`.

#### Example Usage with Custom Parameters

```python
optimal_config = calculate_optimal_transformer_config(
    gpu_specs,
    context_length=1024,
    training_factor=1.2,
    vocab_size=50000,
    reserved_vram=2 * 1024 * 1024 * 1024,  # Reserve 2GB
    batch_size=64
)
```

## Contributing

Contributions are welcome! Feel free to open a pull request or file an issue.

### How to Contribute

1. Fork the repository.
2. Create a new branch (`git checkout -b feature-branch-name`).
3. Make your changes.
4. Commit your changes (`git commit -m 'Add a new feature'`).
5. Push to the branch (`git push origin feature-branch-name`).
6. Open a pull request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Thanks to the OpenAI team for inspiring this project.
- The Python community for providing excellent tools and libraries.

## Contact

For any questions or suggestions, feel free to reach out via [GitHub Issues](https://github.com/yourusername/optimal-transformer-config-finder/issues).

Happy optimizing! 🚀
