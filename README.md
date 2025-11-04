# NanoChat Text Generation

A simple, standalone CPU-compatible text generation tool using the nanochat model from HuggingFace. This project provides an easy way to run the nanochat model locally without requiring GPU hardware.

## Overview

This project includes:
- A standalone text generation script that works on CPU
- Automatic model downloading from HuggingFace (https://huggingface.co/sdobson/nanochat)
- Simple launch script for easy execution
- PyTorch 2.6 compatibility fixes

## Requirements

- **Python**: 3.10 or higher
- **uv**: Python package manager (install from [astral.sh/uv](https://astral.sh/uv))
  ```bash
  curl -LsSf https://astral.sh/uv/install.sh | sh
  ```
- **Internet connection**: Required for initial model download (~2GB)

## Installation

1. Clone or download this repository
2. Make the launch script executable:
   ```bash
   chmod +x launch.sh
   ```

That's it! The script will automatically handle dependency installation via `uv`.

## Usage

### Quick Start

Simply run the launch script:

```bash
./launch.sh
```

This will:
1. Download the nanochat model from HuggingFace (first time only, ~2GB)
2. Generate text based on the default prompt "Tell me about dogs."

### Customization

Edit `launch.sh` to customize the prompt or generation parameters:

```bash
uv run --refresh python "$SCRIPT_DIR/generate_cpu.py" \
  --model-dir /tmp/nanochat \
  --prompt "Your custom prompt here" \
  --max-tokens 200 \
  --temperature 0.8 \
  --top-k 50
```

### Command Line Options

The `generate_cpu.py` script supports the following arguments:

- `--model-dir`: Path to the model directory (default: `/tmp/nanochat`)
- `--prompt`: Text prompt for generation (required)
- `--max-tokens`: Maximum number of tokens to generate (default: 100)
- `--temperature`: Sampling temperature (default: 0.8)
  - Lower values (0.1-0.5) = more focused, deterministic
  - Higher values (0.8-1.5) = more creative, diverse
- `--top-k`: Top-k sampling parameter (default: 50)
  - Limits sampling to the top K most likely tokens

### Direct Script Execution

You can also run the generation script directly:

```bash
uv run generate_cpu.py \
  --model-dir /tmp/nanochat \
  --prompt "Once upon a time" \
  --max-tokens 150 \
  --temperature 0.7
```

## Project Structure

```
nanochat/
├── launch.sh              # Main launch script (downloads model and runs generation)
├── download_model.py      # Script to download model files from HuggingFace
├── generate_cpu.py        # Standalone text generation script
└── README.md             # This file
```

### File Descriptions

- **`launch.sh`**: Convenience script that handles model downloading and text generation in one command
- **`download_model.py`**: Downloads the nanochat model files from HuggingFace using `huggingface_hub`, handling Git LFS automatically
- **`generate_cpu.py`**: Standalone text generation script that includes a minimal GPT implementation. Works on CPU without requiring the full nanochat package

## Model Information

- **Model**: nanochat by sdobson
- **Size**: ~2GB
- **Repository**: [sdobson/nanochat on HuggingFace](https://huggingface.co/sdobson/nanochat)
- **Storage**: Model files are downloaded to `/tmp/nanochat/` (can be changed in the scripts)

## Troubleshooting

### Model Download Issues

If the model download fails:
- Check your internet connection
- Ensure you have enough disk space (~3GB free)
- Try manually running: `uv run download_model.py`

### PyTorch Compatibility

This project includes fixes for PyTorch 2.6 compatibility:
- The `weights_only=False` parameter is used when loading model weights
- This is safe since the model is from a trusted source (HuggingFace)

### Memory Issues

If you encounter memory errors:
- The model runs on CPU, which may be slower but uses less memory than GPU
- Close other applications to free up RAM
- Try reducing `--max-tokens` to generate shorter text

### Cache Issues

If you encounter stale cache issues with `uv`:
```bash
rm -rf ~/.cache/uv/environments-v2/*
```

## How It Works

1. **Model Download**: The `download_model.py` script uses `huggingface_hub` to download the model files, automatically handling Git LFS pointers.

2. **Text Generation**: The `generate_cpu.py` script:
   - Loads the model from the downloaded files
   - Converts bfloat16 weights to float32 for CPU compatibility
   - Uses a minimal GPT implementation (no external dependencies beyond torch/tiktoken)
   - Generates text token by token using top-k sampling

3. **Dependencies**: All dependencies are automatically managed by `uv` using PEP 723 script metadata in the Python files.

## Customization Examples

### Generate a Story
```bash
uv run generate_cpu.py \
  --model-dir /tmp/nanochat \
  --prompt "In a distant galaxy, there was a brave explorer" \
  --max-tokens 200 \
  --temperature 0.9
```

### Generate Technical Content
```bash
uv run generate_cpu.py \
  --model-dir /tmp/nanochat \
  --prompt "Python is a programming language that" \
  --max-tokens 150 \
  --temperature 0.6 \
  --top-k 30
```

### Creative Writing
```bash
uv run generate_cpu.py \
  --model-dir /tmp/nanochat \
  --prompt "The sunset painted the sky in hues of" \
  --max-tokens 250 \
  --temperature 1.0 \
  --top-k 60
```

## License

This project uses the nanochat model from HuggingFace. Please refer to the original model repository for licensing information.

## Credits

- Model: [sdobson/nanochat](https://huggingface.co/sdobson/nanochat)
- Original generation script: Based on [Simon Willison's gist](https://gist.github.com/simonw/912623bf00d6c13cc0211508969a100a)
- Fixed for PyTorch 2.6 compatibility

