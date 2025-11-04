# Get script directory first, before changing to /tmp
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# Download model files using huggingface_hub (handles Git LFS automatically)
cd "$SCRIPT_DIR"
MODEL_FILE="/tmp/nanochat/model_000650.pt"
if [ ! -f "$MODEL_FILE" ] || [ "$(wc -c < "$MODEL_FILE" 2>/dev/null || echo 0)" -lt 1000000 ]; then
    echo "Downloading model files (this may take a while, ~2GB)..."
    uv run --refresh python "$SCRIPT_DIR/download_model.py"
else
    echo "Model files already downloaded, skipping download."
fi

# Run the script from its original location using absolute path
uv run --refresh python "$SCRIPT_DIR/generate_cpu.py" \
  --model-dir /tmp/nanochat \
  --prompt "Tell me about dogs."