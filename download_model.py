# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "huggingface_hub",
# ]
# ///
"""Download nanochat model files using huggingface_hub"""
import os
from huggingface_hub import snapshot_download

model_dir = "/tmp/nanochat"
os.makedirs(model_dir, exist_ok=True)

print("Downloading nanochat model files from HuggingFace...")
print("This may take a while (model is ~2GB)...")

snapshot_download(
    repo_id="sdobson/nanochat",
    local_dir=model_dir,
    local_dir_use_symlinks=False,
)

print(f"Model downloaded successfully to {model_dir}")

