import os

from huggingface_hub import snapshot_download

MODEL_REPO = "camenduru/FLUX.1-dev-diffusers"
MODEL_IGNORE_PATTERNS = ["*.git*", "*.md"]
DEFAULT_MODEL_PATH = "/runpod-volume/models/flux" if os.path.isdir("/runpod-volume") else "/app/models/flux"
MODEL_PATH = os.getenv("MODEL_PATH", DEFAULT_MODEL_PATH)

print(f"Downloading Flux model to {MODEL_PATH}...")

snapshot_download(
    repo_id=MODEL_REPO,
    local_dir=MODEL_PATH,
    ignore_patterns=MODEL_IGNORE_PATTERNS,
)

print("Flux model downloaded successfully.")
