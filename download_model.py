import glob
import json
import os
import shutil

MODEL_REPO = "camenduru/FLUX.1-dev-diffusers"
MODEL_IGNORE_PATTERNS = ["*.git*", "*.md"]
MIN_MODEL_DISK_GB = 35
DEFAULT_MODEL_ROOT = "/runpod-volume/models" if os.path.isdir("/runpod-volume") else "/app/models"
MODEL_PATH = os.getenv("MODEL_PATH", os.path.join(DEFAULT_MODEL_ROOT, "flux"))
HF_CACHE_PATH = os.getenv("HF_HOME", os.path.join(os.path.dirname(MODEL_PATH), ".hf-cache"))

os.environ.setdefault("HF_HOME", HF_CACHE_PATH)
os.environ.setdefault("HUGGINGFACE_HUB_CACHE", HF_CACHE_PATH)
os.environ.setdefault("TRANSFORMERS_CACHE", HF_CACHE_PATH)
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

from huggingface_hub import snapshot_download


def shard_exists(index_dir, shard_name):
    candidates = [
        os.path.normpath(os.path.join(index_dir, shard_name)),
        os.path.normpath(os.path.join(MODEL_PATH, shard_name)),
    ]
    return any(os.path.exists(candidate) for candidate in candidates)


def model_snapshot_is_complete():
    if not os.path.exists(os.path.join(MODEL_PATH, "model_index.json")):
        return False

    for index_path in glob.glob(os.path.join(MODEL_PATH, "**", "*.index.json"), recursive=True):
        with open(index_path, "r", encoding="utf-8") as index_file:
            weight_map = json.load(index_file).get("weight_map", {})

        index_dir = os.path.dirname(index_path)
        for shard_name in set(weight_map.values()):
            if not shard_exists(index_dir, shard_name):
                print(f"Cached model is incomplete; missing shard {shard_name} for {index_path}.")
                return False

    return True


if model_snapshot_is_complete():
    print(f"Using cached Flux model at {MODEL_PATH}.")
    raise SystemExit(0)

if os.path.exists(MODEL_PATH):
    print(f"Removing incomplete Flux model cache at {MODEL_PATH}.")
    shutil.rmtree(MODEL_PATH)

model_parent = os.path.dirname(MODEL_PATH)
os.makedirs(model_parent, exist_ok=True)
free_gb = shutil.disk_usage(model_parent).free / (1024**3)
if free_gb < MIN_MODEL_DISK_GB:
    raise RuntimeError(
        f"Not enough disk space for {MODEL_REPO}. "
        f"Need at least {MIN_MODEL_DISK_GB} GB free at {model_parent}, found {free_gb:.1f} GB. "
        "Attach a RunPod network volume and set MODEL_PATH=/runpod-volume/models/flux."
    )

print(f"Downloading Flux model to {MODEL_PATH}...")

snapshot_download(
    repo_id=MODEL_REPO,
    local_dir=MODEL_PATH,
    ignore_patterns=MODEL_IGNORE_PATTERNS,
)

print("Flux model downloaded successfully.")
