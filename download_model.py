import os

MODEL_REPO = os.getenv("MODEL_REPO", "Stableyogi/Realism-Pony-Checkpoints")
MODEL_FILENAME = os.getenv("MODEL_FILENAME", "realismByStableYogi_ponyV3VAE.safetensors")
MODEL_DISPLAY_NAME = os.getenv("MODEL_DISPLAY_NAME", "Realism Pony")
MIN_MODEL_DISK_GB = int(os.getenv("MIN_MODEL_DISK_GB", "15"))
DEFAULT_MODEL_ROOT = "/runpod-volume/models" if os.path.isdir("/runpod-volume") else "/app/models"
MODEL_PATH = os.getenv("MODEL_PATH", os.path.join(DEFAULT_MODEL_ROOT, "realism-pony"))
HF_CACHE_PATH = os.getenv("HF_HOME", os.path.join(os.path.dirname(MODEL_PATH), ".hf-cache"))
MODEL_FILE_PATH = os.path.join(MODEL_PATH, MODEL_FILENAME)

os.environ.setdefault("HF_HOME", HF_CACHE_PATH)
os.environ.setdefault("HUGGINGFACE_HUB_CACHE", HF_CACHE_PATH)
os.environ.setdefault("TRANSFORMERS_CACHE", HF_CACHE_PATH)
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

from huggingface_hub import hf_hub_download


def model_file_is_complete():
    return os.path.exists(MODEL_FILE_PATH) and os.path.getsize(MODEL_FILE_PATH) > 1024 * 1024 * 1024


if model_file_is_complete():
    print(f"Using cached {MODEL_DISPLAY_NAME} model at {MODEL_FILE_PATH}.")
    raise SystemExit(0)

os.makedirs(MODEL_PATH, exist_ok=True)
stats = os.statvfs(MODEL_PATH)
free_gb = stats.f_bavail * stats.f_frsize / (1024**3)
if free_gb < MIN_MODEL_DISK_GB:
    raise RuntimeError(
        f"Not enough disk space for {MODEL_REPO}. "
        f"Need at least {MIN_MODEL_DISK_GB} GB free at {MODEL_PATH}, found {free_gb:.1f} GB. "
        "Attach a RunPod network volume and set MODEL_PATH=/runpod-volume/models/realism-pony."
    )

print(f"Downloading {MODEL_DISPLAY_NAME} model {MODEL_REPO}/{MODEL_FILENAME} to {MODEL_PATH}...")

hf_hub_download(
    repo_id=MODEL_REPO,
    filename=MODEL_FILENAME,
    local_dir=MODEL_PATH,
    local_dir_use_symlinks=False,
)

print(f"{MODEL_DISPLAY_NAME} model downloaded successfully.")
