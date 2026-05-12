import base64
import glob
import io
import json
import os
import random
import shutil
import threading

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

import runpod
import torch
from diffusers import FluxPipeline
from huggingface_hub import snapshot_download

txt_pipe = None
pipe_lock = threading.Lock()


def model_snapshot_is_complete():
    if not os.path.exists(os.path.join(MODEL_PATH, "model_index.json")):
        return False

    for index_path in glob.glob(os.path.join(MODEL_PATH, "**", "*.index.json"), recursive=True):
        with open(index_path, "r", encoding="utf-8") as index_file:
            weight_map = json.load(index_file).get("weight_map", {})

        index_dir = os.path.dirname(index_path)
        for shard_name in set(weight_map.values()):
            if not os.path.exists(os.path.join(index_dir, shard_name)):
                print(f"Cached model is incomplete; missing {os.path.join(index_dir, shard_name)}.")
                return False

    return True


def ensure_model_available():
    if model_snapshot_is_complete():
        print(f"Using cached Flux model at {MODEL_PATH}.")
        return

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


def get_pipeline():
    global txt_pipe

    if txt_pipe is not None:
        return txt_pipe

    with pipe_lock:
        if txt_pipe is not None:
            return txt_pipe

        ensure_model_available()
        print("Loading Flux model...")
        pipe = FluxPipeline.from_pretrained(
            MODEL_PATH,
            torch_dtype=torch.bfloat16,
            local_files_only=True
        )
        pipe.enable_model_cpu_offload()
        txt_pipe = pipe
        print("Flux model loaded.")
        return txt_pipe


def handler(job):
    job_input = job.get("input", {})

    prompt = job_input.get("prompt", "")
    steps = job_input.get("num_inference_steps", 20)
    guidance = job_input.get("guidance_scale", 3.5)
    width = job_input.get("width", 512)
    height = job_input.get("height", 1024)
    seed = job_input.get("seed", -1)
    reference_image_url = job_input.get("reference_image_url")

    if seed == -1:
        seed = random.randint(0, 2**32 - 1)

    generator = torch.Generator("cpu").manual_seed(seed)

    print(f"Generating — {width}x{height}, steps={steps}, seed={seed}")
    print(f"Prompt: {prompt[:120]}")

    if reference_image_url:
        print("reference_image_url was provided, but this worker currently runs text-to-image only.")

    try:
        pipe = get_pipeline()

        with torch.no_grad():
            image = pipe(
                prompt=prompt,
                num_inference_steps=steps,
                guidance_scale=guidance,
                width=width,
                height=height,
                generator=generator
            ).images[0]
    except Exception as exc:
        print(f"Image generation failed: {exc}")
        return {"error": str(exc)}

    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    img_base64 = base64.b64encode(buffer.getvalue()).decode()

    print("Image generated successfully.")

    return {
        "image": img_base64,
        "seed": seed,
        "width": width,
        "height": height,
    }


runpod.serverless.start({"handler": handler})
