import base64
import gc
import glob
import io
import json
import os
import random
import requests
import shutil
import threading
from PIL import Image, ImageOps

MODEL_REPO = os.getenv("MODEL_REPO", "black-forest-labs/FLUX.2-klein-4B")
MODEL_IGNORE_PATTERNS = ["*.git*", "*.md"]
MIN_MODEL_DISK_GB = 35
MAX_IMAGE_PIXELS = int(os.getenv("MAX_IMAGE_PIXELS", str(1024 * 1024)))
MAX_SEQUENCE_LENGTH = int(os.getenv("MAX_SEQUENCE_LENGTH", "512"))
DEFAULT_INFERENCE_STEPS = int(os.getenv("DEFAULT_INFERENCE_STEPS", "4"))
MAX_INFERENCE_STEPS = int(os.getenv("MAX_INFERENCE_STEPS", "12"))
MODEL_OFFLOAD_MODE = os.getenv("MODEL_OFFLOAD_MODE", "sequential").lower()
PRELOAD_MODEL = os.getenv("PRELOAD_MODEL", "1").lower() not in {"0", "false", "no"}
MAX_REFERENCE_IMAGE_PIXELS = int(os.getenv("MAX_REFERENCE_IMAGE_PIXELS", str(1024 * 1024)))
DEFAULT_MODEL_ROOT = "/runpod-volume/models" if os.path.isdir("/runpod-volume") else "/app/models"
MODEL_PATH = os.getenv("MODEL_PATH", os.path.join(DEFAULT_MODEL_ROOT, "flux2-klein-4b"))
HF_CACHE_PATH = os.getenv("HF_HOME", os.path.join(os.path.dirname(MODEL_PATH), ".hf-cache"))

os.environ.setdefault("HF_HOME", HF_CACHE_PATH)
os.environ.setdefault("HUGGINGFACE_HUB_CACHE", HF_CACHE_PATH)
os.environ.setdefault("TRANSFORMERS_CACHE", HF_CACHE_PATH)
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

import runpod
import torch
from diffusers import Flux2KleinPipeline
from huggingface_hub import snapshot_download

active_pipe = None
pipe_lock = threading.Lock()
inference_lock = threading.Lock()


def shard_exists(index_dir, shard_name):
    candidates = [
        os.path.normpath(os.path.join(index_dir, shard_name)),
        os.path.normpath(os.path.join(MODEL_PATH, shard_name)),
    ]
    return any(os.path.exists(candidate) for candidate in candidates)


def clear_cuda_cache():
    gc.collect()
    if not torch.cuda.is_available():
        return

    torch.cuda.empty_cache()
    try:
        torch.cuda.ipc_collect()
    except RuntimeError:
        pass


def load_reference_image(image_url):
    response = requests.get(image_url, timeout=30)
    response.raise_for_status()

    image = Image.open(io.BytesIO(response.content)).convert("RGB")
    if image.width * image.height > MAX_REFERENCE_IMAGE_PIXELS:
        scale = (MAX_REFERENCE_IMAGE_PIXELS / (image.width * image.height)) ** 0.5
        size = (max(64, int(image.width * scale)), max(64, int(image.height * scale)))
        image = ImageOps.contain(image, size, method=Image.Resampling.LANCZOS)
    return image


def clamp_int(value, default, minimum, maximum):
    try:
        number = int(value)
    except (TypeError, ValueError):
        number = default

    return max(minimum, min(maximum, number))


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
    global active_pipe

    if active_pipe is not None:
        return active_pipe

    with pipe_lock:
        if active_pipe is not None:
            return active_pipe

        ensure_model_available()
        print("Loading Flux2 Klein model...")
        pipe = Flux2KleinPipeline.from_pretrained(
            MODEL_PATH,
            torch_dtype=torch.float16,
            local_files_only=True
        )
        pipe.enable_attention_slicing()
        if getattr(pipe, "vae", None) is not None:
            pipe.vae.enable_slicing()
            pipe.vae.enable_tiling()

        if MODEL_OFFLOAD_MODE == "model":
            pipe.enable_model_cpu_offload()
        else:
            pipe.enable_sequential_cpu_offload()

        active_pipe = pipe
        print(f"Flux2 Klein model loaded with {MODEL_OFFLOAD_MODE} CPU offload.")
        return active_pipe


def handler(job):
    job_input = job.get("input", {})

    prompt = job_input.get("prompt", "")
    steps = clamp_int(job_input.get("num_inference_steps", DEFAULT_INFERENCE_STEPS), DEFAULT_INFERENCE_STEPS, 1, MAX_INFERENCE_STEPS)
    guidance = job_input.get("guidance_scale", 3.5)
    width = job_input.get("width", 512)
    height = job_input.get("height", 768)
    seed = job_input.get("seed", -1)
    reference_image_url = job_input.get("reference_image_url")

    if seed == -1:
        seed = random.randint(0, 2**32 - 1)

    if width * height > MAX_IMAGE_PIXELS:
        return {
            "error": (
                f"Requested image is too large: {width}x{height}. "
                f"Max pixels is {MAX_IMAGE_PIXELS}; lower width/height or raise MAX_IMAGE_PIXELS."
            )
        }

    generator_device = "cuda" if torch.cuda.is_available() else "cpu"
    generator = torch.Generator(device=generator_device).manual_seed(seed)

    print(f"Generating — {width}x{height}, steps={steps}, seed={seed}")
    print(f"Prompt: {prompt[:120]}")

    try:
        pipe = get_pipeline()

        with inference_lock:
            clear_cuda_cache()
            with torch.inference_mode():
                pipeline_args = {
                    "prompt": prompt,
                    "num_inference_steps": steps,
                    "guidance_scale": guidance,
                    "width": width,
                    "height": height,
                    "generator": generator,
                    "max_sequence_length": MAX_SEQUENCE_LENGTH,
                }
                if reference_image_url:
                    reference_image = load_reference_image(reference_image_url)
                    print("Using anchored reference image for Flux2 image conditioning.")
                    pipeline_args["image"] = reference_image

                image = pipe(**pipeline_args).images[0]
    except Exception as exc:
        print(f"Image generation failed: {exc}")
        return {"error": str(exc)}
    finally:
        clear_cuda_cache()

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


if PRELOAD_MODEL:
    print("Preloading Flux2 Klein model before accepting jobs...")
    get_pipeline()
    print("Worker is warm and ready for jobs.")

runpod.serverless.start({"handler": handler})
