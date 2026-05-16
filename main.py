import base64
import gc
import io
import os
import random
import requests
import threading
from PIL import Image, ImageOps

MODEL_REPO = os.getenv("MODEL_REPO", "Stableyogi/Realism-Pony-Checkpoints")
MODEL_FILENAME = os.getenv("MODEL_FILENAME", "realismByStableYogi_ponyV3VAE.safetensors")
MODEL_DISPLAY_NAME = os.getenv("MODEL_DISPLAY_NAME", "Realism Pony")
MIN_MODEL_DISK_GB = int(os.getenv("MIN_MODEL_DISK_GB", "15"))
MAX_IMAGE_PIXELS = int(os.getenv("MAX_IMAGE_PIXELS", str(1024 * 1024)))
DEFAULT_INFERENCE_STEPS = int(os.getenv("DEFAULT_INFERENCE_STEPS", "28"))
MAX_INFERENCE_STEPS = int(os.getenv("MAX_INFERENCE_STEPS", "40"))
MODEL_OFFLOAD_MODE = os.getenv("MODEL_OFFLOAD_MODE", "none").lower()
PRELOAD_MODEL = os.getenv("PRELOAD_MODEL", "1").lower() not in {"0", "false", "no"}
MAX_REFERENCE_IMAGE_PIXELS = int(os.getenv("MAX_REFERENCE_IMAGE_PIXELS", str(1024 * 1024)))
REFERENCE_DENOISE_BASE = float(os.getenv("REFERENCE_DENOISE_BASE", "0.75"))
DEFAULT_REFERENCE_DENOISE = float(os.getenv("DEFAULT_REFERENCE_DENOISE", "0.55"))
PONY_PROMPT_PREFIX = os.getenv("PONY_PROMPT_PREFIX", "score_9, score_8_up, photo, realistic")
PONY_NEGATIVE_PREFIX = os.getenv(
    "PONY_NEGATIVE_PREFIX",
    "score_6, score_5, score_4, worst quality, low quality, blurry, bad anatomy, extra limbs, missing limbs, pixar, disney, cartoon, anime, 3d character, doll face, plastic skin, game render"
)
DEFAULT_MODEL_ROOT = "/runpod-volume/models" if os.path.isdir("/runpod-volume") else "/app/models"
MODEL_PATH = os.getenv("MODEL_PATH", os.path.join(DEFAULT_MODEL_ROOT, "realism-pony"))
HF_CACHE_PATH = os.getenv("HF_HOME", os.path.join(os.path.dirname(MODEL_PATH), ".hf-cache"))
MODEL_FILE_PATH = os.path.join(MODEL_PATH, MODEL_FILENAME)

os.environ.setdefault("HF_HOME", HF_CACHE_PATH)
os.environ.setdefault("HUGGINGFACE_HUB_CACHE", HF_CACHE_PATH)
os.environ.setdefault("TRANSFORMERS_CACHE", HF_CACHE_PATH)
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

import runpod
import torch
from diffusers import (
    EulerAncestralDiscreteScheduler,
    StableDiffusionXLImg2ImgPipeline,
    StableDiffusionXLPipeline,
)
from huggingface_hub import hf_hub_download

active_pipes = None
pipe_lock = threading.Lock()
inference_lock = threading.Lock()


def clear_cuda_cache():
    gc.collect()
    if not torch.cuda.is_available():
        return

    torch.cuda.empty_cache()
    try:
        torch.cuda.ipc_collect()
    except RuntimeError:
        pass


def load_reference_image(image_url, width=None, height=None):
    if image_url.startswith("data:image"):
        try:
            _, encoded_image = image_url.split(",", 1)
        except ValueError as exc:
            raise ValueError("Invalid data URL reference image") from exc
        image_bytes = base64.b64decode(encoded_image)
    else:
        response = requests.get(image_url, timeout=30)
        response.raise_for_status()
        image_bytes = response.content

    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    if image.width * image.height > MAX_REFERENCE_IMAGE_PIXELS:
        scale = (MAX_REFERENCE_IMAGE_PIXELS / (image.width * image.height)) ** 0.5
        size = (max(64, int(image.width * scale)), max(64, int(image.height * scale)))
        image = ImageOps.contain(image, size, method=Image.Resampling.LANCZOS)

    if width and height:
        image = ImageOps.pad(
            image,
            (width, height),
            method=Image.Resampling.LANCZOS,
            color=(255, 255, 255),
            centering=(0.5, 0.5),
        )

    return image


def clamp_int(value, default, minimum, maximum):
    try:
        number = int(value)
    except (TypeError, ValueError):
        number = default

    return max(minimum, min(maximum, number))


def clamp_float(value, default, minimum, maximum):
    try:
        number = float(value)
    except (TypeError, ValueError):
        number = default

    return max(minimum, min(maximum, number))


def model_file_is_complete():
    return os.path.exists(MODEL_FILE_PATH) and os.path.getsize(MODEL_FILE_PATH) > 1024 * 1024 * 1024


def ensure_model_available():
    if model_file_is_complete():
        print(f"Using cached {MODEL_DISPLAY_NAME} model at {MODEL_FILE_PATH}.")
        return

    os.makedirs(MODEL_PATH, exist_ok=True)
    free_gb = os.statvfs(MODEL_PATH).f_bavail * os.statvfs(MODEL_PATH).f_frsize / (1024**3)
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


def apply_memory_settings(pipe):
    pipe.enable_attention_slicing()
    if getattr(pipe, "vae", None) is not None:
        pipe.vae.enable_slicing()
        pipe.vae.enable_tiling()


def place_pipeline(pipe):
    if not torch.cuda.is_available():
        return

    if MODEL_OFFLOAD_MODE == "model":
        pipe.enable_model_cpu_offload()
    elif MODEL_OFFLOAD_MODE == "sequential":
        pipe.enable_sequential_cpu_offload()
    else:
        pipe.to("cuda")


def normalize_prompt(prompt):
    prompt = prompt.strip()
    if not PONY_PROMPT_PREFIX:
        return prompt
    if prompt.lower().startswith(PONY_PROMPT_PREFIX.lower()):
        return prompt
    return f"{PONY_PROMPT_PREFIX}, {prompt}" if prompt else PONY_PROMPT_PREFIX


def normalize_negative_prompt(negative_prompt):
    negative_prompt = negative_prompt.strip()
    if not PONY_NEGATIVE_PREFIX:
        return negative_prompt
    if negative_prompt.lower().startswith(PONY_NEGATIVE_PREFIX.lower()):
        return negative_prompt
    return f"{PONY_NEGATIVE_PREFIX}, {negative_prompt}" if negative_prompt else PONY_NEGATIVE_PREFIX


def get_pipelines():
    global active_pipes

    if active_pipes is not None:
        return active_pipes

    with pipe_lock:
        if active_pipes is not None:
            return active_pipes

        ensure_model_available()
        print(f"Loading {MODEL_DISPLAY_NAME} SDXL model...")
        text_pipe = StableDiffusionXLPipeline.from_single_file(
            MODEL_FILE_PATH,
            torch_dtype=torch.float16,
            use_safetensors=True,
        )
        text_pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(text_pipe.scheduler.config)
        image_pipe = StableDiffusionXLImg2ImgPipeline(**text_pipe.components)
        image_pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(image_pipe.scheduler.config)

        apply_memory_settings(text_pipe)
        apply_memory_settings(image_pipe)
        place_pipeline(text_pipe)
        place_pipeline(image_pipe)

        active_pipes = {
            "text": text_pipe,
            "image": image_pipe,
        }
        print(f"{MODEL_DISPLAY_NAME} model loaded with {MODEL_OFFLOAD_MODE} offload mode.")
        return active_pipes


def handler(job):
    job_input = job.get("input", {})

    prompt = normalize_prompt(job_input.get("prompt", ""))
    negative_prompt = normalize_negative_prompt(job_input.get("negative_prompt", ""))
    steps = clamp_int(job_input.get("num_inference_steps", DEFAULT_INFERENCE_STEPS), DEFAULT_INFERENCE_STEPS, 1, MAX_INFERENCE_STEPS)
    guidance = clamp_float(job_input.get("guidance_scale", 7.0), 7.0, 1.0, 12.0)
    width = job_input.get("width", 512)
    height = job_input.get("height", 768)
    seed = job_input.get("seed", -1)
    reference_image_url = job_input.get("reference_image_url")
    reference_strength = clamp_float(job_input.get("reference_strength", 0.23), 0.23, 0.0, 1.0)
    denoise_strength = clamp_float(
        job_input.get("denoise_strength", REFERENCE_DENOISE_BASE - reference_strength),
        DEFAULT_REFERENCE_DENOISE,
        0.30,
        0.80,
    )

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

    print(f"Generating — {width}x{height}, steps={steps}, cfg={guidance}, seed={seed}")
    print(f"Prompt length={len(prompt)}: {prompt[:180]}")

    try:
        pipes = get_pipelines()

        with inference_lock:
            clear_cuda_cache()
            with torch.inference_mode():
                pipeline_args = {
                    "prompt": prompt,
                    "negative_prompt": negative_prompt,
                    "num_inference_steps": steps,
                    "guidance_scale": guidance,
                    "width": width,
                    "height": height,
                    "generator": generator,
                }
                if reference_image_url:
                    reference_image = load_reference_image(reference_image_url, width, height)
                    print(
                        f"Using anchored reference image for {MODEL_DISPLAY_NAME} img2img: "
                        f"url={reference_image_url[:180]}, size={reference_image.width}x{reference_image.height}, "
                        f"denoise_strength={denoise_strength}."
                    )
                    pipeline_args["image"] = reference_image
                    pipeline_args["strength"] = denoise_strength
                    image = pipes["image"](**pipeline_args).images[0]
                else:
                    image = pipes["text"](**pipeline_args).images[0]
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
    print(f"Preloading {MODEL_DISPLAY_NAME} model before accepting jobs...")
    get_pipelines()
    print("Worker is warm and ready for jobs.")

runpod.serverless.start({"handler": handler})
