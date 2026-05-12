import runpod
import torch
from diffusers import FluxPipeline
import base64
import io
import random
import requests
from PIL import Image

MODEL_PATH = "/app/models/flux"

print("Loading Flux model...")
txt_pipe = FluxPipeline.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.bfloat16,
    local_files_only=True
)
txt_pipe.enable_model_cpu_offload()
# img2img pipeline removed — onboarding uses text-to-image only
print("Flux model loaded.")


def load_reference_image(url, width, height):
    if not url:
        return None

    response = requests.get(url, timeout=30)
    response.raise_for_status()
    image = Image.open(io.BytesIO(response.content)).convert("RGB")
    return image.resize((width, height), Image.LANCZOS)


def handler(job):
    job_input = job.get("input", {})

    prompt = job_input.get("prompt", "")
    steps = job_input.get("num_inference_steps", 20)
    guidance = job_input.get("guidance_scale", 3.5)
    width = job_input.get("width", 512)
    height = job_input.get("height", 1024)
    seed = job_input.get("seed", -1)
    reference_image_url = job_input.get("reference_image_url")
    reference_strength = float(job_input.get("reference_strength", 0.25))

    if seed == -1:
        seed = random.randint(0, 2**32 - 1)

    generator = torch.Generator("cpu").manual_seed(seed)

    print(f"Generating — {width}x{height}, steps={steps}, seed={seed}")
    print(f"Prompt: {prompt[:120]}")

    reference_image = load_reference_image(reference_image_url, width, height)

    with torch.no_grad():
        if reference_image is not None:
            print(f"Using reference image strength={reference_strength}")
            image = img_pipe(
                prompt=prompt,
                image=reference_image,
                strength=reference_strength,
                num_inference_steps=steps,
                guidance_scale=guidance,
                generator=generator
            ).images[0]
        else:
            image = txt_pipe(
                prompt=prompt,
                num_inference_steps=steps,
                guidance_scale=guidance,
                width=width,
                height=height,
                generator=generator
            ).images[0]

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
