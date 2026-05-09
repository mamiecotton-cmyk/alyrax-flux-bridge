import runpod
import torch
from diffusers import FluxPipeline
import base64
import io
import random

MODEL_PATH = "/app/models/flux"

print("Loading Flux model...")
pipe = FluxPipeline.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.bfloat16,
    local_files_only=True
)
pipe.enable_model_cpu_offload()
print("Flux model loaded.")


def handler(job):
    job_input = job.get("input", {})

    prompt = job_input.get("prompt", "")
    steps = job_input.get("num_inference_steps", 20)
    guidance = job_input.get("guidance_scale", 3.5)
    width = job_input.get("width", 512)
    height = job_input.get("height", 1024)
    seed = job_input.get("seed", -1)

    if seed == -1:
        seed = random.randint(0, 2**32 - 1)

    generator = torch.Generator("cpu").manual_seed(seed)

    print(f"Generating — {width}x{height}, steps={steps}, seed={seed}")
    print(f"Prompt: {prompt[:120]}")

    with torch.no_grad():
        image = pipe(
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
