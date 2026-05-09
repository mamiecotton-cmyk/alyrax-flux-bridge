from huggingface_hub import snapshot_download

print("Downloading Flux model...")

snapshot_download(
    repo_id="camenduru/FLUX.1-dev-diffusers",
    local_dir="/app/models/flux",
    ignore_patterns=["*.git*", "*.md"]
)

print("Flux model downloaded successfully.")
