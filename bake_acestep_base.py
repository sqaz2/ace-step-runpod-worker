"""Download the pinned ACE-Step 1.5 2B base weights (Lego-capable) into the image."""
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="Runware/acestep-v15-base-diffusers",
    revision="9f8de7a35b079ea67e00ef00a5236f469ee9ed92",
    local_dir="/models/acestep-v15-base",
)
print("ACE-Step base weights ready")
