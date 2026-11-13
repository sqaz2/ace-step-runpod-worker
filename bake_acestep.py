"""Download the pinned ACE-Step 1.5 2B turbo weights into the image."""
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="Runware/acestep-v15-turbo-diffusers",
    revision="be23effe449c5957947f3020fd63bee23c64abe4",
    local_dir="/models/acestep-v15-turbo",
)
print("ACE-Step weights ready")
