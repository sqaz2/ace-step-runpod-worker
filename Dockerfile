# Runpod serverless worker for ACE-Step 1.5 turbo + Demucs vocal stem.
# Weights are baked in so cold starts do not download models.
# Public image, no secrets.
#
# Build (GitHub Actions): ghcr.io/sqaz2/ace-step-runpod-worker
#
# Runtime: 24 GB GPU (AMPERE_24: L4 / A5000 / 3090). float32, no CPU offload.

FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/models/hf \
    HUGGINGFACE_HUB_CACHE=/models/hf \
    TRANSFORMERS_CACHE=/models/hf \
    TORCH_HOME=/models/torch \
    HF_HUB_ENABLE_HF_TRANSFER=1 \
    PIPELINE_PATH=/models/acestep-v15-turbo \
    DEMUCS_MODEL=htdemucs \
    PIP_DISABLE_PIP_VERSION_CHECK=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 python3-pip python3-venv \
        ffmpeg libsndfile1 git ca-certificates wget \
    && ln -sf /usr/bin/python3 /usr/bin/python \
    && rm -rf /var/lib/apt/lists/*

# PyTorch CUDA wheels first, then the rest. demucs is installed without
# its own torch pin so we keep the CUDA build.
RUN python -m pip install --no-cache-dir --upgrade pip \
    && python -m pip install --no-cache-dir \
        torch==2.5.1 torchaudio==2.5.1 \
        --index-url https://download.pytorch.org/whl/cu124

COPY requirements.txt /tmp/requirements.txt
RUN python -m pip install --no-cache-dir -r /tmp/requirements.txt \
    && python -m pip install --no-cache-dir --no-deps demucs==4.0.1 \
    && rm -rf /root/.cache/pip

# Bake ACE-Step 1.5 2B turbo (~6.3 GB) at a pinned revision.
RUN python -c "\
from huggingface_hub import snapshot_download; \
snapshot_download( \
    'Runware/acestep-v15-turbo-diffusers', \
    revision='be23effe449c5957947f3020fd63bee23c64abe4', \
    local_dir='/models/acestep-v15-turbo', \
    local_dir_use_symlinks=False); \
print('ACE-Step weights ready')"

# Bake Demucs htdemucs into TORCH_HOME.
RUN python -c "\
from demucs.pretrained import get_model; \
m = get_model('htdemucs'); \
print('demucs', type(m).__name__, 'sources', getattr(m, 'sources', None), 'sr', getattr(m, 'samplerate', None))"

# After bake, prefer local files. Handler still falls back if a file is missing.
ENV HF_HUB_OFFLINE=1

WORKDIR /app
COPY handler.py gate.py /app/

# Cheap import check (no GPU in the build machine).
RUN python -c "\
import gate, inspect, torch, diffusers; \
from diffusers import AceStepPipeline; \
print('torch', torch.__version__, 'cuda_built', torch.version.cuda); \
print('diffusers', diffusers.__version__, 'AceStepPipeline ok'); \
print('gate open_db', gate.OPEN_DB)"

CMD ["python", "-u", "/app/handler.py"]
