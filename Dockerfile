# Runpod serverless worker for ACE-Step 1.5 turbo + Demucs vocal stem.
# Weights are baked in so cold starts do not download models.
# Public image, no secrets.
#
# Build (GitHub Actions): ghcr.io/sqaz2/ace-step-runpod-worker
#
# Runtime: 24 GB GPU (AMPERE_24: L4 / A5000 / 3090). float32, no CPU offload.

# Ubuntu 24.04 ships Python 3.12. We need >= 3.11.0 final (diffusers 0.42
# and transformers 5.23 both declare Requires-Python >= 3.11.0; Ubuntu
# 22.04's python3.11 package is 3.11.0rc1, which PEP 440 treats as older).
FROM nvidia/cuda:12.6.3-cudnn-runtime-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    ORT_DISABLE_TELEMETRY=1 \
    HF_HOME=/models/hf \
    HUGGINGFACE_HUB_CACHE=/models/hf \
    TRANSFORMERS_CACHE=/models/hf \
    TORCH_HOME=/models/torch \
    HF_HUB_ENABLE_HF_TRANSFER=1 \
    PIPELINE_PATH=/models/acestep-v15-turbo \
    DEMUCS_MODEL=htdemucs_ft \
    UVR_MODEL_DIR=/models/uvr \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    VIRTUAL_ENV=/opt/venv \
    PATH=/opt/venv/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin

# python3-venv avoids Ubuntu 24.04's PEP 668 "externally managed" pip block.
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 python3-venv python3-dev \
        ffmpeg libsndfile1 git ca-certificates wget curl \
    && python3 -m venv /opt/venv \
    && ln -sf /opt/venv/bin/python /usr/local/bin/python \
    && ln -sf /opt/venv/bin/python /usr/local/bin/python3 \
    && rm -rf /var/lib/apt/lists/*

# PyTorch CUDA wheels first, then the rest. demucs is installed without
# its own torch pin so we keep the CUDA build.
# diffusers 0.42 uses torch.accelerator (PyTorch >= 2.7). 2.11+cu126
# matches the proven Kaggle ACE-Step run (2.11) on this 12.6 runtime.
RUN python -m pip install --no-cache-dir --upgrade pip \
    && python -m pip install --no-cache-dir \
        torch==2.11.0 torchaudio==2.11.0 \
        --index-url https://download.pytorch.org/whl/cu126

COPY requirements.txt /tmp/requirements.txt
RUN python -m pip install --no-cache-dir -r /tmp/requirements.txt \
    && python -m pip install --no-cache-dir --no-deps demucs==4.0.1 \
    && rm -rf /root/.cache/pip

# Bake ACE-Step 1.5 2B turbo (~6.3 GB) at a pinned revision.
COPY bake_acestep.py /tmp/bake_acestep.py
RUN python /tmp/bake_acestep.py

# Bake Demucs htdemucs and htdemucs_ft into TORCH_HOME.
COPY bake_demucs.py /tmp/bake_demucs.py
RUN python /tmp/bake_demucs.py

# UVR De-Echo-Aggressive + DeEcho-DeReverb (same models as the site Dereverb tool).
# Installed after torch so pip cannot replace the CUDA wheel. The two VR-arch
# models and their hash metadata are baked in so a cold start stays offline.
RUN printf 'torch==2.11.0\ntorchaudio==2.11.0\n' > /tmp/torch-constraint.txt
COPY requirements-uvr.txt /tmp/requirements-uvr.txt
RUN python -m pip install --no-cache-dir --no-deps audio-separator==0.48.0     && python -m pip install --no-cache-dir -r /tmp/requirements-uvr.txt         --constraint /tmp/torch-constraint.txt     && rm -rf /root/.cache/pip
COPY models/uvr /models/uvr

# After bake, prefer local files. Handler still falls back if a file is missing.
ENV HF_HUB_OFFLINE=1

WORKDIR /app
COPY handler.py gate.py dereverb.py layers.py tune.py instrumental.py check_imports.py /app/

# Cheap import check (no GPU in the build machine).
RUN python /app/check_imports.py

CMD ["python", "-u", "/app/handler.py"]
