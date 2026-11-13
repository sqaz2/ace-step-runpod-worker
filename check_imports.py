"""Cheap import check used at image-build time (no GPU on the build machine)."""
import os
import sys
from pathlib import Path

import torch
import diffusers
import gate
import dereverb
from diffusers import AceStepPipeline
from audio_separator.separator import Separator

print("python", sys.version)
print("torch", torch.__version__, "cuda_built", torch.version.cuda, "has_accelerator", hasattr(torch, "accelerator"))
print("diffusers", diffusers.__version__, "AceStepPipeline ok")
print("gate open_db", gate.OPEN_DB)
print("dereverb chain", [n for n, _ in dereverb.CHAIN], "dir", dereverb.UVR_DIR)

uvr = Path(os.environ.get("UVR_MODEL_DIR", "/models/uvr"))
for name, _stem in dereverb.CHAIN:
    path = uvr / name
    if not path.is_file():
        raise SystemExit(f"missing UVR model {path}")
    print("uvr model", name, path.stat().st_size)
for meta in ("vr_model_data.json", "mdx_model_data.json"):
    path = uvr / meta
    if not path.is_file():
        raise SystemExit(f"missing UVR metadata {path}")
    print("uvr meta", meta, path.stat().st_size)
print("audio_separator Separator ok")
