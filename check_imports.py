"""Cheap import check used at image-build time (no GPU on the build machine)."""
import os
os.environ.setdefault("ORT_DISABLE_TELEMETRY", "1")
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

# The base-only quality pass must fail during image build if an optional
# pitch/separation dependency is absent, before a paid job reaches it.
if os.environ.get("ACESTEP_VARIANT", "turbo").strip().lower() == "base":
    import parselmouth
    import tune
    import instrumental
    from audio_separator.separator.architectures.mdxc_separator import MDXCSeparator
    from audio_separator.separator.uvr_lib_v5.roformer.bs_roformer import BSRoformer

    if not instrumental.available():
        raise SystemExit(f"missing BYOB separator model {instrumental.MODEL}")
    print("BYOB quality imports ok", parselmouth.__version__, instrumental.MODEL)
