"""Cheap import check used at image-build time (no GPU on the build machine)."""
import sys
import torch
import diffusers
import gate
from diffusers import AceStepPipeline

print("python", sys.version)
print("torch", torch.__version__, "cuda_built", torch.version.cuda, "has_accelerator", hasattr(torch, "accelerator"))
print("diffusers", diffusers.__version__, "AceStepPipeline ok")
print("gate open_db", gate.OPEN_DB)
