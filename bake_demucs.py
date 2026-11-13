"""Download Demucs htdemucs into TORCH_HOME."""
from demucs.pretrained import get_model

m = get_model("htdemucs")
print(
    "demucs",
    type(m).__name__,
    "sources",
    getattr(m, "sources", None),
    "sr",
    getattr(m, "samplerate", None),
)
