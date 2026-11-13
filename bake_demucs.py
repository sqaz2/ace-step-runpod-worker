"""Download Demucs htdemucs and htdemucs_ft into TORCH_HOME / the HF cache."""
from demucs.pretrained import get_model

for name in ("htdemucs", "htdemucs_ft"):
    m = get_model(name)
    n_models = len(getattr(m, "models", [m]))
    print(
        "demucs",
        name,
        type(m).__name__,
        "bag" if n_models > 1 else "single",
        "n=",
        n_models,
        "sources",
        getattr(m, "sources", None),
        "sr",
        getattr(m, "samplerate", None),
    )
