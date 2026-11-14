"""Cleaner instrumental for vocal swap (the beat Will hears after a swap).

Demucs (beat - vocals) left a ringing, tinny instrumental. A Roformer model
run through python-audio-separator gives a fuller one. The model file is
baked into the image next to the UVR models; if it is missing we fall back
to the Demucs subtraction so a swap never fails because of this.
"""
from __future__ import annotations

import logging
import os
import tempfile
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

from dereverb import UVR_DIR, _match_length, _resample

MODEL = os.environ.get("BEAT_SEPARATOR_MODEL", "model_bs_roformer_ep_317_sdr_12.9755.ckpt")
SAMPLE_RATE = 44100


def log(msg: str) -> None:
    print(msg, flush=True)


def available() -> bool:
    return bool(MODEL) and (UVR_DIR / MODEL).is_file()


def separate(beat: np.ndarray, sr: int) -> tuple[np.ndarray, np.ndarray, float, str]:
    """Return (instrumental, vocals, seconds, model) at `sr`, both [T, 2]."""
    from audio_separator.separator import Separator

    wav = np.asarray(beat, dtype=np.float32)
    frames = len(wav)
    work = _resample(wav, sr, SAMPLE_RATE) if sr != SAMPLE_RATE else wav
    t0 = time.time()
    with tempfile.TemporaryDirectory(prefix="inst-job-") as td:
        src = Path(td) / "beat.wav"
        sf.write(str(src), np.clip(work, -1, 1), SAMPLE_RATE, subtype="FLOAT")
        sep = Separator(output_dir=td, model_file_dir=str(UVR_DIR), output_format="WAV",
                        log_level=logging.WARNING, sample_rate=SAMPLE_RATE, use_autocast=False)
        sep.load_model(model_filename=MODEL)
        names = sep.separate(str(src))
        paths = [n if os.path.isabs(n) else os.path.join(td, n) for n in names]
        inst_p = next((p for p in paths if "(Instrumental)" in os.path.basename(p) or "(Other)" in os.path.basename(p)), None)
        voc_p = next((p for p in paths if "(Vocals)" in os.path.basename(p)), None)
        if inst_p is None and voc_p is None:
            raise RuntimeError(f"separator wrote no stems: {names!r}")
        if inst_p is not None:
            inst, rate = sf.read(inst_p, dtype="float32", always_2d=True)
        if voc_p is not None:
            voc, rate = sf.read(voc_p, dtype="float32", always_2d=True)
        if inst_p is None:
            inst = work[: len(voc)] - voc[: len(work)]
        if voc_p is None:
            voc = work[: len(inst)] - inst[: len(work)]
    out = []
    for a in (inst, voc):
        if a.shape[1] == 1:
            a = np.repeat(a, 2, axis=1)
        if sr != SAMPLE_RATE:
            a = _resample(a, SAMPLE_RATE, sr)
        out.append(_match_length(a[:, :2], frames))
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    secs = time.time() - t0
    log(f"instrumental via {MODEL} in {secs:.1f}s")
    return out[0], out[1], secs, MODEL
