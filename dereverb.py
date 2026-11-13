"""UVR De-Echo-Aggressive then UVR-DeEcho-DeReverb, on GPU when available.

Same models and defaults as generative.download's Dereverb tool and the
catalogue run that produced 03_normal_vocals_cleaner: python-audio-separator
0.48, VR window 512, aggression 5, 44.1 kHz. Stereo stays stereo.

Separators are built inside the job's temp dir (the catalogue script does
the same). Reusing a loaded instance across output directories left the
second model unable to open the first model's wav.
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

from gate import rms_db

UVR_DIR = Path(os.environ.get("UVR_MODEL_DIR", "/models/uvr"))
CHAIN = (
    ("UVR-De-Echo-Aggressive.pth", "No Echo"),
    ("UVR-DeEcho-DeReverb.pth", "No Reverb"),
)
SAMPLE_RATE = 44100
VR_PARAMS = {
    "batch_size": 1,
    "window_size": 512,
    "aggression": 5,
    "enable_tta": False,
    "enable_post_process": False,
    "post_process_threshold": 0.2,
    "high_end_process": False,
}

_LOADED = False


def log(msg: str) -> None:
    print(msg, flush=True)


def _resample(wav: np.ndarray, src_sr: int, dst_sr: int) -> np.ndarray:
    if src_sr == dst_sr:
        return np.asarray(wav, dtype=np.float32)
    import torchaudio

    t = torch.from_numpy(np.ascontiguousarray(wav.T, dtype=np.float32))
    y = torchaudio.functional.resample(t, src_sr, dst_sr)
    return y.numpy().T.astype(np.float32)


def _match_length(wav: np.ndarray, frames: int) -> np.ndarray:
    if len(wav) == frames:
        return wav
    if len(wav) > frames:
        return wav[:frames]
    pad = np.zeros((frames - len(wav), wav.shape[1]), dtype=np.float32)
    return np.concatenate([wav, pad], axis=0)


def verify_uvr_models() -> None:
    """Fail at preload if a baked model or its metadata is missing."""
    global _LOADED
    if not UVR_DIR.is_dir():
        raise RuntimeError(f"UVR models not found at {UVR_DIR}")
    missing = []
    for name, _stem in CHAIN:
        if not (UVR_DIR / name).is_file():
            missing.append(name)
    for meta in ("vr_model_data.json", "mdx_model_data.json"):
        if not (UVR_DIR / meta).is_file():
            missing.append(meta)
    if missing:
        raise RuntimeError(f"Missing baked UVR files in {UVR_DIR}: {missing}")
    _LOADED = True
    log(f"uvr models present in {UVR_DIR}: {[n for n, _ in CHAIN]}")


def get_uvr() -> None:
    verify_uvr_models()


def _make_separator(output_dir: str, stem: str):
    from audio_separator.separator import Separator

    return Separator(
        output_dir=output_dir,
        model_file_dir=str(UVR_DIR),
        output_format="WAV",
        output_single_stem=stem,
        log_level=logging.WARNING,
        sample_rate=SAMPLE_RATE,
        vr_params=dict(VR_PARAMS),
        use_autocast=False,
    )


def _pick_stem(names: list[str], stem: str, work: str) -> str:
    """Same filename rule as the catalogue script."""
    candidates = []
    for n in names:
        path = n if os.path.isabs(n) else os.path.join(work, n)
        base = os.path.basename(path)
        if f"({stem})" in base and "No Dry" not in base:
            candidates.append(path)
    if not candidates:
        raise RuntimeError(f"UVR did not write a '{stem}' stem; got {names!r}")
    return candidates[0]


def dereverb_vocals(vocals: np.ndarray, sr: int) -> tuple[np.ndarray, int, float]:
    """Run the two-model UVR chain. Returns dry audio at the input sample rate."""
    verify_uvr_models()
    wav = np.asarray(vocals, dtype=np.float32)
    if wav.ndim == 1:
        wav = wav[:, None]
    if wav.ndim == 2 and wav.shape[0] <= 2 and wav.shape[0] < wav.shape[1]:
        wav = wav.T
    orig_sr = int(sr)
    orig_frames = len(wav)
    orig_ch = wav.shape[1]
    work_wav = _resample(wav, orig_sr, SAMPLE_RATE) if orig_sr != SAMPLE_RATE else wav

    t0 = time.time()
    device = "unknown"
    with tempfile.TemporaryDirectory(prefix="uvr-job-") as work:
        src = Path(work) / "in.wav"
        # PCM_24 matches the catalogue stems and avoids FLOAT/librosa issues.
        sf.write(str(src), np.clip(work_wav, -1, 1), SAMPLE_RATE, subtype="PCM_24")
        current = str(src)
        last_paths: list[str] = []
        for name, stem in CHAIN:
            sep = _make_separator(work, stem)
            sep.load_model(model_filename=name)
            device = getattr(sep, "torch_device", "unknown")
            names = sep.separate(current)
            current = _pick_stem(names, stem, work)
            if not os.path.isfile(current):
                raise RuntimeError(f"UVR stem missing after {name}: {current} (names={names!r})")
            last_paths = [n if os.path.isabs(n) else os.path.join(work, n) for n in names]
            log(f"uvr {name} -> {os.path.basename(current)} device={device}")
        dry, rate = sf.read(current, dtype="float32", always_2d=True)
        if rate != SAMPLE_RATE:
            dry = _resample(dry, rate, SAMPLE_RATE)
            rate = SAMPLE_RATE
        if orig_ch == 1:
            dry = dry.mean(axis=1, keepdims=True)
        elif dry.shape[1] == 1:
            dry = np.repeat(dry, orig_ch, axis=1)
        else:
            dry = dry[:, :orig_ch]
        if orig_sr != SAMPLE_RATE:
            dry = _resample(dry, SAMPLE_RATE, orig_sr)
        dry = _match_length(dry, orig_frames)
        # Copies in memory; temp dir cleanup removes the wavs.

    elapsed = time.time() - t0
    log(
        f"uvr dereverb in {elapsed:.1f}s device={device} "
        f"in_rms={rms_db(wav):.1f} out_rms={rms_db(dry):.1f} shape={dry.shape}"
    )
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return dry.astype(np.float32), orig_sr, elapsed
