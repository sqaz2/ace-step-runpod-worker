"""Extra vocal layers for bring-your-own-beat: doubles, harmonies, ad-libs
helpers, key estimate, level match and edge fades. Pure DSP, no new models."""
from __future__ import annotations

import numpy as np
import torch

NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
# Krumhansl-Kessler key profiles.
MAJOR = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
MINOR = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])
MAJOR_SCALE = (0, 2, 4, 5, 7, 9, 11)
MINOR_SCALE = (0, 2, 3, 5, 7, 8, 10)


def mono(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    return x if x.ndim == 1 else x.mean(axis=1)


def estimate_key(audio: np.ndarray, sr: int) -> dict:
    """Key of the beat from its chroma. A suggestion: about 60-70% right."""
    import librosa

    y = librosa.resample(mono(audio), orig_sr=sr, target_sr=22050)
    chroma = librosa.feature.chroma_cqt(y=y, sr=22050).mean(axis=1)
    best = (-2.0, 0, "major")
    for tonic in range(12):
        for mode, prof in (("major", MAJOR), ("minor", MINOR)):
            r = float(np.corrcoef(chroma, np.roll(prof, tonic))[0, 1])
            if r > best[0]:
                best = (r, tonic, mode)
    r, tonic, mode = best
    return {"tonic": tonic, "mode": mode, "name": f"{NOTE_NAMES[tonic]} {mode}", "confidence": round(r, 3)}


def edge_fades(x: np.ndarray, sr: int, ms: float = 10.0) -> np.ndarray:
    x = np.array(x, dtype=np.float32, copy=True)
    n = min(len(x) // 4, int(sr * ms / 1000))
    if n > 1:
        ramp = np.linspace(0.0, 1.0, n, dtype=np.float32)[:, None] if x.ndim == 2 else np.linspace(0.0, 1.0, n, dtype=np.float32)
        x[:n] *= ramp
        x[-n:] *= ramp[::-1]
    return x


def active_rms_db(x: np.ndarray, sr: int, floor_db: float = -45.0) -> float:
    """Loudness of the sung parts only (50 ms frames above a floor)."""
    m = mono(x)
    hop = max(1, int(sr * 0.05))
    n = len(m) // hop
    if n < 1:
        return -120.0
    f = m[: n * hop].reshape(n, hop)
    db = 20 * np.log10(np.sqrt(np.mean(f ** 2, axis=1)) + 1e-9)
    on = db > floor_db
    if not on.any():
        return -120.0
    return float(20 * np.log10(np.sqrt(np.mean(f[on] ** 2)) + 1e-9))


def level_match(x: np.ndarray, sr: int, target_db: float, max_gain_db: float = 12.0) -> tuple[np.ndarray, float]:
    cur = active_rms_db(x, sr)
    if cur < -100:
        return x, 0.0
    gain = float(np.clip(target_db - cur, -max_gain_db, max_gain_db))
    out = x * (10 ** (gain / 20))
    peak = float(np.max(np.abs(out))) if out.size else 0.0
    if peak > 0.98:
        out = out / peak * 0.98
        gain += 20 * np.log10(0.98 / peak)
    return out.astype(np.float32), round(gain, 2)


def _shift(x: np.ndarray, sr: int, semitones: float) -> np.ndarray:
    import torchaudio

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    t = torch.from_numpy(np.ascontiguousarray(np.asarray(x, dtype=np.float32).T)).to(dev)
    with torch.no_grad():
        y = torchaudio.functional.pitch_shift(t, sr, n_steps=semitones, bins_per_octave=12, n_fft=2048)
    out = y.detach().cpu().numpy().T.astype(np.float32)
    if len(out) < len(x):
        out = np.concatenate([out, np.zeros((len(x) - len(out), out.shape[1]), dtype=np.float32)])
    return out[: len(x)]


def make_double(vocal: np.ndarray, sr: int, seed: int = 0) -> np.ndarray:
    """ADT-style double: about +9 cents, a slowly wandering 18-32 ms delay,
    slightly darker and 2 dB down. Not a new performance, but it thickens."""
    rng = np.random.default_rng(seed)
    v = np.asarray(vocal, dtype=np.float32)
    if v.ndim == 1:
        v = np.stack([v, v], axis=1)
    detuned = _shift(v, sr, 0.09)
    n = len(detuned)
    t = np.arange(n, dtype=np.float32) / sr
    rate = 0.25 + 0.2 * rng.random()
    delay = (0.025 + 0.007 * np.sin(2 * np.pi * rate * t + rng.random() * 6.28)) * sr
    idx = np.clip(np.arange(n, dtype=np.float32) - delay, 0, n - 1)
    i0 = np.floor(idx).astype(np.int64)
    i1 = np.minimum(i0 + 1, n - 1)
    frac = (idx - i0)[:, None]
    out = detuned[i0] * (1 - frac) + detuned[i1] * frac
    # gentle one-pole low-pass so it sits behind the lead
    a = np.exp(-2 * np.pi * 9000 / sr)
    from scipy.signal import lfilter
    out = lfilter([1 - a], [1, -a], out, axis=0).astype(np.float32)
    return out * (10 ** (-2 / 20))


def make_harmony(vocal: np.ndarray, sr: int, key: dict, interval: str = "third") -> tuple[np.ndarray, dict]:
    """Key-aware parallel harmony. Tracks the sung note every 50 ms and picks
    the diatonic third (3 or 4 semitones) or fifth (6/7) above that stays in
    the key, then switches between two pitch-shifted copies with short fades."""
    import librosa

    v = np.asarray(vocal, dtype=np.float32)
    if v.ndim == 1:
        v = np.stack([v, v], axis=1)
    scale = MAJOR_SCALE if key.get("mode") == "major" else MINOR_SCALE
    tonic = int(key.get("tonic", 0))
    in_scale = {(tonic + s) % 12 for s in scale}
    options = (3, 4) if interval == "third" else (7, 6)

    y16 = librosa.resample(mono(v), orig_sr=sr, target_sr=16000)
    hop = 800  # 50 ms
    f0 = librosa.yin(y16, fmin=70, fmax=900, sr=16000, frame_length=2048, hop_length=hop)
    rms = librosa.feature.rms(y=y16, frame_length=2048, hop_length=hop)[0]
    f0 = f0[: len(rms)]
    voiced = rms > (np.max(rms) * 0.05 + 1e-6)
    choice = np.zeros(len(f0), dtype=np.int64)  # 0 -> options[0], 1 -> options[1]
    for i, (hz, ok) in enumerate(zip(f0, voiced)):
        if not ok or not np.isfinite(hz) or hz <= 0:
            choice[i] = choice[i - 1] if i else 0
            continue
        pc = int(round(librosa.hz_to_midi(hz))) % 12
        a, b = ((pc + options[0]) % 12) in in_scale, ((pc + options[1]) % 12) in in_scale
        choice[i] = 0 if a and not b else 1 if b and not a else (choice[i - 1] if i else 0)
    # Smooth single-frame flips.
    for i in range(1, len(choice) - 1):
        if choice[i - 1] == choice[i + 1] != choice[i]:
            choice[i] = choice[i - 1]

    lo, hi = _shift(v, sr, options[0]), _shift(v, sr, options[1])
    per = sr * hop / 16000.0
    sel = np.interp(np.arange(len(v)) / per, np.arange(len(choice)), choice.astype(np.float32))
    # ~20 ms crossfades between the two shifts
    k = max(1, int(sr * 0.02))
    sel = np.convolve(sel, np.ones(k, dtype=np.float32) / k, mode="same")[:, None]
    out = lo * (1 - sel) + hi * sel
    report = {"interval": interval, "key": key.get("name"), "options": list(options),
              "share_upper": round(float(np.mean(choice)), 3)}
    return (out * (10 ** (-4 / 20))).astype(np.float32), report


def gaps_mask(lead: np.ndarray, sr: int, open_below_db: float = -38.0, hold_ms: float = 120.0, fade_ms: float = 30.0) -> np.ndarray:
    """1 where the lead is quiet (room for ad-libs), 0 under the lead."""
    m = mono(lead)
    hop = max(1, int(sr * 0.01))
    n = int(np.ceil(len(m) / hop))
    pad = np.zeros(n * hop - len(m), dtype=np.float32)
    f = np.concatenate([m, pad]).reshape(n, hop)
    db = 20 * np.log10(np.sqrt(np.mean(f ** 2, axis=1)) + 1e-9)
    busy = db > open_below_db
    hold = max(1, int(hold_ms / 10))
    busy = np.convolve(busy.astype(np.float32), np.ones(2 * hold + 1), mode="same") > 0
    mask = (~busy).astype(np.float32)
    k = max(1, int(fade_ms / 10))
    mask = np.convolve(mask, np.ones(k) / k, mode="same")
    return np.repeat(mask, hop)[: len(m)].astype(np.float32)


ADLIB_LINES = ["(yeah)", "(uh)", "(let's go)", "(come on)", "(woo)", "(that's right)", "(okay)", "(hey)"]


def adlib_lyrics(lead_lyrics: str, seconds: float) -> str:
    """Short, sparse call-outs with one per lead line, so Lego sings few words."""
    lines = [l for l in lead_lyrics.splitlines() if l.strip() and not l.strip().startswith("[")]
    count = max(4, min(len(lines) or 4, int(seconds / 3)))
    picks = [ADLIB_LINES[i % len(ADLIB_LINES)] for i in range(count)]
    return "[verse]\n" + "\n".join(picks)
