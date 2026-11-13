"""Light vocal gate matching the onomatopoeia / This Is My Story settings.

open -40 dB, close -50 dB, hold 250 ms, pre 30 ms, 20 ms attack/release,
plus a mild spectral denoise of leftover hiss before the gate.
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from scipy.signal import istft, stft

OPEN_DB = -40.0
CLOSE_DB = -50.0
HOLD_MS = 250.0
MIN_OPEN_MS = 100.0
PRE_MS = 30.0
ATTACK_MS = 20.0
RELEASE_MS = 20.0
DENOISE_REDUCE_DB = 8.0


def rms_db(x: np.ndarray) -> float:
    return 20 * math.log10(float(np.sqrt(np.mean(np.square(x))) + 1e-12))


def _mono(x: np.ndarray) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float32)
    if arr.ndim == 2:
        return arr.mean(axis=1).astype(np.float32)
    return arr.astype(np.float32)


def spectral_denoise(x: np.ndarray, sr: int, noise: np.ndarray, reduce_db: float = DENOISE_REDUCE_DB) -> np.ndarray:
    n_fft, hop = 2048, 512
    _, _, X = stft(x, fs=sr, nperseg=n_fft, noverlap=n_fft - hop, window="hann", padded=True, boundary="zeros")
    _, _, N = stft(noise, fs=sr, nperseg=n_fft, noverlap=n_fft - hop, window="hann", padded=True, boundary="zeros")
    noise_mag = np.median(np.abs(N), axis=1, keepdims=True)
    mag = np.abs(X)
    floor = 10 ** (-reduce_db / 20.0)
    clean_mag = np.maximum(mag - 1.0 * noise_mag, mag * floor)
    gain = np.clip(clean_mag / (mag + 1e-12), floor, 1.0)
    gain = gain.copy()
    for i in range(1, gain.shape[1]):
        gain[:, i] = 0.6 * gain[:, i] + 0.4 * gain[:, i - 1]
    _, y = istft(X * gain, fs=sr, nperseg=n_fft, noverlap=n_fft - hop, window="hann")
    return y[: len(x)].astype(np.float32)


def frame_rms_db(x: np.ndarray, sr: int, frame_ms: float = 20.0, hop_ms: float = 10.0):
    frame = max(1, int(sr * frame_ms / 1000.0))
    hop = max(1, int(sr * hop_ms / 1000.0))
    if len(x) < frame:
        return np.array([0.0]), np.array([rms_db(x)]), frame, hop
    frames = sliding_window_view(x, frame)[::hop]
    rms = np.sqrt(np.mean(np.square(frames), axis=1) + 1e-12)
    db = 20 * np.log10(rms)
    times = (np.arange(len(db)) * hop) / sr
    return times, db, frame, hop


def build_mask(
    x: np.ndarray,
    sr: int,
    open_db: float = OPEN_DB,
    close_db: float = CLOSE_DB,
    hold_ms: float = HOLD_MS,
    min_open_ms: float = MIN_OPEN_MS,
    pre_ms: float = PRE_MS,
    attack_ms: float = ATTACK_MS,
    release_ms: float = RELEASE_MS,
) -> tuple[np.ndarray, list]:
    times, db, frame, hop = frame_rms_db(x, sr)
    n_frames = len(db)
    hold_frames = max(1, int(round((hold_ms / 1000.0) * sr / hop)))
    min_open_frames = max(1, int(round((min_open_ms / 1000.0) * sr / hop)))
    pre_frames = max(0, int(round((pre_ms / 1000.0) * sr / hop)))

    open_flags = np.zeros(n_frames, dtype=bool)
    active = False
    hold_left = 0
    run_start = 0
    for i, v in enumerate(db):
        if not active:
            if v >= open_db:
                active = True
                run_start = i
                hold_left = hold_frames
                open_flags[i] = True
        else:
            if v >= close_db:
                hold_left = hold_frames
                open_flags[i] = True
            else:
                if hold_left > 0:
                    hold_left -= 1
                    open_flags[i] = True
                else:
                    if i - run_start < min_open_frames:
                        open_flags[run_start:i] = False
                    active = False
    if active and (n_frames - run_start < min_open_frames):
        open_flags[run_start:] = False

    extended = open_flags.copy()
    for i in range(n_frames):
        if open_flags[i] and (i == 0 or not open_flags[i - 1]):
            a = max(0, i - pre_frames)
            extended[a : i + 1] = True
    open_flags = extended

    regions = []
    in_r = False
    rs = 0
    for i, on in enumerate(open_flags):
        if on and not in_r:
            rs = i * hop
            in_r = True
        elif not on and in_r:
            regions.append((rs, min(len(x), i * hop + frame - hop)))
            in_r = False
    if in_r:
        regions.append((rs, len(x)))

    attack = max(1, int(sr * attack_ms / 1000.0))
    release = max(1, int(sr * release_ms / 1000.0))

    merged = []
    for a, b in regions:
        if merged and a <= merged[-1][1] + release:
            merged[-1] = (merged[-1][0], max(merged[-1][1], b))
        else:
            merged.append((a, b))

    mask = np.zeros(len(x), dtype=np.float32)
    for a, b in merged:
        a = max(0, a)
        b = min(len(x), b)
        if b <= a:
            continue
        end = min(len(x), b + release)
        mask[a:b] = 1.0
        fa = min(attack, b - a)
        mask[a : a + fa] = np.maximum(mask[a : a + fa], np.linspace(0, 1, fa, dtype=np.float32))
        fr = end - b
        if fr > 0:
            mask[b:end] = np.maximum(mask[b:end], np.linspace(1, 0, fr, dtype=np.float32))
        if (b - a) < (attack + 8):
            mid = np.hanning(b - a).astype(np.float32)
            mask[a:b] = mid

    regions_sec = []
    for a, b in merged:
        end = min(len(x), b + release)
        regions_sec.append((round(a / sr, 3), round(end / sr, 3), round((end - a) / sr, 3)))
    return mask, regions_sec


def gate_vocal(audio: np.ndarray, sr: int) -> tuple[np.ndarray, dict[str, Any]]:
    """Apply light denoise + gate. Stereo is kept; the envelope is from the mid."""
    wav = np.asarray(audio, dtype=np.float32)
    if wav.ndim == 1:
        wav = wav[:, None]
    if wav.ndim == 2 and wav.shape[0] <= 2 and wav.shape[0] < wav.shape[1]:
        wav = wav.T
    mid = _mono(wav)
    times, db, frame, hop = frame_rms_db(mid, sr)
    quiet = (db > -90) & (db < -48)
    noise_bits = []
    for i, q in enumerate(quiet):
        if q:
            a, b = i * hop, min(len(mid), i * hop + frame)
            noise_bits.append(mid[a:b])
    noise = np.concatenate(noise_bits) if noise_bits else mid[: int(min(2.0, len(mid) / sr) * sr)]
    if rms_db(noise) > -90:
        denoised_mid = spectral_denoise(mid, sr, noise, reduce_db=DENOISE_REDUCE_DB)
        # apply the mid-derived gain to every channel, falling back to raw if denoise collapses
        if rms_db(denoised_mid) > -90:
            # denoise each channel with the same noise profile (from mid leftovers)
            channels = []
            for c in range(wav.shape[1]):
                channels.append(spectral_denoise(wav[:, c], sr, noise, reduce_db=DENOISE_REDUCE_DB))
            # match length
            n = min(len(ch) for ch in channels)
            denoised = np.stack([ch[:n] for ch in channels], axis=1)
            mid = _mono(denoised)
        else:
            denoised = wav
    else:
        denoised = wav

    mask, regions = build_mask(
        mid if rms_db(mid) > -90 else _mono(wav),
        sr,
        open_db=OPEN_DB,
        close_db=CLOSE_DB,
        hold_ms=HOLD_MS,
        min_open_ms=MIN_OPEN_MS,
        pre_ms=PRE_MS,
        attack_ms=ATTACK_MS,
        release_ms=RELEASE_MS,
    )
    n = min(len(mask), len(denoised))
    gated = denoised[:n] * mask[:n, None]
    report = {
        "open_db": OPEN_DB,
        "close_db": CLOSE_DB,
        "hold_ms": HOLD_MS,
        "pre_ms": PRE_MS,
        "attack_ms": ATTACK_MS,
        "release_ms": RELEASE_MS,
        "input_rms_db": round(rms_db(wav), 2),
        "gated_rms_db": round(rms_db(gated), 2),
        "noise_profile_rms_db": round(rms_db(noise), 2),
        "open_fraction": round(float(np.mean(mask > 0.5)), 4),
        "open_seconds": round(float(np.mean(mask[:n] > 0.5) * n / sr), 2),
        "n_regions": len(regions),
        "regions_sec": regions,
    }
    return gated.astype(np.float32), report
