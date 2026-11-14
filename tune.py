"""Gentle "tune to key" for generated vocals (PSOLA via Praat/parselmouth).

Natural-sounding by design:
  * works per note: each sung note's centre (median pitch) is pulled toward
    the nearest note of the beat's key by `strength` (0..1), so vibrato,
    slides and the singer's own movement inside the note are kept;
  * notes more than `max_snap_cents` from any key note are only pulled to the
    nearest semitone (keeps deliberate blue/chromatic notes);
  * consonants, rap/spoken parts and anything not corrected are left as the
    original samples (crossfaded in/out), so PSOLA only touches pitched notes.
"""
from __future__ import annotations

import re

import numpy as np

NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
FLAT = {"Db": "C#", "Eb": "D#", "Gb": "F#", "Ab": "G#", "Bb": "A#", "Cb": "B", "Fb": "E", "E#": "F", "B#": "C"}
MAJOR_SCALE = (0, 2, 4, 5, 7, 9, 11)
MINOR_SCALE = (0, 2, 3, 5, 7, 8, 10)


def parse_keyscale(s: str | None) -> dict | None:
    """'F major', 'F# minor', 'Bbm', 'A min' -> {'tonic', 'mode', 'name'}; None if unreadable."""
    if not isinstance(s, str):
        return None
    m = re.match(r"^\s*([A-Ga-g])([#b]?)\s*(major|minor|maj|min|m)?\s*$", s.strip())
    if not m:
        return None
    note = m.group(1).upper() + (m.group(2) or "")
    note = FLAT.get(note, note)
    if note not in NOTE_NAMES:
        return None
    mode = "minor" if (m.group(3) or "major").lower() in ("minor", "min", "m") else "major"
    tonic = NOTE_NAMES.index(note)
    return {"tonic": tonic, "mode": mode, "name": f"{note} {mode}", "confidence": 1.0, "source": "user"}


def _mono(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    return x if x.ndim == 1 else x.mean(axis=1)


def note_segments(midi: np.ndarray, voiced: np.ndarray, min_frames: int = 5, jump: float = 0.7):
    """Runs of voiced frames without big jumps -> list of (start, end) frame indices."""
    segs, i, n = [], 0, len(midi)
    while i < n:
        if not voiced[i]:
            i += 1
            continue
        j = i + 1
        while j < n and voiced[j] and abs(midi[j] - midi[j - 1]) < jump:
            j += 1
        if j - i >= min_frames:
            segs.append((i, j))
        i = j
    return segs


def target_for(center: float, key: dict | None, max_snap_cents: float) -> float:
    semi = float(np.round(center))
    if not key:
        return semi
    scale = MAJOR_SCALE if key.get("mode") == "major" else MINOR_SCALE
    tonic = int(key.get("tonic", 0))
    best, best_d = semi, 1e9
    for octave in (-1, 0, 1):
        base = 12 * (np.floor(center / 12) + octave) + tonic
        for s in scale:
            cand = base + s
            d = abs(cand - center)
            if d < best_d:
                best, best_d = cand, d
    return float(best) if best_d * 100 <= max_snap_cents else semi


def pitch_score(audio: np.ndarray, sr: int, key: dict | None) -> dict:
    """How in tune a take is: median note distance (cents) to the key, and the share of clearly wrong notes."""
    import parselmouth

    snd = parselmouth.Sound(_mono(audio), sampling_frequency=sr)
    p = snd.to_pitch_ac(time_step=0.01, pitch_floor=70, pitch_ceiling=900)
    f0 = p.selected_array["frequency"]
    voiced = f0 > 0
    midi = np.where(voiced, 69 + 12 * np.log2(np.maximum(f0, 1e-6) / 440.0), 0.0)
    devs = []
    for a, b in note_segments(midi, voiced, min_frames=8):
        c = float(np.median(midi[a:b]))
        devs.append(abs(c - target_for(c, key, 100.0)) * 100)
    if not devs:
        return {"notes": 0, "median_cents": None, "off_share": None, "score": 50.0}
    devs = np.array(devs)
    med, off = float(np.median(devs)), float(np.mean(devs > 40))
    return {"notes": int(len(devs)), "median_cents": round(med, 1), "off_share": round(off, 3), "score": round(med + 60 * off, 2)}


def tune_to_key(audio: np.ndarray, sr: int, key: dict | None, strength: float = 0.6,
                max_snap_cents: float = 70.0) -> tuple[np.ndarray, dict]:
    import parselmouth
    from parselmouth.praat import call

    x = np.asarray(audio, dtype=np.float32)
    strength = float(np.clip(strength, 0.0, 1.0))
    report = {"strength": strength, "key": key.get("name") if key else None, "applied": False}
    if strength <= 0 or len(x) < sr:
        return x, report
    m = _mono(x)
    if np.max(np.abs(m)) < 1e-4:
        return x, report
    snd = parselmouth.Sound(m, sampling_frequency=sr)
    step, floor, ceil = 0.01, 70.0, 900.0
    pitch = snd.to_pitch_ac(time_step=step, pitch_floor=floor, pitch_ceiling=ceil)
    f0 = pitch.selected_array["frequency"]
    times = np.asarray(pitch.xs())
    voiced = f0 > 0
    midi = np.where(voiced, 69 + 12 * np.log2(np.maximum(f0, 1e-6) / 440.0), 0.0)

    offset = np.zeros(len(f0))  # semitones per frame
    before, after = [], []
    for a, b in note_segments(midi, voiced):
        c = float(np.median(midi[a:b]))
        t = target_for(c, key, max_snap_cents)
        off = strength * (t - c)
        offset[a:b] = off
        before.append(abs(c - t) * 100)
        after.append(abs(c + off - t) * 100)
    report.update({"notes": len(before)})
    if not before or np.max(np.abs(offset)) * 100 < 3:
        report.update({"median_cents_before": round(float(np.median(before)), 1) if before else None})
        return x, report

    manip = call(snd, "To Manipulation", step, floor, ceil)
    tier = call(manip, "Extract pitch tier")
    n_pts = int(call(tier, "Get number of points"))
    new_tier = call("Create PitchTier", "tuned", snd.xmin, snd.xmax)
    for i in range(1, n_pts + 1):
        tt = call(tier, "Get time from index", i)
        ff = call(tier, "Get value at index", i)
        k = int(np.clip(np.searchsorted(times, tt), 0, len(times) - 1))
        if k > 0 and abs(times[k - 1] - tt) < abs(times[k] - tt):
            k -= 1
        call(new_tier, "Add point", tt, ff * 2 ** (offset[k] / 12.0))
    call([new_tier, manip], "Replace pitch tier")
    tuned = call(manip, "Get resynthesis (overlap-add)").values[0].astype(np.float32)
    if len(tuned) < len(m):
        tuned = np.concatenate([tuned, np.zeros(len(m) - len(tuned), dtype=np.float32)])
    tuned = tuned[: len(m)]

    # Use the PSOLA signal only where a note was moved (20 ms crossfades).
    hop = int(round(step * sr))
    mask_f = (np.abs(offset) > 0.005).astype(np.float32)
    frame_pos = np.clip((times * sr).astype(np.int64), 0, len(m) - 1)
    mask = np.zeros(len(m), dtype=np.float32)
    for k in np.nonzero(mask_f)[0]:
        s = max(0, frame_pos[k] - hop // 2)
        mask[s: s + hop] = 1.0
    w = max(1, int(sr * 0.02))
    mask = np.clip(np.convolve(mask, np.ones(w, dtype=np.float32) / w, mode="same"), 0.0, 1.0)

    if x.ndim == 1:
        out = x * (1 - mask) + tuned * mask
    else:
        mono_rms = float(np.sqrt(np.mean(m ** 2)) + 1e-9)
        out = np.empty_like(x)
        for ch in range(x.shape[1]):
            g = float(np.sqrt(np.mean(x[:, ch].astype(np.float64) ** 2)) / mono_rms)
            out[:, ch] = x[:, ch] * (1 - mask) + tuned * g * mask
    report.update({
        "applied": True,
        "notes_moved": int(np.sum(np.array(before) > 1)),
        "median_cents_before": round(float(np.median(before)), 1),
        "median_cents_after": round(float(np.median(after)), 1),
        "max_shift_cents": round(float(np.max(np.abs(offset)) * 100), 1),
        "share_of_audio_touched": round(float(np.mean(mask > 0.5)), 3),
    })
    return out.astype(np.float32), report
