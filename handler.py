#!/usr/bin/env python3
"""Runpod serverless worker: ACE-Step 1.5 turbo → full mix + gated dry vocal.

Proven settings from the Kaggle prototypes:
  Runware/acestep-v15-turbo-diffusers, 16 steps, shift 3.0, float32.
  No CPU offload on a 24 GB GPU. float16 is avoided (NaN on T4; untrusted here).

After generation, Demucs htdemucs_ft (two-stem, shifts=0) pulls a vocal,
then UVR-De-Echo-Aggressive and UVR-DeEcho-DeReverb dry it (same models and
defaults as the site Dereverb tool: window 512, aggression 5), then a light
gate (open -40 dB, close -50 dB, hold 250 ms). Both the gated vocal (primary)
and the full mix are returned as 192 kb/s mp3.
"""
from __future__ import annotations

import base64
import gc
import json
import os
import subprocess
import tempfile
import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
import torch

from dereverb import dereverb_vocals, get_uvr
from gate import gate_vocal, rms_db
import layers as vlayers

MODEL_DIR = Path(os.environ.get("PIPELINE_PATH", "/models/acestep-v15-turbo"))
DEMUCS_NAME = os.environ.get("DEMUCS_MODEL", "htdemucs_ft")
# "turbo" (default, /create) or "base" (bring-your-own-beat, Lego vocals).
VARIANT = os.environ.get("ACESTEP_VARIANT", "turbo").strip().lower()
IS_BASE = VARIANT == "base"
MODEL_ID = os.environ.get(
    "ACESTEP_MODEL_ID",
    "Runware/acestep-v15-base-diffusers" if IS_BASE else "Runware/acestep-v15-turbo-diffusers",
)
STEPS = int(os.environ.get("ACESTEP_STEPS", "50" if IS_BASE else "16"))
GUIDANCE = float(os.environ.get("ACESTEP_GUIDANCE", "7.0" if IS_BASE else "1.0"))
SHIFT = 3.0
MIN_DURATION = 10.0
MAX_DURATION = 240.0
# Bring-your-own-beat limits.
BEAT_MAX_SECONDS = 120.0
BEAT_MAX_BYTES = 24 * 1024 * 1024
BEAT_URL_TIMEOUT = 30
MP3_BITRATE = "192k"

PIPE = None
DEMUCS = None
DEMUCS_DEVICE = None


def log(msg: str) -> None:
    print(msg, flush=True)


def cuda_free() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        try:
            torch.cuda.synchronize()
        except Exception:
            pass


def to_audio(audio) -> np.ndarray:
    if isinstance(audio, (list, tuple)):
        audio = audio[0]
    if hasattr(audio, "detach"):
        audio = audio.detach().cpu().float().numpy()
    arr = np.asarray(audio)
    if arr.dtype == object:
        arr = np.asarray(arr.tolist())
    arr = np.asarray(arr, dtype=np.float32)
    if arr.ndim == 3:
        arr = arr[0]
    if arr.ndim == 2 and arr.shape[0] <= 2 and arr.shape[0] < arr.shape[1]:
        arr = arr.T
    if arr.ndim == 1:
        arr = arr[:, None]
    return arr


def quality(arr: np.ndarray) -> dict[str, Any]:
    finite = np.isfinite(arr)
    n = arr.size
    n_bad = int((~finite).sum())
    good = arr[finite]
    peak = float(np.max(np.abs(good))) if good.size else 0.0
    rms = float(np.sqrt(np.mean(np.square(good)))) if good.size else 0.0
    unique = int(np.unique(np.round(good, 5)).size) if good.size else 0
    ok = n_bad == 0 and peak > 1e-3 and unique > 100
    return {
        "ok": ok,
        "samples": n,
        "nonfinite": n_bad,
        "peak": peak,
        "rms": rms,
        "unique": unique,
        "mean": float(good.mean()) if good.size else 0.0,
        "shape": list(arr.shape),
        "reason": None if ok else ("non-finite samples" if n_bad else "silent or collapsed audio"),
    }


def peak_limit(arr: np.ndarray, ceiling: float = 0.98) -> np.ndarray:
    peak = float(np.max(np.abs(arr))) if arr.size else 0.0
    if peak > ceiling:
        arr = arr / peak * ceiling
    return arr


def encode_mp3(wav: np.ndarray, sr: int, path: Path, bitrate: str = MP3_BITRATE) -> None:
    wav = peak_limit(np.asarray(wav, dtype=np.float32))
    wav_path = path.with_suffix(".wav")
    sf.write(str(wav_path), wav, sr, subtype="PCM_16")
    subprocess.run(
        ["ffmpeg", "-y", "-i", str(wav_path), "-codec:a", "libmp3lame", "-b:a", bitrate, str(path)],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    wav_path.unlink(missing_ok=True)
    if not path.exists() or path.stat().st_size < 256:
        raise RuntimeError("ffmpeg did not write a usable mp3")


def b64_file(path: Path) -> str:
    return base64.b64encode(path.read_bytes()).decode("ascii")


def nan_watch(pipe, step, timestep, kwargs):
    latents = kwargs.get("latents")
    if latents is None:
        return
    n_nan = int(torch.isnan(latents).sum().item())
    n_inf = int(torch.isinf(latents).sum().item())
    peak = float(latents.detach().float().abs().max().item()) if latents.numel() else 0
    log(f"  step {step}: nan={n_nan} inf={n_inf} peak={peak:.4f} dtype={latents.dtype}")
    if n_nan or n_inf:
        pipe._interrupt = True
        log("  interrupting: non-finite latents")


def enable_vae_tiling(vae) -> None:
    if vae is None:
        return
    for name in ("enable_tiling", "enable_slicing", "enable_vae_tiling"):
        fn = getattr(vae, name, None)
        if callable(fn):
            try:
                fn()
                log(f"VAE: {name}()")
                return
            except Exception as e:
                log(f"VAE {name}() failed: {e}")


def decode_tiled(pipe, xt, chunk_sec: float = 20.0, overlap_sec: float = 2.0):
    if hasattr(xt, "detach"):
        latents = xt.detach()
    else:
        latents = torch.as_tensor(xt)
    if latents.ndim == 2:
        latents = latents.unsqueeze(0)
    b, t, c = latents.shape
    lps = float(getattr(pipe, "latents_per_second", 25.0))
    sr = int(getattr(pipe, "sample_rate", 48000))
    chunk = max(1, int(round(chunk_sec * lps)))
    overlap = max(0, int(round(overlap_sec * lps)))
    if overlap >= chunk:
        overlap = max(0, chunk // 8)
    hop = max(1, chunk - overlap)
    log(f"tiled decode: T={t} lps={lps} chunk={chunk} overlap={overlap}")
    vae = pipe.vae
    enable_vae_tiling(vae)
    param = next(vae.parameters())

    def decode_one(sl_btc):
        x = sl_btc.transpose(1, 2).contiguous().to(device=param.device, dtype=param.dtype)
        with torch.no_grad():
            out = vae.decode(x).sample
        return out.detach().float().cpu()

    pieces = []
    for i, s in enumerate(range(0, t, hop)):
        e = min(t, s + chunk)
        if e <= s:
            continue
        sl = latents[:, s:e, :]
        t0 = time.time()
        wav = decode_one(sl)
        n_samples = wav.shape[-1]
        spp = n_samples / float(e - s)
        log(f"  chunk {i} latent[{s}:{e}] -> {n_samples} samples in {time.time()-t0:.1f}s")
        pieces.append((s, e, wav, spp))
        if e >= t:
            break
        cuda_free()
    if not pieces:
        raise RuntimeError("tiled decode produced no chunks")
    spp = float(np.median([p[3] for p in pieces]))
    total = int(round(t * spp))
    ch = pieces[0][2].shape[-2] if pieces[0][2].ndim == 3 else 1
    batch = pieces[0][2].shape[0]
    canvas = np.zeros((batch, ch, total), dtype=np.float32)
    weight = np.zeros((total,), dtype=np.float32)
    for s, e, wav, _ in pieces:
        if wav.ndim == 2:
            wav = wav[:, None, :]
        wav_np = wav.numpy() if hasattr(wav, "numpy") else np.asarray(wav)
        a = int(round(s * spp))
        b = a + wav_np.shape[-1]
        if b > total:
            wav_np = wav_np[..., : total - a]
            b = total
        n = b - a
        if n <= 0:
            continue
        w = np.ones(n, dtype=np.float32)
        fade = int(round(overlap * spp))
        fade = min(fade, n // 3) if n > 6 else 0
        if fade > 1:
            ramp = np.linspace(0, 1, fade, dtype=np.float32)
            if s > 0:
                w[:fade] = ramp
            if e < t:
                w[-fade:] *= ramp[::-1]
        canvas[:, :, a:b] += wav_np[..., :n] * w
        weight[a:b] += w
    weight = np.maximum(weight, 1e-6)
    canvas = canvas / weight[None, None, :]
    log(f"tiled decode assembled {total} samples ({total/sr:.2f}s)")
    return canvas, sr


def load_pipe():
    from diffusers import AceStepPipeline

    src = str(MODEL_DIR) if MODEL_DIR.exists() else MODEL_ID
    log(f"loading AceStepPipeline from {src} float32, no offload")
    t0 = time.time()
    try:
        pipe = AceStepPipeline.from_pretrained(src, torch_dtype=torch.float32, local_files_only=MODEL_DIR.exists())
    except TypeError:
        pipe = AceStepPipeline.from_pretrained(src, dtype=torch.float32, local_files_only=MODEL_DIR.exists())
    pipe = pipe.to("cuda")
    enable_vae_tiling(getattr(pipe, "vae", None))
    log(f"pipeline ready in {time.time()-t0:.1f}s on {torch.cuda.get_device_name(0)}")
    return pipe


def load_demucs():
    from demucs.pretrained import get_model

    t0 = time.time()
    model = get_model(DEMUCS_NAME)
    model.eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # stay on CPU until we need it, so it doesn't sit in VRAM next to ACE-Step
    model.to("cpu")
    log(f"demucs {DEMUCS_NAME} loaded in {time.time()-t0:.1f}s (moved to GPU per job)")
    return model, device


def get_pipe():
    global PIPE
    if PIPE is None:
        PIPE = load_pipe()
    return PIPE


def get_demucs():
    global DEMUCS, DEMUCS_DEVICE
    if DEMUCS is None:
        DEMUCS, DEMUCS_DEVICE = load_demucs()
    return DEMUCS, DEMUCS_DEVICE


def generate_mix(
    pipe,
    tags: str,
    lyrics: str,
    duration: float,
    bpm: float | None,
    seed: int,
    language: str,
    extra: dict[str, Any] | None = None,
):
    log(f"DiT generate duration={duration}s steps={STEPS} cfg={GUIDANCE} shift={SHIFT} bpm={bpm} seed={seed} task={(extra or {}).get('task_type', 'text2music')}")
    t0 = time.time()
    gen = torch.Generator(device="cpu").manual_seed(int(seed))
    kwargs = dict(
        prompt=tags,
        lyrics=lyrics,
        audio_duration=float(duration),
        vocal_language=language,
        num_inference_steps=STEPS,
        shift=SHIFT,
        timesignature="4",
        generator=gen,
        callback_on_step_end=nan_watch,
    )
    if IS_BASE:
        kwargs["guidance_scale"] = GUIDANCE
    if bpm is not None:
        kwargs["bpm"] = float(bpm)
    if extra:
        kwargs.update(extra)

    # Prefer a direct audio decode on 24 GB; fall back to latent + tiled VAE.
    audio = None
    sr = int(getattr(pipe, "sample_rate", 48000))
    decode_path = None
    try:
        result = pipe(output_type="np", **kwargs)
        elapsed = time.time() - t0
        raw = getattr(result, "audios", None)
        if raw is None:
            raw = getattr(result, "audio", result)
        audio = to_audio(raw)
        decode_path = "direct-np"
        log(f"direct decode in {elapsed:.1f}s shape={audio.shape}")
    except torch.cuda.OutOfMemoryError:
        log("direct decode OOM, falling back to latent + tiled VAE")
        cuda_free()
        audio = None
    except Exception as e:
        log(f"direct decode failed ({e}); falling back to latent + tiled VAE")
        traceback.print_exc()
        cuda_free()
        audio = None

    if audio is None:
        t1 = time.time()
        result = pipe(output_type="latent", **kwargs)
        xt = getattr(result, "audios", None)
        if xt is None:
            xt = getattr(result, "audio", result)
        if hasattr(xt, "detach"):
            n_nan = int(torch.isnan(xt).sum().item())
            n_inf = int(torch.isinf(xt).sum().item())
            if n_nan or n_inf:
                raise RuntimeError(f"non-finite latents nan={n_nan} inf={n_inf}")
        audio, sr = decode_tiled(pipe, xt, chunk_sec=20.0, overlap_sec=2.0)
        audio = to_audio(audio)
        decode_path = "tiled-gpu-20s"
        elapsed = time.time() - t0
        log(f"tiled path finished in {elapsed:.1f}s (dit {t1-t0:.1f}s)")
        del xt
    else:
        elapsed = time.time() - t0

    q = quality(audio)
    log(f"mix quality: {q}")
    if not q["ok"]:
        raise RuntimeError(f"bad mix audio: {q['reason']}")
    audio = peak_limit(audio)
    return audio, sr, elapsed, decode_path, q


def separate_vocals(mix: np.ndarray, sr: int) -> tuple[np.ndarray, int, float]:
    from demucs.apply import apply_model
    from demucs.audio import convert_audio

    model, device = get_demucs()
    t0 = time.time()
    wav = np.asarray(mix, dtype=np.float32)
    if wav.ndim == 1:
        wav = np.stack([wav, wav], axis=0)
    else:
        wav = wav.T  # [C, T]
    if wav.shape[0] == 1:
        wav = np.concatenate([wav, wav], axis=0)
    wav_t = torch.from_numpy(np.ascontiguousarray(wav))
    wav_t = convert_audio(wav_t, sr, model.samplerate, model.audio_channels)
    model.to(device)
    try:
        with torch.no_grad():
            sources = apply_model(
                model,
                wav_t[None].to(device),
                device=device,
                split=True,
                overlap=0.25,
                shifts=0,
            )[0]
        vocals = sources[list(model.sources).index("vocals")]
        # Keep the native Demucs rate (44.1 kHz). UVR runs at 44.1, then we
        # resample to the mix rate, matching the 03_normal_vocals_cleaner take.
        out = vocals.cpu().numpy().T  # [T, C]
        out_sr = int(model.samplerate)
    finally:
        # Free VRAM so a warm ACE-Step request isn't squeezed.
        model.to("cpu")
        cuda_free()
    elapsed = time.time() - t0
    log(f"demucs {DEMUCS_NAME} vocals in {elapsed:.1f}s shape={out.shape} sr={out_sr} rms_db={rms_db(out):.1f}")
    return out, out_sr, elapsed


def _decode_beat_bytes(data: bytes, start: float, end: float | None, sr: int) -> np.ndarray:
    """Decode any ffmpeg-readable audio, clip [start, end), return [T, 2] float32 at sr."""
    with tempfile.TemporaryDirectory() as td:
        src = Path(td) / "beat.bin"
        src.write_bytes(data)
        cmd = ["ffmpeg", "-nostdin", "-v", "error", "-y"]
        if start > 0:
            cmd += ["-ss", f"{start:.3f}"]
        cmd += ["-i", str(src)]
        if end is not None:
            cmd += ["-t", f"{max(0.0, end - start):.3f}"]
        cmd += ["-map", "0:a:0", "-ac", "2", "-ar", str(sr), "-f", "f32le", "-acodec", "pcm_f32le", "pipe:1"]
        proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=120)
        if proc.returncode != 0:
            raise ValueError("Could not read the beat audio. Try a WAV or MP3.")
        pcm = np.frombuffer(proc.stdout, dtype=np.float32)
    if pcm.size < 2:
        raise ValueError("The beat clip is empty.")
    pcm = pcm[: pcm.size - (pcm.size % 2)].reshape(-1, 2).copy()
    if not np.all(np.isfinite(pcm)):
        pcm = np.nan_to_num(pcm)
    return pcm


def _fetch_beat_url(url: str) -> bytes:
    import urllib.request

    if not isinstance(url, str) or not url.startswith("https://"):
        raise ValueError("The beat URL must start with https://")
    req = urllib.request.Request(url, headers={"User-Agent": "ace-step-runpod-worker"})
    with urllib.request.urlopen(req, timeout=BEAT_URL_TIMEOUT) as resp:
        ctype = (resp.headers.get("content-type") or "").lower()
        if ctype and not (ctype.startswith("audio/") or ctype.startswith("video/") or "octet-stream" in ctype):
            raise ValueError("The beat URL is not an audio file.")
        data = resp.read(BEAT_MAX_BYTES + 1)
    if len(data) > BEAT_MAX_BYTES:
        raise ValueError("The beat file is too big.")
    return data


def parse_beat(raw: Any) -> dict[str, Any] | None:
    """Validate the optional bring-your-own-beat block. Returns None when absent."""
    if raw is None:
        return None
    if not isinstance(raw, dict):
        raise ValueError("beat must be an object.")
    b64 = raw.get("audio_base64") or raw.get("mp3_base64") or raw.get("wav_base64")
    url = raw.get("url")
    if bool(b64) == bool(url):
        raise ValueError("Send the beat as exactly one of audio_base64 or url.")
    if url is not None and (not isinstance(url, str) or not url.startswith("https://") or len(url) > 2048):
        raise ValueError("The beat URL must start with https://")
    if b64 is not None and not isinstance(b64, str):
        raise ValueError("beat audio_base64 must be a string.")
    if b64 and len(b64) > BEAT_MAX_BYTES * 4 // 3 + 16:
        raise ValueError("The beat file is too big.")
    start = float(raw.get("start_sec") or 0.0)
    end = raw.get("end_sec")
    end = None if end in (None, "") else float(end)
    if not np.isfinite(start) or start < 0:
        raise ValueError("beat start_sec must be 0 or more.")
    if end is not None and (not np.isfinite(end) or end <= start):
        raise ValueError("beat end_sec must be after start_sec.")
    if end is not None and end - start > BEAT_MAX_SECONDS + 0.5:
        raise ValueError(f"Pick {int(BEAT_MAX_SECONDS)} seconds of beat or less.")
    bpm = raw.get("bpm")
    if bpm in (None, ""):
        bpm = None
    else:
        bpm = float(bpm)
        if not np.isfinite(bpm) or bpm < 30 or bpm > 300:
            raise ValueError("Beat BPM should be between 30 and 300, or left blank.")
    vocals = str(raw.get("vocals") or "auto").strip().lower()
    if vocals not in ("auto", "keep", "remove"):
        raise ValueError("beat vocals must be auto, keep, or remove.")
    return {"b64": b64, "url": url, "start": start, "end": end, "bpm": bpm, "vocals": vocals}


def load_beat(beat: dict[str, Any], sr: int) -> np.ndarray:
    if beat["b64"]:
        try:
            data = base64.b64decode(beat["b64"], validate=False)
        except Exception as exc:
            raise ValueError("The beat audio is not valid base64.") from exc
        if len(data) > BEAT_MAX_BYTES:
            raise ValueError("The beat file is too big.")
    else:
        data = _fetch_beat_url(beat["url"])
    pcm = _decode_beat_bytes(data, beat["start"], beat["end"], sr)
    secs = len(pcm) / float(sr)
    if secs < MIN_DURATION:
        raise ValueError(f"Pick at least {int(MIN_DURATION)} seconds of beat.")
    if secs > BEAT_MAX_SECONDS + 0.5:
        pcm = pcm[: int(BEAT_MAX_SECONDS * sr)]
    return pcm


def fit_length(wav: np.ndarray, frames: int) -> np.ndarray:
    wav = np.asarray(wav, dtype=np.float32)
    if wav.ndim == 1:
        wav = wav[:, None]
    if wav.shape[1] == 1:
        wav = np.concatenate([wav, wav], axis=1)
    if len(wav) >= frames:
        return wav[:frames]
    pad = np.zeros((frames - len(wav), wav.shape[1]), dtype=np.float32)
    return np.concatenate([wav, pad], axis=0)


# A clip "has a vocal" when Demucs finds a vocal that is loud relative to the
# clip and present for a fair share of it. Instrumentals usually leak < -30 dB.
VOCAL_REL_DB = float(os.environ.get("BEAT_VOCAL_REL_DB", "-20"))
VOCAL_ACTIVE_FRACTION = float(os.environ.get("BEAT_VOCAL_ACTIVE_FRACTION", "0.15"))


def detect_and_strip_vocal(beat: np.ndarray, sr: int, mode: str) -> tuple[np.ndarray, dict[str, Any], np.ndarray | None, np.ndarray | None]:
    """Return (context_audio, report, instrumental_or_None, old_vocal_or_None). Never edits `beat`."""
    report: dict[str, Any] = {"mode": mode, "detected": False, "removed": False}
    if mode == "keep":
        return beat, report, None, None
    vocals, voc_sr, sep_s = separate_vocals(beat, sr)
    if voc_sr != sr:
        import torchaudio
        t = torch.from_numpy(np.ascontiguousarray(vocals.T, dtype=np.float32))
        vocals = torchaudio.functional.resample(t, voc_sr, sr).numpy().T.astype(np.float32)
    vocals = fit_length(vocals, len(beat))
    rel_db = rms_db(vocals) - rms_db(beat)
    mono = np.mean(vocals, axis=1)
    hop = int(sr * 0.05)
    n = max(1, len(mono) // hop)
    frames = mono[: n * hop].reshape(n, hop)
    frame_db = 20 * np.log10(np.sqrt(np.mean(frames ** 2, axis=1)) + 1e-9)
    beat_db = rms_db(beat)
    active = float(np.mean(frame_db > beat_db - 12.0)) if n else 0.0
    detected = rel_db > VOCAL_REL_DB and active > VOCAL_ACTIVE_FRACTION
    report.update({"detected": bool(detected), "vocal_rel_db": round(float(rel_db), 2), "vocal_active_fraction": round(active, 3), "detect_seconds": round(sep_s, 2)})
    log(f"vocal check: {report}")
    if mode == "remove" or detected:
        instrumental = (beat - vocals).astype(np.float32)
        report["removed"] = True
        return instrumental, report, instrumental, vocals
    return beat, report, None, None


def parse_input(event: dict) -> dict[str, Any]:
    inp = event.get("input") if isinstance(event, dict) else None
    if not isinstance(inp, dict):
        raise ValueError("Send a JSON object with tags and lyrics.")
    tags = inp.get("tags") if isinstance(inp.get("tags"), str) else ""
    lyrics = inp.get("lyrics") if isinstance(inp.get("lyrics"), str) else ""
    tags, lyrics = tags.strip(), lyrics.strip()
    if not tags and not lyrics:
        raise ValueError("Add some style tags or lyrics first.")
    if len(tags) > 512:
        raise ValueError("The style tags are too long.")
    if len(lyrics) > 4096:
        raise ValueError("The lyrics are too long.")
    if not lyrics:
        lyrics = "[Instrumental]"
    duration = float(inp.get("duration", 60))
    if not np.isfinite(duration):
        raise ValueError("Duration must be a number.")
    duration = max(MIN_DURATION, min(MAX_DURATION, duration))
    bpm = inp.get("bpm", None)
    if bpm is not None and bpm != "":
        bpm = float(bpm)
        if not np.isfinite(bpm) or bpm < 30 or bpm > 300:
            raise ValueError("BPM should be between 30 and 300, or left blank.")
    else:
        bpm = None
    language = inp.get("vocal_language") or inp.get("vocalLanguage") or "en"
    language = str(language).strip()[:8] or "en"
    seed = inp.get("seed", None)
    if seed is None or seed == "":
        seed = int(torch.randint(0, 2**31 - 1, (1,)).item())
    else:
        seed = int(seed)
    beat = parse_beat(inp.get("beat"))
    if beat is not None and not IS_BASE:
        raise ValueError(
            "This endpoint runs ACE-Step turbo, which cannot write vocals over your beat. "
            "Bring-your-own-beat needs the base-model endpoint."
        )
    if beat is not None and beat["bpm"] is not None:
        bpm = beat["bpm"]
    if beat is not None and lyrics == "[Instrumental]":
        raise ValueError("Add some lyrics so there is something to sing.")
    raw_layers = inp.get("layers") or []
    if not isinstance(raw_layers, list) or any(l not in ("double", "harmony", "adlibs") for l in raw_layers):
        raise ValueError("layers can include double, harmony, and adlibs.")
    harmony_interval = "fifth" if inp.get("harmony_interval") == "fifth" else "third"
    timesig = str(inp.get("timesignature") or "4").strip()
    if timesig not in ("2", "3", "4", "6"):
        timesig = "4"
    keyscale = inp.get("keyscale")
    keyscale = keyscale.strip()[:16] if isinstance(keyscale, str) and keyscale.strip() else None
    return {
        "layers": sorted(set(raw_layers)) if beat is not None else [],
        "harmony_interval": harmony_interval,
        "timesignature": timesig,
        "keyscale": keyscale,
        "beat": beat,
        "return_raw": bool(inp.get("return_raw")),
        "tags": tags,
        "lyrics": lyrics,
        "duration": duration,
        "bpm": bpm,
        "vocal_language": language,
        "seed": seed,
    }


def handler(event: dict) -> dict[str, Any]:
    t_all = time.time()
    job = parse_input(event)
    log(f"job: duration={job['duration']} seed={job['seed']} tags={job['tags']!r}")
    device_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
    if not torch.cuda.is_available():
        raise RuntimeError("This worker needs a GPU.")

    pipe = get_pipe()
    # Make sure the pipeline is on GPU (warm jobs after demucs should already be).
    try:
        pipe.to("cuda")
    except Exception:
        pass
    cuda_free()

    if job["beat"] is not None:
        return handle_beat(pipe, job, t_all, device_name)

    mix, sr, infer_s, decode_path, mix_q = generate_mix(
        pipe,
        job["tags"],
        job["lyrics"],
        job["duration"],
        job["bpm"],
        job["seed"],
        job["vocal_language"],
    )
    cuda_free()

    vocals, voc_sr, sep_s = separate_vocals(mix, sr)
    dry, dry_sr, uvr_s = dereverb_vocals(vocals, voc_sr)
    if dry_sr != sr:
        import torchaudio
        t_dry = torch.from_numpy(np.ascontiguousarray(dry.T, dtype=np.float32))
        dry = torchaudio.functional.resample(t_dry, dry_sr, sr).numpy().T.astype(np.float32)
        dry_sr = sr
    gated, gate_report = gate_vocal(dry, dry_sr)
    log(f"gate: {json.dumps({k: gate_report[k] for k in gate_report if k != 'regions_sec'})}")

    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        vocals_mp3 = td / "vocals.mp3"
        mix_mp3 = td / "mix.mp3"
        encode_mp3(gated, dry_sr, vocals_mp3)
        encode_mp3(mix, sr, mix_mp3)
        vocals_b64 = b64_file(vocals_mp3)
        mix_b64 = b64_file(mix_mp3)
        vocals_bytes = vocals_mp3.stat().st_size
        mix_bytes = mix_mp3.stat().st_size

    total = time.time() - t_all
    log(f"done in {total:.1f}s infer={infer_s:.1f}s sep={sep_s:.1f}s uvr={uvr_s:.1f}s voc={vocals_bytes} mix={mix_bytes}")
    return {
        "ok": True,
        "vocals_mp3_base64": vocals_b64,
        "mix_mp3_base64": mix_b64,
        "vocals_bytes": vocals_bytes,
        "mix_bytes": mix_bytes,
        "sample_rate": sr,
        "duration_actual": round(len(mix) / float(sr), 3),
        "duration_requested": job["duration"],
        "seed": job["seed"],
        "steps": STEPS,
        "shift": SHIFT,
        "bpm": job["bpm"],
        "tags": job["tags"],
        "lyrics": job["lyrics"],
        "infer_seconds": round(infer_s, 2),
        "separate_seconds": round(sep_s, 2),
        "uvr_seconds": round(uvr_s, 2),
        "total_seconds": round(total, 2),
        "decode_path": decode_path,
        "dtype": "float32",
        "device": device_name,
        "model": MODEL_ID,
        "separator": DEMUCS_NAME,
        "dereverb": "UVR-De-Echo-Aggressive+UVR-DeEcho-DeReverb",
        "gate": gate_report,
        "mix_quality": mix_q,
        "primary": "vocals",
    }


def handle_beat(pipe, job: dict[str, Any], t_all: float, device_name: str) -> dict[str, Any]:
    """Lego: write a vocal stem from the beat as audio context. The beat is never altered."""
    sr = int(getattr(pipe, "sample_rate", 48000))
    t_dec = time.time()
    beat = load_beat(job["beat"], sr)
    frames = len(beat)
    beat_secs = frames / float(sr)
    log(f"beat: {beat_secs:.2f}s at {sr} Hz, decoded in {time.time()-t_dec:.1f}s rms_db={rms_db(beat):.1f}")
    # If the clip already has a singer, swap it: Lego writes over the instrumental.
    context, vocal_report, instrumental, old_vocal = detect_and_strip_vocal(beat, sr, job["beat"].get("vocals", "auto"))
    key = vlayers.estimate_key(context, sr)
    keyscale = job["keyscale"] or (key["name"] if key["confidence"] >= 0.75 else None)
    log(f"key estimate {key} -> keyscale={keyscale}")
    src = torch.from_numpy(np.ascontiguousarray(context.T, dtype=np.float32))  # [2, T]

    def lego(track: str, tags: str, lyrics: str, seed: int):
        extra = {"task_type": "lego", "track_name": track, "src_audio": src, "repainting_start": 0.0,
                 "repainting_end": -1, "timesignature": job["timesignature"]}
        if keyscale:
            extra["keyscale"] = keyscale
        return generate_mix(pipe, tags, lyrics, beat_secs, job["bpm"], seed, job["vocal_language"], extra=extra)

    def dry(stem: np.ndarray) -> tuple[np.ndarray, dict[str, Any], float, float]:
        vocals, voc_sr, sep = separate_vocals(stem, sr)
        d, d_sr, uvr = dereverb_vocals(vocals, voc_sr)
        if d_sr != sr:
            import torchaudio
            t_dry = torch.from_numpy(np.ascontiguousarray(d.T, dtype=np.float32))
            d = torchaudio.functional.resample(t_dry, d_sr, sr).numpy().T.astype(np.float32)
        g, rep = gate_vocal(d, sr)
        return fit_length(g, frames), rep, sep, uvr

    stem, stem_sr, infer_s, decode_path, stem_q = lego("vocals", job["tags"], job["lyrics"], job["seed"])
    cuda_free()
    if stem_sr != sr:
        raise RuntimeError(f"unexpected sample rate {stem_sr}")
    stem = fit_length(stem, frames)
    # Same dry chain as /create, even though Lego already returns a vocal stem.
    gated, gate_report, sep_s, uvr_s = dry(stem)
    log(f"gate: {json.dumps({k: gate_report[k] for k in gate_report if k != 'regions_sec'})}")

    # Level: match the vocal we swapped out, or sit a little under the beat.
    target_db = vlayers.active_rms_db(old_vocal, sr) if old_vocal is not None else vlayers.active_rms_db(context, sr) - 2.0
    if target_db < -60:
        target_db = vlayers.active_rms_db(context, sr) - 2.0
    gated, lead_gain = vlayers.level_match(gated, sr, target_db)
    gated = vlayers.edge_fades(gated, sr)

    layer_audio: dict[str, np.ndarray] = {}
    layer_report: dict[str, Any] = {}
    t_layers = time.time()
    if "double" in job["layers"]:
        layer_audio["double"] = vlayers.edge_fades(vlayers.make_double(gated, sr, job["seed"]), sr)
        layer_report["double"] = {"method": "detuned, time-varied copy of the lead (+9 cents, 18-32 ms)"}
    if "harmony" in job["layers"]:
        harm, rep = vlayers.make_harmony(gated, sr, key, job["harmony_interval"])
        layer_audio["harmony"] = vlayers.edge_fades(harm, sr)
        layer_report["harmony"] = rep
    if "adlibs" in job["layers"]:
        a_stem, _, a_infer, _, _ = lego("backing_vocals", "ad-libs, hype background vocal shouts, sparse, short phrases, dry vocal",
                                        vlayers.adlib_lyrics(job["lyrics"], beat_secs), job["seed"] + 101)
        cuda_free()
        a_gated, a_rep, _, _ = dry(fit_length(a_stem, frames))
        a_gated = a_gated * vlayers.gaps_mask(gated, sr)[:, None]
        a_gated, a_gain = vlayers.level_match(a_gated, sr, target_db - 5.0)
        layer_audio["adlibs"] = vlayers.edge_fades(a_gated, sr)
        layer_report["adlibs"] = {"method": "second Lego take (backing_vocals) gated to the gaps in the lead",
                                  "infer_seconds": round(a_infer, 2), "open_fraction_before_gaps": a_rep.get("open_fraction"),
                                  "rms_db": round(vlayers.active_rms_db(layer_audio["adlibs"], sr), 1)}
    layers_s = time.time() - t_layers

    # Preview only: the beat (instrumental when the old vocal was swapped) plus the dry vocal(s).
    preview = context + gated + sum((v for v in layer_audio.values()), np.zeros_like(gated))

    out: dict[str, Any] = {}
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        vocals_mp3 = td / "vocals.mp3"
        preview_mp3 = td / "preview.mp3"
        encode_mp3(gated, sr, vocals_mp3)
        encode_mp3(preview, sr, preview_mp3)
        out["vocals_mp3_base64"] = b64_file(vocals_mp3)
        out["preview_mp3_base64"] = b64_file(preview_mp3)
        out["vocals_bytes"] = vocals_mp3.stat().st_size
        out["preview_bytes"] = preview_mp3.stat().st_size
        for name, audio in layer_audio.items():
            path = td / f"{name}.mp3"
            encode_mp3(audio, sr, path, bitrate="128k")
            out[f"{name}_mp3_base64"] = b64_file(path)
        if instrumental is not None:
            inst_mp3 = td / "instrumental.mp3"
            encode_mp3(instrumental, sr, inst_mp3)
            out["instrumental_mp3_base64"] = b64_file(inst_mp3)
        if job["return_raw"]:
            raw_mp3 = td / "raw.mp3"
            encode_mp3(stem, sr, raw_mp3)
            out["raw_stem_mp3_base64"] = b64_file(raw_mp3)

    total = time.time() - t_all
    log(f"beat job done in {total:.1f}s infer={infer_s:.1f}s sep={sep_s:.1f}s uvr={uvr_s:.1f}s")
    out.update({
        "ok": True,
        "task_type": "lego",
        "track_name": "vocals",
        "sample_rate": sr,
        "beat_duration": round(beat_secs, 3),
        "duration_actual": round(frames / float(sr), 3),
        "aligned": True,
        "beat_start_sec": job["beat"]["start"],
        "seed": job["seed"],
        "steps": STEPS,
        "guidance_scale": GUIDANCE,
        "shift": SHIFT,
        "bpm": job["bpm"],
        "tags": job["tags"],
        "lyrics": job["lyrics"],
        "infer_seconds": round(infer_s, 2),
        "separate_seconds": round(sep_s, 2),
        "uvr_seconds": round(uvr_s, 2),
        "total_seconds": round(total, 2),
        "decode_path": decode_path,
        "dtype": "float32",
        "device": device_name,
        "model": MODEL_ID,
        "separator": DEMUCS_NAME,
        "dereverb": "UVR-De-Echo-Aggressive+UVR-DeEcho-DeReverb",
        "gate": gate_report,
        "stem_quality": stem_q,
        "voiceSwapApplied": False,
        "beat_vocal": vocal_report,
        "key": key,
        "keyscale_sent": keyscale,
        "timesignature": job["timesignature"],
        "lead_gain_db": lead_gain,
        "lead_target_db": round(float(target_db), 1),
        "layers": layer_report,
        "layers_seconds": round(layers_s, 2),
        "primary": "vocals",
    })
    return out


def preload() -> None:
    log(f"=== ace-step 1.5 worker starting ({VARIANT}, {MODEL_ID}, {STEPS} steps) ===")
    if torch.cuda.is_available():
        log(f"cuda {torch.version.cuda} device={torch.cuda.get_device_name(0)} "
            f"vram={torch.cuda.get_device_properties(0).total_memory/1024**3:.1f}G")
    else:
        log("WARNING: no CUDA at startup")
    get_pipe()
    get_demucs()
    get_uvr()
    log("preload complete")


if __name__ == "__main__":
    import runpod

    preload()
    runpod.serverless.start({"handler": handler})
