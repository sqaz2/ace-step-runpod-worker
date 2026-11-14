# ACE-Step 1.5 Runpod worker

Serverless worker used by [generative.download](https://generative.download) `/create`.
It writes a song from Suno-style tags and lyrics with **ACE-Step 1.5** (2B turbo),
then splits out a vocal with **Demucs htdemucs_ft**, dries it with **UVR De-Echo-Aggressive** then **UVR-DeEcho-DeReverb** (same models as the site Dereverb tool), and applies a light gate.
The job returns **both** files: the gated vocal (primary) and the full mix.

The image is public and contains no secrets. Model weights are baked in so a
cold start does not download them.

## Settings

| | |
|---|---|
| Model | `Runware/acestep-v15-turbo-diffusers` (MIT), pinned to `be23effe` |
| Steps / shift | 16 / 3.0 |
| Dtype | float32 (float16 produced NaNs on a T4; not used) |
| Offload | none, on a 24 GB GPU |
| Separator | Demucs `htdemucs_ft`, two-stem, `shifts=0`, overlap 0.25 |
| Dereverb | UVR-De-Echo-Aggressive then UVR-DeEcho-DeReverb (window 512, aggression 5), on the GPU |
| Gate | open −40 dB, close −50 dB, hold 250 ms, pre 30 ms, 20 ms fades |
| Duration | 10–240 seconds |
| Output | 192 kb/s mp3, vocals + full mix |

## Input

```json
{
  "input": {
    "tags": "hip hop, boom bap, male vocals, gritty",
    "lyrics": "[verse]\n...\n[chorus]\n...",
    "duration": 60,
    "bpm": 90,
    "seed": 13,
    "vocal_language": "en"
  }
}
```

`bpm` and `seed` are optional. Duration is capped at 240 seconds.

## Output

```json
{
  "ok": true,
  "vocals_mp3_base64": "...",
  "mix_mp3_base64": "...",
  "seed": 13,
  "infer_seconds": 18.2,
  "separate_seconds": 3.1,
  "gate": { "open_fraction": 0.72, "n_regions": 5 },
  "primary": "vocals"
}
```

## Image

`ghcr.io/sqaz2/ace-step-runpod-worker:latest`

Built on GitHub Actions from this repository (see `.github/workflows/build.yml`).
The endpoint on Runpod is configured separately: flex workers, min 0, max 1,
no network volume.

## Bring-your-own-beat (base model, separate endpoint)

`Dockerfile.base` builds `ghcr.io/sqaz2/ace-step-runpod-worker:base-vN` with
**ACE-Step 1.5 2B base** (`Runware/acestep-v15-base-diffusers`, pinned
`9f8de7a3`), `ACESTEP_VARIANT=base`, 50 steps, guidance 7. It reuses the v3
Python env, Demucs and UVR weights but not the turbo weights. It runs on its
own Runpod endpoint; the turbo `/create` endpoint is unchanged and rejects a
`beat` with a clear error.

Add a `beat` object to the input. ACE-Step runs `task_type="lego"`,
`track_name="vocals"` with the beat as `src_audio`, then the same dry chain as
`/create` (htdemucs_ft, UVR x2, gate), so only a dry vocal comes back, the same
length as the clip.

```json
{
  "input": {
    "tags": "solo male rap vocal, dry vocal, clear voice",
    "lyrics": "[verse]\n...",
    "bpm": 92,
    "seed": 7,
    "beat": {
      "audio_base64": "<mp3 or wav>",
      "start_sec": 0,
      "end_sec": 40,
      "bpm": 92,
      "vocals": "auto"
    }
  }
}
```

- Send exactly one of `audio_base64` (`mp3_base64`/`wav_base64` aliases) or an `https://` `url`.
- The clip must be 10–120 s long.
- `vocals`:
  - `auto` (default): if Demucs finds a real singer in the clip, it is separated out and Lego writes over the instrumental.
  - `remove`: always separate the singer out first.
  - `keep`: send the clip as-is.
- Output:
  - `vocals_mp3_base64` (primary)
  - `preview_mp3_base64` (the beat, or the instrumental, plus the vocal)
  - `instrumental_mp3_base64`, only when an old vocal was removed
  - `beat_vocal`, the detection report
  - `task_type: "lego"` and `aligned: true`
  - `raw_stem_mp3_base64` when `return_raw: true`
