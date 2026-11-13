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
