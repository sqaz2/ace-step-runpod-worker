# Handoff: Generative audio track (bring-your-own-beat vocals)

Written Fri 13 Nov 2026, 11:35 PM America/Edmonton, for whoever picks this up next (e.g. ChatGPT).
Owner: Will Crowell (GitHub sqaz2). Nothing here is merged or live.

## What Will wants
- In the studio: add a track of type "Generative audio". A tip says to arm it and select part of the beat.
  While armed with a selection, a panel shows lyrics/vibe + Generate. Dry vocals come back on that track,
  aligned to the selection. The beat is never altered.
- If the selection already has a vocal: SWAP it (separate, write new vocal over the instrumental,
  keep the old vocal on a muted "Original vocal (muted)" track; never delete; one undo).
- Options: Double, Harmony (3rd/5th), Ad-libs, each on its own track under the lead.
- Vocals must sound CLEAN and dry. They don't need to sound like Will. Same dry chain as /create:
  htdemucs_ft -> UVR De-Echo-Aggressive -> UVR-DeEcho-DeReverb -> gate.

## Where the code is
- Site PR (OPEN, do not merge until Will approves): https://github.com/sqaz2/Generative-download/pull/204
  branch `byob-generative-vocals`. Key files: src/components/GenerativeVocals.tsx, src/lib/beat-vocals.ts,
  worker/beat-vocals-gateway.ts (+ test). Route sits behind the studio login like /api/create/jobs.
  Uses secret CREATE_BACKEND_KEY (RunPod key) and var CREATE_BEAT_RUNPOD_ENDPOINT_ID.
- GPU worker PR (OPEN): https://github.com/sqaz2/ace-step-runpod-worker/pull/1, branch `byob-lego`.
  ACE-Step 1.5 BASE model in Lego mode (track_name "vocals"), then the dry chain.
  Live test image: ghcr.io/sqaz2/ace-step-runpod-worker:base-v3.
- The last commit on each branch is "WIP: quality pass (unfinished)". It was pushed mid-work, untested.
  It adds tune.py (gentle pitch correction to key) and instrumental.py (cleaner separation for swap).

## RunPod
- BYOB endpoint: 8moi76dhfpgq1t (min 0 / max 1, 24 GB GPUs, idle timeout 30 s, FlashBoot).
- /create endpoint (LIVE, don't break): 3pppqyerkvp5pm, image ghcr.io/sqaz2/ace-step-runpod-worker:v3 (turbo).
- About $9.7 credit left. Keep tests cheap. After changing an image, cycle max workers 0 -> 1 or an
  old-image worker lingers.
- Cold start is about 5 min. A run with all lanes takes about 1.5 min on the GPU.

## Will's verdict on the first test clips (box: /workspace/vocal-ai/prompt-song/byob-test/)
- 02 "add" over the beat: "awful and unlistenable".
- 05 swap: better, but has a tinny ringing sound from stem separation; pitch off ("natural bad singing, authentic").
- 07 layers: the BEST so far, more in pitch and super natural. Not amazing singing, but okay.

## Next steps (the quality pass that was in progress)
1. Find why 07 sounded more in tune than 02/05 (seed 11, swap clip, settings) and make that the default.
2. A "Tune to key" toggle (default on, natural strength): key detection + partial pitch correction (tune.py).
3. Cleaner swap instrumental: try htdemucs_ft with shifts, or BS-Roformer / MelBand-Roformer via
   audio-separator, plus mild filtering (instrumental.py). The instrumental is what Will hears, so it matters.
4. Make v2 comparison clips (add mix, swap mix, swap instrumental, lead dry; tune on and off) and have Will listen.
5. Only after Will says go: merge #204 (Workers Builds deploys on merge to main) and merge the worker PR.

## Rules from Will
- Never print or commit secrets. Don't overwrite the v1/v3 voice model files.
- Don't remove the /voice page or the home launcher. No voice picker UI unless asked.
- Plain, human explanations; he's often on his phone.
