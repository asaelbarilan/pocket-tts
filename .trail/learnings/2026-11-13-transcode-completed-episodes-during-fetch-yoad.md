---
title: Transcode completed episodes during fetch
author: Yoad
date: 2026-11-13
branch: yoad/semantic-fusion
id: l-e80aa266
scope: repo
status: active
src: agent
touches:
  - hebrew_training/AUDIO_V2.md
  - hebrew_training/audio_v2_tagger/build_selected.py
  - tests/test_build_official_manifest.py
---

## Decided
- Start a bounded pool of ffmpeg transcodes as soon as each audio-v2 episode has its completed metadata marker and full source audio, in parallel with the 32-worker Hugging Face fetch. The remote machine has spare CPU and storage bandwidth; deferring all WAV conversion until the 32,564-episode fetch finishes unnecessarily extends the dataset ETA.
- Keep originals while the fetch is active so fetch resume does not redownload them. Wait for all converter workers before the row-filtered manifest build; only remove MP3/M4A after the whole manifest and every WAV pass validation. WAV creation remains atomic and reused on restart.
## Tried and ruled out
- Running the manifest builder on the partially downloaded corpus: its initial recording list and train/valid split assume a complete corpus; it could silently omit episodes. Pipeline ffmpeg only, then build the manifest once all episodes are present.
- Deleting sources immediately after per-episode WAV conversion: current fetch logic uses metadata.json to skip downloads and may need the source until the whole run is verified.
## In the code:
- hebrew_training/audio_v2_tagger/transcode_selected.py converts only complete episodes and is safe to resume.
- hebrew_training/audio_v2_tagger/build_selected.py runs the transcoder concurrently with the fetch, waits for it before manifest generation and preserves the WAV-only final cleanup.
- hebrew_training/AUDIO_V2.md documents concurrency and retention.
