---
title: Parallelize audio-v2 fetch with rate-limit backoff
author: Yoad
date: 2026-11-13
branch: yoad/semantic-fusion
id: l-daf4c3bb
scope: repo
status: active
src: agent
touches:
  - hebrew_training/AUDIO_V2.md
  - hebrew_training/audio_v2_tagger/build_selected.py
  - hebrew_training/fetch_audio_v2.py
  - tests/test_fetch_audio_v2.py
---

## Decided
- For the frozen model-selected audio-v2 export, use 32 parallel episode downloads by default rather than 8 and inspect real 429/failed counts while running. The user explicitly wants as much parallelism as Hugging Face permits; the host has ample CPU and the 8-worker run showed no rate limits.
- Each source download retries transient errors and follows Hugging Face Retry-After on 429/503 rather than failing until the end of a multi-day full pass. Resume the detached job over completed episode folders when changing concurrency.
## Tried and ruled out
- Keeping the original 8 workers despite zero observed rate limits: leaves available parallel bandwidth unused.
- Blind retries without respecting Retry-After: synchronized requests can worsen upstream throttling.
## In the code:
- hebrew_training/audio_v2_tagger/build_selected.py: `--fetch-workers` default 32.
- hebrew_training/fetch_audio_v2.py: `_download_audio` retries/backoff; metadata.json still marks a finished episode.
- hebrew_training/AUDIO_V2.md: operator override for persistent throttling.
