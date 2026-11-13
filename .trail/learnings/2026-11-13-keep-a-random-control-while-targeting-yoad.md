---
title: Keep a random control while targeting uncertain clips
author: Yoad
date: 2026-11-13
branch: yoad/semantic-fusion
id: l-d2b41df6
scope: repo
status: active
src: agent
touches:
  - hebrew_training/AUDIO_V2.md
  - hebrew_training/audio_v2_tagger/index.html
  - hebrew_training/audio_v2_tagger/model.py
  - hebrew_training/audio_v2_tagger/server.py
  - tests/test_audio_v2_tagger.py
---

## Decided
- Keep seven duration-weighted random-control clips in the audio-v2 tagger queue and add seven model-targeted clips after held-out log loss beats the prior. Store each rating's selection mode and compute direct Wilson hours only from controls. Rank error-prone feature ranges by out-of-fold error times population-hour share; refresh unserved targeted clips after each fit.
- Rebuild a previous clip before removing its rating on undo, so older than 40 clips and sessions after restart can still move backward accurately.
## Tried and ruled out
- Treating all ratings as a uniform hours sample after active selection: targeted uncertainty samples bias the direct hours estimate and invalidate a Wilson interval over all labels.
- Relying on the 40-clip cache for undo: eviction or restart means the true previous clip is no longer in memory.
## In the code:
- hebrew_training/audio_v2_tagger/server.py: `Tagger.undo`, `worker`, `_fit`, `label` and queue selection modes.
- hebrew_training/audio_v2_tagger/model.py: `analyze`, `cross_validate`, `confusion_ranges`, `active_scores`.
- hebrew_training/audio_v2_tagger/index.html: undo navigation and random vs targeted explanations.
