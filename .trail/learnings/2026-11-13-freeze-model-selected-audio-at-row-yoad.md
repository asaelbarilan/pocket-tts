---
title: Freeze model-selected audio at row granularity
author: Yoad
date: 2026-11-13
branch: yoad/semantic-fusion
id: l-90a8d444
scope: repo
status: active
src: agent
touches:
  - hebrew_training/AUDIO_V2.md
  - hebrew_training/build_official_manifest.py
  - tests/test_audio_v2_tagger.py
  - tests/test_build_official_manifest.py
---

## Decided
- A tagger rating is for a merged utterance row, not for a recording or individual Whisper segment. For the model's rating >= 3 rule, freeze the current labels/settings/model and score all rows in the existing table. Download episodes with at least one passing row, then filter the manifest by exact recording/start/end from the frozen row selection. Keep the original label log and settings for future fits.
- Make the download/build runner resumable and detached because a full cut can take days and SSH/1Password sessions drop.
## Tried and ruled out
- `fetch_audio_v2 select`'s Whisper thresholds: they cannot express the learned model (especially when source and word features contribute), and selecting episodes alone admits rejected rows into the manifest.
- Assuming the tagger's hour estimate is the exact selected hours: it is computed over a fixed hour-weighted sample, whereas the export scores the full table.
## In the code:
- hebrew_training/audio_v2_tagger/export_selection.py: frozen fit and selected rows.
- hebrew_training/build_official_manifest.py: `--row-selection` matching the builder's own row boundaries.
- hebrew_training/audio_v2_tagger/build_selected.py: resumable download/build orchestration.
