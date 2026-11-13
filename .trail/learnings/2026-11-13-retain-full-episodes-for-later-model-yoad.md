---
title: Retain full episodes for later model cuts
author: Yoad
date: 2026-11-13
branch: yoad/semantic-fusion
id: l-a9463ddf
scope: repo
status: active
src: agent
touches:
  - hebrew_training/AUDIO_V2.md
  - hebrew_training/audio_v2_tagger/build_selected.py
  - hebrew_training/build_official_manifest.py
---

## Decided
- For this model-selected audio-v2 cut, fetch only episodes that contain at least one selected row, but keep each complete source recording and transcoded WAV for later cuts and new fits. The user chose this explicitly after seeing that 3,611,043 passing rows span 32,564 episodes and need about 1,092 GiB of source downloads: flexibility and source diversity justify the extra storage.
- Keep the row-level manifest filter: retaining full audio for future cuts does not mean adding rejected rows to the current training manifest.
## Tried and ruled out
- Streaming and storing just selected clips: would save source transfer but produce millions of separate clips, and the next change to the filtering model would require new source fetches.
- Capping the dataset to a small episode subset: would omit most passing rows and source diversity, contrary to the user's choice.
## In the code:
- hebrew_training/audio_v2_tagger/build_selected.py fetches selected episodes, then builds filtered manifests; hebrew_training/build_official_manifest.py `--row-selection` limits training rows.
- hebrew_training/AUDIO_V2.md records the volume and reuse contract.
