---
title: Rating buttons follow physical keyboard order
author: Yoad
date: 2026-11-13
branch: yoad/semantic-fusion
cost: { minutes: 60, tokens: 378446 }
id: l-76259e38
scope: repo
status: active
src: agent
touches:
  - hebrew_training/audio_v2_tagger/index.html
  - hebrew_training/audio_v2_tagger/pyproject.toml
  - hebrew_training/audio_v2_tagger/uv.lock
  - hebrew_training/audio_v2_tagger/model.py
  - AGENTS.md
  - hebrew_training/AUDIO_V2.md
  - hebrew_training/audio_v2_rows.py
  - hebrew_training/audio_v2_table.py
  - hebrew_training/audio_v2_tagger/__init__.py
  - hebrew_training/audio_v2_tagger/server.py
---

## Decided
- Keep the audio-v2 tagger page RTL, but lay out the three rating buttons from left to right as 1, 2, 3. This matches the physical QWERTY keyboard, making rapid audio grading less error-prone.
## Tried and ruled out
- Inheriting RTL on the rating grid: it puts 1 at the right edge, opposite the user's keyboard.
## In the code:
- hebrew_training/audio_v2_tagger/index.html: `.buttons` uses `direction:ltr` while the document and transcript remain RTL.
