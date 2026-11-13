---
title: RTL tagger words need break opportunities
author: Yoad
date: 2026-11-13
branch: yoad/semantic-fusion
cost: { minutes: 58, tokens: 378446 }
id: l-ddf93757
scope: repo
status: active
src: agent
touches:
  - hebrew_training/audio_v2_tagger/pyproject.toml
  - hebrew_training/audio_v2_tagger/uv.lock
  - hebrew_training/audio_v2_tagger/model.py
  - AGENTS.md
  - hebrew_training/AUDIO_V2.md
  - hebrew_training/audio_v2_rows.py
  - hebrew_training/audio_v2_table.py
  - hebrew_training/audio_v2_tagger/__init__.py
  - hebrew_training/audio_v2_tagger/index.html
  - hebrew_training/audio_v2_tagger/server.py
---

## Decided
- Render each transcript word as an inline-block with max-width and overflow-wrap, and allow the card to shrink in the grid. Adjacent Alpine x-for spans have no whitespace nodes, so margin alone cannot give the browser a reliable place to wrap lines.
## Tried and ruled out
- Spacing words with margin-inline alone: a long Hebrew transcript remained an unbreakable run and overflowed into the adjacent statistics column.
## In the code:
- hebrew_training/audio_v2_tagger/index.html: .text, .w, and .card CSS; preserve per-word highlighting and RTL direction.
