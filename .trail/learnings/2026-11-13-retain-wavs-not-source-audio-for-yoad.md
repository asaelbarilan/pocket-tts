---
title: "Retain WAVs, not source audio, for later cuts"
author: Yoad
date: 2026-11-13
branch: yoad/semantic-fusion
id: l-dd65960e
scope: repo
status: active
replaces: l-a9463ddf
src: agent
touches:
  - hebrew_training/AUDIO_V2.md
  - hebrew_training/audio_v2_tagger/build_selected.py
  - hebrew_training/build_official_manifest.py
  - tests/test_build_official_manifest.py
---

## Decided
- The user clarified that selected episodes may be fetched whole but original MP3/M4A files must not be retained. Keep the complete transcoded WAV per fetched recording, plus transcript and metadata, so future model cuts can be rebuilt from WAVs. Verify all selected episodes have finished WAVs and the manifest builder succeeded before deleting any source audio.
- Preserve the row-level manifest selection: full WAV reuse does not let unselected rows into training.
## Tried and ruled out
- Keeping source audio alongside WAVs for flexibility: unnecessary duplication; the user explicitly wants only the WAVs retained.
- Deleting each source on download or before all WAVs are verified: a failed transcode would permanently lose that recording and make recovery require another remote fetch.
## In the code:
- hebrew_training/audio_v2_tagger/build_selected.py: `remove_sources_after_wavs` runs only after the successful manifest build and validates all WAVs first.
- hebrew_training/build_official_manifest.py: fallback to saved `--audio-dir` WAV when a corpus episode has no MP3/M4A.
- hebrew_training/AUDIO_V2.md describes the reusable WAV layout.
