# Voice prompt extraction

Builds a large, varied pool of 3-5 s voice prompts for the synthetic-dataset work, cut only where
no word is clipped. Reads the official manifests (`train_aligned.jsonl` / `valid_aligned.jsonl`,
else `_diac`, else the built ones; `manifest_files.py`). Nothing is written to the datasets.

```bash
# 1. extract (random, reproducible)
uv run python -m hebrew_training.voice_prompts extract \
  plenum=/mlspeech/hebrew/pockettts-data/plenum \
  crowd=/mlspeech/hebrew/pockettts-data/crowd \
  --output /mlspeech/hebrew/voice-prompts --per-speaker 10 --max-prompts 200000

# 2. later, optional: embed, look at the similarity report, then thin the pool
uv run python -m hebrew_training.voice_prompts dedupe /mlspeech/hebrew/voice-prompts
uv run python -m hebrew_training.voice_prompts dedupe /mlspeech/hebrew/voice-prompts --threshold 0.9
```

## How a cut is chosen

A *seam* is a gap of at least `--min-gap-ms` (300) between two consecutive words, and the cut
goes in its middle, so each side keeps at least 150 ms of margin against word-timing error. A
prompt runs from one seam to another that is `--min-duration`..`--max-duration` away. The seams
come from the row's word gaps and, unless `--no-edge-cuts`, from `lead_gap` / `tail_gap`, so a
prompt can begin at a row's first word and end at its last. Windows also need `--min-words`, no
inner pause over `--max-inner-gap-ms` and word time of `--min-speech-ratio` of the clip.
Rows without measured word times (`alignment_backend: original`, `timing: estimated`) are
skipped. Use aligned manifests: Whisper's own word times are looser.

`--per-row` windows per row, `--per-speaker` per speaker (else `user_id`, else the recording;
the Knesset corpora only have the recording), `--max-prompts` overall, all by seeded random
priority.

## Output

`<output>/prompts/000000.wav` (mono 16-bit, `--sample-rate` 24000) and `<output>/manifest.jsonl`.
Each manifest row has `id`, `audio`, `dataset`, `split`, `stage`, `manifest` + `line`,
`source_path`, `source_start` / `source_end` (seconds in the source WAV), `duration`,
`sample_rate`, `speaker` / `user_id` / `entry_id` when present, `text`, `num_words`,
`gap_before_ms` / `gap_after_ms` (clip edge to first / last word) and `seam_before_ms` /
`seam_after_ms` (the whole gap the cut sits in).

## Dedupe

Prompts are embedded the way Pocket-TTS reads a voice prompt: Mimi's encoder output
(`mimi.encode_to_latent`), mean and standard deviation over time, so the vector carries timbre,
channel, loudness, pitch range and delivery, not only speaker identity. `--space conditioning`
uses the sequence after `speaker_proj_weight` instead (what the transformer is fed; depends on
`--language` / `--config` / `--checkpoint`). Vectors are standardized per channel across the
folder and compared by cosine. There is no standard threshold for this space, so the first run
(neither flag) prints the nearest-neighbour similarity quantiles and stops; then pass
`--threshold T` (greedy, in manifest order) or `--keep N` (farthest-point: N prompts spread as
widely as possible). Check a few pairs near the threshold by ear before trusting it.

Writes `embeddings.npz` (cached), `manifest.dedup.jsonl` and `duplicates.jsonl`; `--prune` also
deletes the removed WAVs and makes `manifest.dedup.jsonl` the new `manifest.jsonl` (the old one
is kept as `manifest.before_dedup.jsonl`). Ids are never renumbered.

## UI

```bash
uv run python -m hebrew_training.voice_prompts.ui   # http://127.0.0.1:8765
```

FastAPI + Alpine.js from a CDN (needs internet in the browser), no build step; code in `ui/`.
It binds to localhost because the paths you enter are paths on the machine running it.

- **Datasets & target**: dataset folders (`[NAME=]DIR`, checked as you add them) and the target
  folder. Both are kept in the browser's local storage.
- **Extract**: all the `extract` options. A run *adds* `Samples to add this run` prompts to the
  target (`--append`): numbering continues, spans already extracted are avoided, and the
  per-speaker cap counts what is already there. The seed is new each run unless you set one.
  "Start fresh" deletes the target's prompts first (it asks to confirm). One run at a time, with
  progress and Cancel; a cancelled run keeps what it wrote.
- **Preview**: random prompts to listen to, with their manifest row.
- **Similarity**: two random (or chosen) prompts, their similarity in the `dedupe` space, and where
  it falls among random pairs of the pool. The first comparison loads the model and embeds a
  reference sample of 200 prompts (or uses `embeddings.npz` after a `dedupe` run).

CLI `extract` has the same `--append`.
