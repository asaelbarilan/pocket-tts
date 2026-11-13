# ivrit.ai audio-v2 (weakly labeled, >20k h)

`ivrit-ai/audio-v2` (1.1 TiB audio) + `ivrit-ai/audio-v2-transcripts` (8.1 GiB of
`full_transcript.json.gz` that matter). Machine transcripts (ivrit.ai whisper-large-v3-turbo,
stable-ts aligned, VAD), no human check, **speakers unknown** and often mixed. Gated, one
checkbox, auto-approved. License: ivrit.ai v1.

## Structure (measured)

- transcripts: `source/episode/{full_transcript.json.gz, speech_probs.frame.gz}` (always exactly
  3 levels); audio: `source/episode.{m4a,mp3}` (flat, 2 levels). Join key `(source, episode)`.
- 34,580 episodes with both, 78 sources; 276 audio files have no transcript, 6 transcript folders
  lack `full_transcript.json.gz`. Ignore `.done` markers, `speech_probs.frame.gz` and the YouTube
  caption files (`.vtt`, `.xml`, `.json3`, `.srv*`, `.ttml`).
- audio is 48 kHz stereo (Opus in `.m4a`, MP3 in `.mp3`), ~100 kbps; the builder transcodes to
  24 kHz mono WAV.
- transcript segments carry `avg_logprob`, `compression_ratio`, `no_speech_prob` and per-word
  `start/end/probability`; the builder's existing schema needs no conversion.
- There is no quality manifest (unlike knesset-plenums), hence the survey stage.

## Workflow (`hebrew_training/fetch_audio_v2.py`)

```
python -m hebrew_training.fetch_audio_v2 --work data/audio_v2_work list
python -m hebrew_training.fetch_audio_v2 --work data/audio_v2_work survey      # ~8 GiB, resumable
python -m hebrew_training.fetch_audio_v2 --work data/audio_v2_work report --min-avg-logprob -0.5 --max-compression 2.4
python -m hebrew_training.fetch_audio_v2 --work data/audio_v2_work select --min-avg-logprob -0.5 --max-compression 2.4 --hours 3000
python -m hebrew_training.fetch_audio_v2 --work data/audio_v2_work fetch --out data/audio_v2 --max-gb 400
python -m hebrew_training.build_official_manifest --corpus data/audio_v2 --out-dir ... \
    --audio-glob audio.m4a,audio.mp3 --min-avg-logprob -0.5 --max-compression-ratio 2.4
```

`survey --limit-episodes N` takes the first N of a fixed random order, so a small survey is an
unbiased sample (`report` says how much of the corpus it covers). `select` is nested: raising
`--hours` later only adds episodes. `fetch` refuses downloads above `--max-gb`.

The builder's `--min-avg-logprob`, `--max-compression-ratio` and `--max-no-speech-prob` are off by
default and ignore segments that lack the field, so other corpora build as before. Use the
same thresholds for `select` and the builder, or the hours will not match.

## Speaker key

`metadata.json` `user_id` defaults to the source (show), so a host does not land in both splits.
It is a proxy, not a speaker: `speaker_known` is `false`. `--speaker-key recording` gives a
recording-disjoint split instead. A later speaker-classification pass can rewrite `user_id`
in `metadata.json` with nothing else to redo.

## First sample (600 episodes, 1.7% of the corpus, 311 h of speech)

- `avg_logprob` is the discriminating metric: p5/p50/p95 = -0.43 / -0.17 / -0.07; >= -0.5 keeps
  96.9% of speech, >= -0.3 keeps 89.7%, >= -0.2 keeps 66%.
- `compression_ratio` is tight (p5/p50/p95 = 1.75 / 2.15 / 2.35); <= 2.4 keeps 96%, <= 2.2 keeps 63%.
- `no_speech_prob` is ~0.03 everywhere (stable-ts/VAD already removed those segments): useless.
- median word probability >= 0.8 keeps 95.7%.
- `-0.5` / `2.4` together keep 92.9% of speech; 366 of 600 episodes lose < 5%.

Re-run `report` on the full survey before choosing thresholds; these numbers are a sample.

## Choosing the cut by ear: the tagger

Whisper's confidence numbers say little about what *sounds* usable, so a small browser tool lets
you rate rows and estimates the data cut from the ratings.

1. **Prep (once, main env):** `python -m hebrew_training.audio_v2_rows --work W --workers 8`
   re-reads the transcripts (~8 GB through the Hub, nothing kept) and writes one feature row per
   candidate training row to `W/rows/` (the builder's own merge; Whisper, word-probability,
   word-duration and gap features). Resumable.
2. **Run (own env, no torch):** from the repo root,

   ```
   uv run --project hebrew_training/audio_v2_tagger \
       python -m hebrew_training.audio_v2_tagger.server --work W --port 8765
   ssh -L 8765:localhost:8765 technion-compute-113      # then open http://localhost:8765
   ```

   The tagger has its own `pyproject.toml`/`uv.lock` (fastapi, xgboost, pandas): it is used rarely
   and shares no dependencies with the model's environment. Tests:
   `PYTHONPATH=. uv run --project hebrew_training/audio_v2_tagger --group dev pytest tests/test_audio_v2_tagger.py`.
3. **Rate:** keys 1 / 2 / 3 (clearly not / in a bunch / clearly yes), S skip, Backspace undo.
   The first 7 prefetched clips (`--ahead`) are random and drawn in proportion to duration.
   Once the model beats the prior, 7 additional clips (`--active-ahead`) target uncertain
   predictions and feature ranges with high held-out error. The latter are refreshed after every
   fit; only the random stream provides an unbiased direct hours estimate. Clips are encoded on
   demand by seeking inside the remote file (mp3: download, cut, delete). The `--cache` is local
   disk and preloading pauses above `--max-disk-fraction` 0.8. Older clips are evicted, but undo
   reconstructs the actual previous clip if needed. Ratings append to `W/tagger/labels.jsonl`.
4. **Read:** estimated kept hours and WAV size for "rating >= 3" and ">= 2", from the labels alone
    (Wilson interval on *random* ratings) and from an xgboost model (bootstrap interval), plus
    cross-validated log loss against the class prior. Feature-range diagnostics rank where
    out-of-fold prediction error affects the most population hours. CV uses the random ratings
    when there are enough; until then it is provisional. Targeted ratings improve the model but
    are not included in the direct estimate; model bootstrap intervals are not guaranteed to
    cover the truth under adaptive selection.

The tagger shows a row's words using `audio_v2_table.row_segments`, a copy of the builder's segment
rule that a test pins to `clean_segments`.

## Build the model-selected dataset

Stop the tagger after labeling so the label log is stable. The fitted rule is
`P(rating >= 3) >= tagger/settings.json:cutoff` (not the old Whisper thresholds).
`audio_v2_tagger.build_selected` freezes the label log, settings, seed, feature schema and
XGBoost model in a new snapshot folder; scores every row in `W/rows`; fetches only episodes
containing selected rows; then builds manifests with `--row-selection` so *unselected rows
in those episodes do not enter training*. Preserve `W/tagger/labels.jsonl` and
`W/tagger/settings.json` for future labeling/refitting. Use a different snapshot and corpus
directory when making a new cut. The download can be hundreds of GiB and the WAVs larger:
check capacity before starting, and run it detached on the data host (never in SSH's foreground).

```
python -u -m hebrew_training.audio_v2_tagger.build_selected \
  --work W --snapshot W/model3-YYYYMMDD \
  --corpus DATA/model3-corpus --manifest DATA/model3-manifest \
  --spool-dir /tmp
```

This command is resumable: rerun the same command if it exits after the fit or download.
The immutable snapshot contains `fit.json`, `labels.jsonl`, `settings.json`, `model.ubj`,
`selection.json` and `selected_rows.npz`; `W/selection.json` drives the fetcher. Look at
`W/model3-YYYYMMDD.status`, the detached log and `W/model3-YYYYMMDD.done.json` to track it.
