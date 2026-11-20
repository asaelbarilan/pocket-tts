# Numeric-speech dataset

Teaches the model to say digits. Each numeric pair has the same sentence written two ways
(`digits`: "עד 143 משתתפים", `cardinal`: "עד מאה ארבעים ושלושה משתתפים"). Pocket-TTS speaks the
cardinal form well, so we synthesize that, and train with the digits form as the text.

```bash
uv run python -m hebrew_training.build_numeric_speech_manifest \
  --pairs data_local/numeric_pairs --prompts data_local/voice_prompts_out \
  --out-dir data_local/numeric_speech --config <hebrew.yaml> --checkpoint <checkpoint.pt> \
  [--device cuda] [--limit 20]     # --limit: a listen-first smoke run; --dry-run: show the assignment
```

Inputs: every `*.jsonl` in `--pairs` (any split, each `id` once) and a voice prompt library from
`python -m hebrew_training.voice_prompts extract` (README there). Use a Hebrew checkpoint:
`pocket_tts/config/hebrew.yaml` alone points at the released English weights.

## Output (`--out-dir`)

`audio/<id[:2]>/<id>.wav` is **prompt + generated speech** in one file. `train.jsonl` /
`valid.jsonl` rows:

```json
{"path": "...", "start": 0.0, "duration": 8.1, "prompt_end": 3.785,
 "transcript": "<digits form>", "id", "split", "archetype_id", "cardinal",
 "prompt_id", "prompt_text", "speaker", "sample_rate"}
```

There are no `words`, `alt_words` or `nikud`: nothing is aligned. `failures.jsonl` lists pairs
that failed (rerun to retry them).

## The cut sentinel

`prompt_end` is the exact length of the prepended prompt, in seconds. `training/dataloader.py`
cuts there and nowhere else: audio before = voice prompt, audio after = target (up to
`max_duration_sec`), text = `transcript` untouched. No cut is searched, nothing is drawn, so the
sample is the same every epoch. `MIN_CUT_SEC` does not apply. A bad `prompt_end` (outside the
audio) or a prompt longer than `max_voice_prompt_sec` raises (the sample is skipped with a
warning) instead of being truncated; the builder drops library prompts over `--max-prompt-sec`
(5.0, as in the configs) so this does not happen. Rows without `prompt_end` behave as before.

The dataset viewer shows it: the prompt part of the waveform is orange, a magenta line marks the
cut, and *Prompt* / *Target* buttons play each side (`prompt_end` is the viewer's source, never
re-derived).

## Speakers

The library's own `split` is ignored. Its speakers (`speaker`, else `user_id`, else the source
recording) are split here, by hash, into train and valid (`--valid-speaker-fraction`, default 0.1,
at least one), so no speaker is on both sides. Pair splits in `--valid-splits` (default dev,test)
go to valid speakers and `valid.jsonl`. Within a side each pair gets a different speaker: speakers
are dealt in a shuffled cycle, every speaker once before any repeats, and a speaker's prompts
rotate. The result depends only on `--seed`, the pair ids and the library, so every run and shard
agrees. Check the pool has enough speakers: with S train speakers, a pair set of N reuses each
about N/S times.

## Large runs

Rows are appended one by one and a rerun skips what exists (a torn last line is ignored). To use
several GPUs, run one process per shard, then merge:

```bash
for i in 0 1 2 3; do
  CUDA_VISIBLE_DEVICES=$i uv run python -m hebrew_training.build_numeric_speech_manifest \
    ... --device cuda --num-shards 4 --shard-index $i &      # writes train.shard-$i.jsonl
done; wait
uv run python -m hebrew_training.build_numeric_speech_manifest --out-dir <dir> --merge
```

Other details: the generated tail is cut to 0.2 s of silence (the loader's `TRAIL_SEC`) so EOS has
a consistent target; the spoken text is the pair's `cardinal`, with the model's own `fold_text`.
Synthetic speech should stay a small share of a curriculum phase; judge it by Hebrew CER on real
digit sentences, not loss.
