# Official manifest alignment

`align_official_manifest.py` reads `train_diac.jsonl` / `valid_diac.jsonl` when the folder
has them (the optional diacritize pass, `NIQQUD.md`), otherwise `train.jsonl` / `valid.jsonl`
from `build_official_manifest`, and writes `train_aligned.jsonl` and `valid_aligned.jsonl`
in the same folder. The source manifests and audio are read-only. `--input-stage built`
or `diac` pins which file is read. The aligner needs nothing but the builder's rows (a `_diac`
file is those rows plus `nikud`), so it checks their structure and does not compare file dates;
delete a stale `_diac` file if the manifest was rebuilt.

Before aligning anything it checks the first 20 rows of each input: `path`, numeric `start`
and `duration`, `words` entries with a string `word`, no `alignment_backend` /
`original_words` (the file has not been aligned already), and a `nikud` list, if present,
with one entry per word.

The aligner changes only the word times (and `start` / `duration`, and `transcript`, which
is rebuilt from `words`). `words` keeps its text, its length and its order, and every other
key -- `nikud`, `alt_words`, `speaker` -- is carried over, so this pass and the diacritize pass
give the same rows in either order. A row that would come out with a different number of
words is kept as it was, never written misaligned with its `nikud` or `alt_words`.

## Isolated runtimes

Install Viter and the released **Hebrew grapheme** model/dictionary:

```bash
uv run python hebrew_training/setup_viter_environment.py
```

This uses `uv`, Python 3.11, and a separate environment at
`/home/yoadsnapir/time-align/viter-env`. It downloads the `models-v1.0`
`hebrew_graphemes.viter` and `hebrew_graphemes.dict` assets, verifies SHA-256,
and installs `viter_requirements.txt`. This backend supports only the Hebrew
grapheme model. Missing Hebrew dictionary words are split into letters, merging
final forms and preserving supported letter-plus-geresh phones. The released
dictionary is never rewritten; extensions live in a temporary copy.

MMS continues to use `/home/yoadsnapir/time-align/mms-ctc/ctc-env`, including
for fallback subprocesses. Its existing CUDA PyTorch, torchaudio (with CTC
forced alignment), transformers, and uroman dependencies stay isolated there.

To create the MMS runtime on a new host (with `ffmpeg` available on `PATH`):

```bash
export MMS_CTC_ENV=/home/yoadsnapir/time-align/mms-ctc/ctc-env
uv venv --python 3.11 "$MMS_CTC_ENV"
uv pip install --python "$MMS_CTC_ENV/bin/python" \
  torch==2.8.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu128
uv pip install --python "$MMS_CTC_ENV/bin/python" \
  "transformers>=4.40,<5" uroman soundfile scipy numpy
```

The matching Torch/torchaudio 2.8 versions retain `torchaudio.functional.forced_align`,
which was removed in torchaudio 2.9. Model weights are downloaded on first use.
Only source code and setup instructions belong in Git; keep these environments and
downloaded assets outside the repository.

Set `VITER_ENV` / `MMS_CTC_ENV`, or use `--viter-environment` /
`--mms-environment`, to relocate these runtimes. `--environment` overrides the
selected backend's environment. `--viter-model` / `--viter-dictionary` accept
other paths to the same type of Hebrew grapheme assets; `--model` selects MMS's
model, including for fallback.

## Migration

```bash
uv run python hebrew_training/align_official_manifest.py /mlspeech/hebrew/pockettts-data/plenum --aligner viter --batch-size 16 --devices all --decode-workers 4
```

The command can also be split across Bash lines with `\` as the final character
on every continued line (no trailing spaces after it).

Viter is the default. For MMS-only processing, use `--aligner mms`.
`--splits valid` processes validation only. `--limit 16` processes at most 16
additional source rows per split for a smoke test; rerunning resumes after those
rows. Each backend keeps persistent workers and processes bounded batches,
preserving source order. Default split order is validation, then training.

Both backends decode 0.5 seconds before and 1 second after the source span, then
rebase output `start`, `duration`, and relative `words` to the first/last aligned
word. Every output row includes the exact source `original_start`,
`original_duration`, and `original_words` snapshot. Original word offsets remain
relative to `original_start`; new word offsets are relative to the new `start`.

Viter uses Vulkan GPU acceleration. On this Linux NVIDIA host, `--devices all`
starts one worker per GPU by default; `--devices 0,2` selects specific GPUs.
`--workers-per-gpu 4` runs four persistent workers on each selected GPU, overlapping
independent audio decoding, CPU alignment, GPU scoring, and MMS fallback stages.
Each worker has its own model and lazy MMS fallback process. Queues remain bounded
and output remains in source order. Batch size and worker counts can be changed
when resuming; they do not invalidate the existing checkpoint.
The Mesa Vulkan device-select layer (`libvulkan1`, `mesa-vulkan-drivers`) and an
NVIDIA Vulkan-capable driver are needed for per-worker device selection.
Workers select adapters by PCI bus ID, respecting `CUDA_VISIBLE_DEVICES`.
`--viter-threads` defaults to four compute threads per worker. `--devices cpu`
runs a single CPU worker. Speaker adaptation pools only same-speaker rows within
a batch; rows without a speaker use the recording path as their speaker key.

## Fallback and comparison

- Any digit in the transcript or word text bypasses Viter and goes to MMS.
- Unsupported dictionary words and failed Viter alignments also go to MMS.
- If MMS fallback fails, the output row retains its original `start`, `duration`,
  `words`, and transcript, plus the original snapshot. It is never dropped.
- MMS rejects rows containing words it cannot spell rather than silently
  dropping those words (notably standalone digit tokens).
- `alignment_backend` is `viter`, `mms`, or `original`.
- Fallback rows include `alignment_fallback_reason`; retained-original rows also
  include `alignment_error` and an entry in `*_aligned.failed.jsonl`.
- Explicit MMS-only mode retains its previous error handling: unsuccessful rows
  are recorded in the failure file rather than written to the output.

Progress is saved in `*_aligned.progress.json`; repeat the same command to
resume. Backend/model/beam settings must match the saved progress to prevent
mixing configurations. Completed outputs are skipped. Use `--overwrite` to
restart an output, including when replacing an earlier MMS migration with Viter.

## Verified locally

An eight-A40 run over 1,188 validation rows produced 1,180 Viter rows, four MMS
fallback rows, and four retained-original rows. No source row was lost; source
manifest SHA-256, preserved timing snapshots, word labels, padding bounds, and
rebased spans passed verification. A separate real-audio fixture confirmed digit
and missing-audio failures retain the original row. MMS-only smoke testing and
the focused migration tests also pass.

### Throughput tuning on eight A40 GPUs

The same 4,096 contiguous training rows were benchmarked with four Viter compute
threads and four decode workers per alignment worker. These rates include worker
startup and lazy MMS initialization, but exclude dictionary preparation:

| Batch size | Workers per GPU | Rows/s |
|---:|---:|---:|
| 16 | 1 | 33.41 |
| 16 | 2 | 40.26 |
| 16 | 4 | 55.92 |
| 16 | 8 | 55.80 |
| 64 | 2 | 73.14 |
| 64 | 4 | 85.27 |
| 128 | 2 | 79.10 |
| 128 | 4 | 85.52 |

Recommended setup for this host (128 logical CPUs):

```bash
uv run python hebrew_training/align_official_manifest.py /mlspeech/hebrew/pockettts-data/plenum --aligner viter --batch-size 64 --workers-per-gpu 4 --devices all --decode-workers 4 --viter-threads 4
```

This runs 32 alignment workers. Batch 128 provided no meaningful gain over 64;
eight workers per GPU at batch 16 also did not help. Peak GPU memory across the
tested configurations stayed below 7.1 GB per GPU. All runs preserved the same
4,096 rows, with five retaining original timings after failed MMS fallback.
Long-run speed depends on clip durations, fallback frequency, and storage behavior.

Larger Viter batches pool more same-speaker clips for speaker adaptation, so
boundaries can differ from smaller batches. The timing schema and source snapshots
are preserved. Changing only worker count retains the same source batch grouping.

To resume an existing migration, stop the current process with Ctrl-C, wait for
its workers to exit, then use the recommended command **without `--overwrite`**.
The last durable checkpoint is reused; rows written after it are safely replayed.
