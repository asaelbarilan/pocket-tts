# Bounded Hebrew / English / niqqud tokenizer experiment

This is tokenizer training only. It does not change or train Pocket TTS acoustic weights.
All generated corpora, model weights, caches and results live under the already gitignored
`artifacts/tokenizer-he-en-niqqud/` directory.

## Reproduce on this server

The local config is `hebrew_training/configs/tokenizer_he_en_niqqud.yaml`. It uses the committee
manifest plus the separate FineWeb `he/` and `en/` directories, never their mixed parent.
Download the reviewed Dicta snapshot once:

```bash
uv run --no-sync python - <<'PY'
from huggingface_hub import snapshot_download
snapshot_download(
    'dicta-il/dictabert-large-char-menaked',
    revision='4dfc20b94565d7868e1c8c00bf039d38c071a062',
    local_dir='artifacts/tokenizer-he-en-niqqud/dicta-model',
)
PY

OMP_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false uv run --no-sync python -u \
    -m hebrew_training.tokenizer_experiment \
    --config hebrew_training/configs/tokenizer_he_en_niqqud.yaml

# If 32M is still improving, extend to 64M. Old vocalization is reused, not regenerated.
OMP_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false uv run --no-sync python -u \
    -m hebrew_training.tokenizer_experiment \
    --config hebrew_training/configs/tokenizer_he_en_niqqud.yaml --extend-to 64
```

The Dicta snapshot is public (CC BY 4.0), about 1.22 GB, and executes its custom model code.
The local environment needs Transformers as well as SentencePiece, Torch and PyYAML.
The `training` extra alone does not install Transformers; it is currently in the `dev` group.

## Sampling and splitting

The builder's optional `sampling.mode: bounded` draws deterministic random byte windows from
uncompressed `.txt` or `.jsonl` files. It never scans the full sources. Windows are byte-weighted,
not a uniform sample of records. Each document contributes at most approximately 4,000 cleaned
characters. Cleaned lines are deduplicated. A hard I/O budget fails the run rather than
silently scanning indefinitely or accepting an undersized corpus.

JSONL groups use `speaker`, `user_id` or `path` (the committee's `speaker` is a recording label,
not a verified human speaker identity). Plain-text records use a content hash. Stable hash
splits keep groups in either training or held-out evaluation, not both. Exact repeated lines
are removed across the two splits, and evaluation additionally excludes unmarked duplicates
found in any training source. This is not semantic near-duplicate removal.
Plain-text line boundaries are treated as records: without original web-page IDs we cannot
guarantee true page-level separation when an upstream exporter flattened pages into lines.

Sampling caches record source paths/sizes/mtimes, seed, charset, byte reads and split counts.
Changing the sampling setup requires a new cache directory. Increasing a requested budget
rebuilds the sample deterministically; use a new vocalization cache if its input changes.

## Vocalization and mixtures

The mix is **20% committee / 40% general Hebrew / 20% vocalized Hebrew / 20% English**, measured
in Unicode characters, excluding line separators. `target_mb` is historical naming: it means
millions of characters, not UTF-8 megabytes. Complete-line clipping can overshoot slightly.

`hebrew_nikud.diacritize_sources: [manifest, hebrew]` collects extra input beyond each plain
slice, balances the two domains by input characters, and vocalizes inside the builder.
The actual processed input prefix is recorded in `hebrew_nikud.progress.json`. Extensions use
`vocalization_seed_dir` to validate and reuse the previous pool's processed prefix. Only the
remaining vocalized character budget is generated. The pool summary separates reused output
characters from new candidate input characters per source.
`preserve_matres: true` asks Dicta to keep matres lectionis rather than delete them; the
experiment verifies held-out output preserves all non-niqqud content. This intentionally
retains full spelling, rather than changing user text to conventional defective spelling.

One GPU runs Dicta in bfloat16, batch 32. Tokenizer fitting is CPU unigram SentencePiece with
8,000 pieces, identity normalization, full observed character coverage and byte fallback.
The config's `required_chars` reserves all 18 niqqud marks, including rare holam-haser, meteg
and rafe even if they never appear in a bounded sample; no synthetic sentences are added.
Every input goes through the charset filter. Vocalization caches validate input/config hashes
and resume by input cursor even when a generated line was rejected by the filter.
An offset adapter replaces scalar CPU-tensor indexing in Dicta's output reconstruction with
Python integers. On a real 32-line Hebrew batch on an A40, it reduced batch time from 0.258 s
to 0.119 s with identical output (three repeats after warmup); this is a microbenchmark, not
a whole-corpus throughput guarantee.
The adapter also suppresses duplicate one-character offsets from tokenizer normalization
(notably `…` expanding to three tokens), which otherwise makes Dicta repeat punctuation.
Legacy caches are migrated only when the sole content difference is this proven ellipsis bug;
other unexpected content changes fail closed.

## Size selection and outputs

The maximum corpus is built/vocalized once. Nested 4/8/16/32-million-character corpora reuse its
per-source slices and keep the requested mix; no repeated vocalization for each tokenizer.
Held-out committee, general Hebrew, English, vocalized committee and vocalized general Hebrew
are evaluated separately. No code-switch benchmark is invented when none is supplied.

The operational selection rule is the smallest tested size with tokens/character within 1%
of the largest on **every** evaluation set, and no unknown tokens or round-trip failures.
This is a size/quality tradeoff, not proof of a universal optimum or of speech quality.
If the largest size wins, extend the experiment rather than claiming saturation.

Outputs:

- `REPORT.md`, `results.json`: comparison, timings, selection and effective config;
- `selected/tokenizer.model`, `selected/tokenizer.vocab`: selected trained tokenizer;
- `selected/corpus.txt`: its exact training input;
- `evaluation32m/`, `evaluation64m/`: held-out sets and overlap audits for each maximum pool;
- `cache32/samples/*/sampling.json`: bounded read/split/filter provenance;
- `pool32/source_slices/`: reusable maximum source slices.

`--reuse-pool` skips maximum-pool construction when it is already complete. Per-size training
is reused only if the input fingerprint and vocabulary size match. Keep old directories intact
and use new versioned output/cache paths for source, model or preprocessing changes.
For the latest extended pool, use its effective `artifacts/tokenizer-he-en-niqqud/experiment_config.yaml`
with `--reuse-pool`, rather than the original 32M config. All sizes are rescored against the
same held-out sets for the final pool; exact duplicates discovered in additional training text
are excluded from evaluation. Old trained models are reused when their corpus fingerprints match.

## Measured result (2026-10-16)

Tested 4, 8, 16, 32, 40, 48 and 64 million characters, all with the same 8,000-piece vocabulary
and all 18 niqqud marks reserved. **48M is the smallest tested size within 1% of 64M on every
held-out set.** The 40M candidate missed the tolerance on vocalized committee text. This is a
practical choice among the tested sizes, not a global optimum over every possible corpus.

| Held-out text | 48M tokens/character | 64M tokens/character |
|---|---:|---:|
| Committee | 0.2970 | 0.2986 |
| General Hebrew | 0.3462 | 0.3470 |
| English | 0.3300 | 0.3278 |
| Vocalized committee | 0.2711 | 0.2705 |
| Vocalized general Hebrew | 0.3248 | 0.3230 |

The selected fit took 62.4 seconds on CPU (64M: 79.5 seconds). All 5,214 evaluation lines
round-tripped exactly, with zero unknown tokens. Hebrew/vocalized sets used zero byte fallback;
English used four byte tokens (0.0080%). Direct coverage checks passed for 52 ASCII Latin
letters, 27 Hebrew letters, 10 digits and all 18 niqqud marks.

Recorded successful source-sampling passes read 843.3 MB total, including the 32M pilot and
64M extension, rather than scanning the roughly 57 GB of input files. Dicta output was reused
between sizes. Selected files and full comparisons are in `artifacts/tokenizer-he-en-niqqud/`.

Re-evaluate/refine from the cached maximum pool without sampling or vocalizing again:

```bash
OMP_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false uv run --no-sync python -u \
    -m hebrew_training.tokenizer_experiment \
    --config artifacts/tokenizer-he-en-niqqud/experiment_config.yaml --reuse-pool \
    --sizes 4 8 16 32 40 48 64
```
