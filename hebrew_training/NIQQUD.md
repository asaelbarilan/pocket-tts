# Vocalized (niqqud) rows

Training can swap some words of a Hebrew sample for their vocalized forms. The manifest
side is one pass; the loader side is not written yet.

```bash
python -m hebrew_training.diacritize_official_manifest /data/plenum \
    --model <dicta> --revision <pinned>
```

`--diacritizer fake` runs without a GPU or a model download. It puts one vowel mark on every
letter, so it checks the mechanics and nothing else; the summary files say `"synthetic": true`.

## What a row looks like

```json
{"words": [{"word": "שלום", ...}, ...], "nikud": ["שָׁלוֹם", ...], ...}
```

`nikud` is one string per entry of `words`. `words` is never changed, so everything that
reads `words` -- the aligner, the data report, the tokenizer's manifest source, WER tooling --
sees what it always saw, and `alt_words` indices stay valid. A row without `nikud` was not
vocalized; that is every row of an older manifest and most rows of a new one.

The invariant, checked when the row is written and by the manifest check on read:

    strip_marks(nikud[i]) == words[i]

`strip_marks` (`nikud.py`) is NFC followed by dropping U+05B0–05BD, 05BF, 05C1, 05C2, 05C7 and
the cantillation block U+0591–05AF. Maqaf, paseq, sof pasuq and geresh stay. The diacritizer is
run with matres lectionis preserved, so it adds marks and nothing else.

## Which rows

1. **Eligible**: has a Hebrew letter and no marks already. There is no length rule: vocalized
   text is longer, but a row of audio that fits the model always has text that fits it too, so
   every row that can take marks may.
2. **Picked**: `--rate` (default 0.3) of the eligible train rows, by a hash of the seed and
   the row's `path` and (original) `start`. The same rows are picked on a rebuild and whether
   the pass runs before or after alignment.
3. **Kept**: only if the diacritizer's output has the same word count and strips back to
   `words` exactly. Otherwise the row stays plain and is counted in `train_diac.summary.json`.

Validation uses `--valid-rate` (default 1.0): every eligible validation row is vocalized. That
lets validation measure the vocalized case, and a loader can still show it a mix like
training's by drawing which words keep their marks. `alt_words` (digits) comes from the
manifest builder for both splits, so validation already has digit rows.

Rows longer than the diacritizer's own limit (it truncates past about 400 characters) are cut
at word boundaries, vocalized in pieces and put back together.

## Stage order and names

    build_official_manifest   -> train.jsonl
    diacritize_official_manifest -> train_diac.jsonl      (optional)
    align_official_manifest   -> train_aligned.jsonl      (optional)

The diacritize pass reads `train.jsonl` and refuses a file that already has `nikud`. The
aligner reads `train_diac.jsonl` if there is one, else `train.jsonl`, and refuses a file that
has been aligned. Each pass checks the first 20 rows for the structure it expects before it
does anything. Each carries the row over as it was and adds only its own keys, so
diacritize-then-align and align-then-diacritize give the same rows
(`tests/test_diacritize_official_manifest.py`).

The diacritize pass is resumable (`train_diac.progress.json`) and writes one output row per
input row. Rows it does not vocalize are copied byte for byte.

## GPUs

`diacritize_official_manifest` finds the CUDA devices itself
and print what they found, for example
`found 4 GPU(s): 0: NVIDIA A40 (44.4 GiB); ... Using 4: 0,1,2,3 (NVIDIA A40 / NVIDIA L4)`.

- default (`--gpus all`): every GPU.
- `--gpus 1,3,4`: only those. Indices are torch's, so they follow `CUDA_VISIBLE_DEVICES`.
- `--gpus cpu` / `--gpus mps`: one device, no GPU needed (CPU DictaBERT is very slow).
- No GPU found and the real model asked for: an error that says so; nothing falls back silently.

One device runs in-process. Several run one worker process per GPU: each loads its own copy of
the model on its GPU (it sees only that GPU, as `cuda:0`) and takes batches from a shared queue,
and results are put back in order, so the output does not depend on how many GPUs were used.
`--batch-size` is lines per Dicta batch on each GPU (default 32); `--batch-rows` defaults to
`2 x batch-size x GPUs` rows per round, at least 128.

## Audio

The diacritize pass never touches audio. The builder's transcoded WAVs (`OUT_DIR/audio/<id>.wav`)
are never rewritten once complete (the file is renamed into place last); a rebuild reuses them,
into the same folder or, with `--audio-dir OLD/audio`, into a new one.
