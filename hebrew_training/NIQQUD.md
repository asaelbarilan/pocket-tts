# Vocalized (niqqud) rows

Training can swap some words of a Hebrew sample for their vocalized forms. The manifest
side is one pass; the loader side is `training/text_augment.py` (see "Training" below).

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
training's by drawing which words keep their marks (see "Training" below). `alt_words`
(digits) comes from the manifest builder for both splits, so validation already has digit rows.

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

## Training: what the loader does with `nikud` and `alt_words`

`training/dataloader.py` trains on `words[i:]` after it has chosen its cut `i`. For each sample
`training/text_augment.py` then decides which form each word takes. Rows with neither key are
not touched: same text, no random numbers drawn, seeded runs bit-identical to before.

Three probabilities, keys of the `data:` section of the config (`DataArgs`), defaults in code,
valid in [0, 1] (anything else is an error at config load), written out in
`finetune_hebrew.yaml` and `depth_distill_hebrew.yaml`:

| key | default | meaning |
|---|---|---|
| `alt_words_sample_prob` | 0.30 | share of samples with `alt_words` that get any digits. 0 turns digits off |
| `alt_word_swap_prob` | 0.60 | in a chosen sample, chance each number is swapped for its digits |
| `nikud_word_prob` | 0.10 | chance each word of a sample with `nikud` is trained vocalized. 0 turns it off |

1. **Vocalize**: each word of `words[i:]` independently becomes `nikud[j]` with probability
   `nikud_word_prob`. There is no per-row rate and no "at least one" rule; plain text is the
   default and marks only appear through this draw.
2. **Digits**: only spans with `from >= i` are candidates. A span the cut falls inside stays
   spoken (the audio of its first words is gone); one before the cut is not in the text. If the
   sample was chosen (`alt_words_sample_prob`) each candidate is swapped with
   `alt_word_swap_prob`; if none was, one random candidate is, so a chosen sample always has at
   least one swap (so `alt_word_swap_prob: 0` means exactly one per chosen sample, not off). No
   candidate, no swap. The swapped-in text is the plain `replace` tokens: niqqud never applies
   inside a swapped span.
3. Order: vocalize first, then swap, on the same list of strings
   (`swap_alternates`, a copy of the builder's reference, with a test that they agree). Token
   count may change; there is no length filter, because a sample with audio of the proper
   duration always has text that fits.
4. A row with `words` but no usable cut (the fallback branch) gets the same treatment with `i =
   0`, instead of the raw `transcript`. Only rows that have `alt_words` or `nikud` (with a
   feature on) take this; every other row still uses its `transcript` as before.

Only the label `digits` of `alt_words` is used; other labels are ignored. A `nikud` list of the
wrong length, or spans out of range or overlapping, make the sample fail (it is skipped and
counted like an unreadable one) instead of training on misaligned text.

### Reproducibility

All draws come from `random.Random` instances seeded from one integer per sample, never from
the loader's shared generator, which the I/O thread pool uses in a racing order.

- **Training**: `get_entry` (sequential, outside the pool) takes the seed from a stream of its
  own, `random.Random("text:<seed>")`, only for rows that have something to vary. The cut and
  shuffle stream is never touched, so adding these keys to a manifest changes no other row's
  cuts. A row varies from visit to visit; the same loader seed replays the same sequence.
- **Validation** (`fixed_draws=True`, set by `validate()` in `training/train.py`): the seed
  is SHA-256 of `seed`, `path` and `start`, so a row gets the same digit/niqqud draws on every
  evaluation and in any order. Validation uses the same probabilities as training; its rows
  carry `nikud` (`--valid-rate 1.0`), so about 10% of validation words are vocalized and about
  30% of digit rows show digits.
- Draws are keyed by position, not by cut: word `j` and span `k` always get the same uniform
  number whatever the cut is, and the vocalize and digit draws use separate generators, so
  changing one probability does not change the other feature's outcome. Two cuts of one row
  agree on every word they share (the cut itself still comes from the loader's cut stream).

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
