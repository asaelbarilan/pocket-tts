# The MILIM-Bench split

`renikud/MILIM-Bench` (MIT) is 1,653 Hebrew sentences in 11 categories with gold IPA for
3,110 target words. It is the measurement this project was missing: the homograph set in
`../eval_sentences_he.jsonl` holds twelve decisions, and twelve cannot separate one model
from another.

The benchmark text is not vendored here. Fetch it and regenerate the splits:

```
curl -L -o gold.tsv https://huggingface.co/datasets/renikud/MILIM-Bench/resolve/main/data/gold.tsv
python -I ../milim_split.py gold.tsv . --by sentence
python -I ../milim_split.py gold.tsv . --by form
```

The split is a SHA-1 of the sentence text, not a seed, so it is the same on every machine
and after any reshuffle of the file. `milim_*_assignment.tsv` are committed as the audit
trail: `sha1_8` is the first eight hex digits of the SHA-1 of the sentence, so a split can
be checked without redistributing the sentences.

Read `../milim_split.py`'s docstring before using either split. Four things in it will
otherwise cost an afternoon:

- `predictions.tsv` has a `#meta` first row, so it is one row longer than `gold.tsv` and
  the benchmark's own `src/evaluate.py` raises on its length check.
- Rows 940 and 1129 (zero-based) have a target index past the end of the sentence. Dropped.
- `ilspeech-v2` is half the benchmark by target word (1,539 of 3,107), so a micro-average
  is 50% one control category. Quote the macro average over the eleven categories.
- `--by form` collapses `ilspeech-v2` to 11 target words, because its sentences carry ten
  targets each and nearly all of them straddle. Score that category under `--by sentence`.

Cite the benchmark as Melichov, Kolani and Alper, *ReNikud: Audio-Supervised Hebrew
Grapheme-to-Phoneme Conversion*, 2026.
