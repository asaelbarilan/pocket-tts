"""Split MILIM-Bench into train / validation / test, two ways.

    python -I hebrew_training/milim_split.py <gold.tsv> <out dir> [--by sentence|form]

MILIM-Bench (`renikud/MILIM-Bench`, MIT) is 1,653 Hebrew sentences in 11 categories with
gold IPA for 3,110 target words, written as `token_index=IPA`. It is the measurement this
project was missing: the homograph validation set here holds twelve decisions, which cannot
separate one model from another.

Three things about the file have to be handled, and all three are measured, not assumed:

1. **Row 0 of `predictions.tsv` is a `#meta` row**, so that file has one more row than
   `gold.tsv` and the benchmark's own `src/evaluate.py` raises on its length check. Scoring
   code has to drop it. `gold.tsv` itself has no meta row.
2. **Two rows carry a target index past the end of the sentence** (rows 940 and 1129,
   zero-based; two of the submitters noted it in the meta row). No system can answer them.
   They are dropped here rather than scored as misses.
3. **`ilspeech-v2` is half the benchmark by target word** -- 1,539 of 3,107, because its
   sentences carry about ten targets each while every other category carries one or two. A
   micro-average over target words is therefore 50% one control category. Report the macro
   average over the eleven categories, which this script prints for every split.

The split is deterministic and does not depend on row order or on a random seed: it comes
from a SHA-1 of the sentence text, so the same sentence lands in the same split on any
machine and after any reshuffle of the file.

Two split modes, because they measure different things and the honest answer needs both:

`--by sentence` (default)
    60/20/20 over sentences, stratified by category. The same ambiguous form appears in
    train and in test, in different sentences. This is the deployment setting: the TTS
    will meet `בוקר` in the wild and you would of course have trained on it. It measures
    "given both readings are known, is the context being used".

`--by form`
    60/20/20 over the *target forms* (nikud-stripped), so a form never crosses a split. A
    sentence whose targets fall in more than one split is dropped, and the count is
    reported. It measures generalisation to words never seen, which is much harder and the
    number will be lower. Quote both or the easy one flatters.

    **This mode destroys `ilspeech-v2`**: its sentences carry about ten targets each, so
    almost every one straddles, and the category drops from 1,539 target words to 11. Score
    `ilspeech-v2` under `--by sentence` only. The other ten categories survive intact,
    because they carry one or two targets per sentence.

MILIM is wide and shallow -- 1,284 distinct target forms, none appearing more than four
times -- which is the opposite of what mining a vocalized corpus gives you (where `אוכל`
and `רוצה` would dominate and a model that always guesses the frequent reading would look
good). No per-form reweighting is needed here, and the `--by form` split is meaningful
rather than a near-duplicate of the sentence split.

For the gate of `docs/one-hebrew-semantic-encoder.md` (a few thousand parameters) the
sentence split's ~990 training sentences are plenty. For training an encoder they are not,
and MILIM should be held out as the measurement while the training data comes from a
vocalized corpus or an LLM label factory.
"""

from __future__ import annotations

import collections
import csv
import hashlib
import re
import sys
from pathlib import Path

NIKUD = re.compile(r"[֑-ׇ]")
FRACTIONS = (0.6, 0.2, 0.2)
SPLITS = ("train", "val", "test")


def parse_label(label: str) -> dict[int, str]:
    out = {}
    for part in label.split():
        if "=" in part:
            index, ipa = part.split("=", 1)
            out[int(index)] = ipa
    return out


def bucket(key: str, salt: str) -> str:
    """Stable 0..9999 position for `key`, turned into a split by the fractions above."""
    digest = hashlib.sha1(f"{salt}:{key}".encode("utf-8")).hexdigest()
    position = int(digest[:8], 16) % 10000 / 10000
    edge = 0.0
    for name, fraction in zip(SPLITS, FRACTIONS):
        edge += fraction
        if position < edge:
            return name
    return SPLITS[-1]


def load(gold_path: Path) -> list[dict]:
    rows = list(csv.DictReader(gold_path.open(encoding="utf-8-sig"), delimiter="\t"))
    kept = []
    dropped = 0
    for row in rows:
        targets = parse_label(row["Label"])
        tokens = row["Text"].split()
        if max(targets, default=-1) >= len(tokens):
            dropped += 1  # point 2 in the module docstring
            continue
        kept.append(
            {
                "category": row["Category"],
                "text": row["Text"],
                "label": row["Label"],
                "targets": targets,
                "forms": [NIKUD.sub("", tokens[i]) for i in sorted(targets)],
            }
        )
    print(f"{len(rows)} rows, {dropped} dropped for a target index past the end of the sentence")
    return kept


def assign_by_sentence(rows: list[dict]) -> None:
    # Stratified because the salt includes the category: each category is hashed into its
    # own 60/20/20, so no category can land lopsided by chance.
    for row in rows:
        row["split"] = bucket(row["text"], row["category"])


def assign_by_form(rows: list[dict]) -> int:
    for row in rows:
        wanted = {bucket(form, "form") for form in row["forms"]}
        row["split"] = wanted.pop() if len(wanted) == 1 else None
    straddling = sum(r["split"] is None for r in rows)
    return straddling


def report(rows: list[dict], before: list[dict] | None = None) -> None:
    categories = sorted({r["category"] for r in rows})
    if before is not None:
        for category in categories:
            kept = sum(len(r["targets"]) for r in rows if r["category"] == category)
            total = sum(len(r["targets"]) for r in before if r["category"] == category)
            if total and kept < 0.5 * total:
                print(
                    f"WARNING {category}: {kept} of {total} target words survive the form "
                    f"split. Score this category under --by sentence instead."
                )
    print()
    print(f"{'category':22s}  " + "  ".join(f"{s:>12s}" for s in SPLITS) + f"  {'targets':>8s}")
    for category in categories:
        group = [r for r in rows if r["category"] == category]
        cells = []
        for split in SPLITS:
            part = [r for r in group if r["split"] == split]
            cells.append(f"{len(part):4d}s {sum(len(r['targets']) for r in part):5d}w")
        cells = [f"{c:>12s}" for c in cells]
        print(
            f"{category:22s}  " + "  ".join(cells) + f"  {sum(len(r['targets']) for r in group):8d}"
        )
    cells = []
    for split in SPLITS:
        part = [r for r in rows if r["split"] == split]
        words = sum(len(r["targets"]) for r in part)
        cells.append(f"{len(part):4d}s {words:5d}w".rjust(12))
    print(f"{'TOTAL':22s}  " + "  ".join(cells))


def main() -> None:
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    mode = "sentence"
    if "--by" in sys.argv:
        mode = sys.argv[sys.argv.index("--by") + 1]
    gold_path, out_dir = Path(args[0]), Path(args[1])
    rows = load(gold_path)

    before = None
    if mode == "sentence":
        assign_by_sentence(rows)
    elif mode == "form":
        straddling = assign_by_form(rows)
        print(f"{straddling} sentences dropped: their target forms fall in different splits")
        before, rows = rows, [r for r in rows if r["split"]]
    else:
        raise SystemExit(f"--by takes 'sentence' or 'form', not {mode!r}")

    report(rows, before)

    out_dir.mkdir(parents=True, exist_ok=True)
    for split in SPLITS:
        part = [r for r in rows if r["split"] == split]
        path = out_dir / f"milim_{mode}_{split}.tsv"
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle, delimiter="\t")
            writer.writerow(["Category", "Text", "Label"])
            for row in part:
                writer.writerow([row["category"], row["text"], row["label"]])
        print(f"wrote {path}  {len(part)} sentences")

    # The assignment itself, by text hash, so the split can be reproduced or audited
    # without redistributing the sentences.
    ids = out_dir / f"milim_{mode}_assignment.tsv"
    with ids.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(["sha1_8", "category", "split", "n_targets"])
        for row in sorted(rows, key=lambda r: (r["category"], r["text"])):
            writer.writerow(
                [
                    hashlib.sha1(row["text"].encode("utf-8")).hexdigest()[:8],
                    row["category"],
                    row["split"],
                    len(row["targets"]),
                ]
            )
    print(f"wrote {ids}")

    forms = collections.Counter(f for r in rows for f in r["forms"])
    line = f"\n{len(forms)} distinct target forms; most frequent: " + ", ".join(
        f"{f} {n}" for f, n in forms.most_common(8)
    )
    # Not print(): a Windows console is cp1252 and raises on the Hebrew.
    sys.stdout.buffer.write(line.encode("utf-8") + b"\n")


if __name__ == "__main__":
    main()
