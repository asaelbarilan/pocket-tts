"""Score every system in MILIM-Bench's `predictions.tsv`, per target word.

    python -I hebrew_training/milim_systems.py <gold.tsv> <predictions.tsv> <out.txt>

The benchmark ships `src/evaluate.py`, which scores per *row* (a row is right only if every
target in it is right) and reports WER and CER. This scores per *target word*, which is the
unit a front end actually decides, and it adds the thing the released numbers do not show:
which systems are right in different places.

Three handling notes, all of them measured rather than assumed:

1. Row 0 of `predictions.tsv` is a `#meta` row carrying each system's model id, prompt id
   and token counts. It is not data. The repo's own `evaluate.py` does not skip it, which
   is why its length check fails against the published `gold.tsv`.
2. Rows 940 and 1129 (zero-based) have a gold target index past the end of the sentence.
   No system can answer them; they are dropped, not scored as misses.
3. `ilspeech-v2` is 1,539 of the 3,107 scorable target words, because its sentences carry
   about ten targets each. A micro-average is therefore half one control category, so the
   macro average over the eleven categories is printed alongside it and is the number to
   quote.

Normalisation is the benchmark's own: keep only its phoneme characters, map x/r/g to
chi/resh/gimel. The stress mark is kept, so stress counts as part of the answer.
"""

from __future__ import annotations

import collections
import csv
import sys
from pathlib import Path

PHONEME_CHARS = set("abdefhijklmnopstuvwzɡʁʃʒʔˈχ")
TRANS = str.maketrans({"x": "χ", "r": "ʁ", "g": "ɡ"})
# The pair the complementarity table compares. ReNikud is the local g2p already in the
# pipeline; the second is whichever system tops the macro table.
BASE = "renikud"


def norm(text: str) -> str:
    return "".join(c for c in text.translate(TRANS) if c in PHONEME_CHARS)


def parse_label(label: str) -> dict[int, str]:
    out = {}
    for part in label.split():
        if "=" in part:
            index, ipa = part.split("=", 1)
            out[int(index)] = norm(ipa)
    return out


def prediction_targets(text: str, want) -> dict[int, str]:
    """A system may answer `index=IPA` per target, or emit the whole sentence in IPA."""
    parts = [p for p in text.split() if p]
    if parts and all("=" in p for p in parts):
        indexed = parse_label(text)
        return {i: indexed.get(i, "") for i in want}
    tokens = text.split()
    return {i: (norm(tokens[i]) if i < len(tokens) else "") for i in want}


def load(gold_path: Path, pred_path: Path):
    gold = list(csv.DictReader(gold_path.open(encoding="utf-8-sig"), delimiter="\t"))
    pred = list(csv.DictReader(pred_path.open(encoding="utf-8-sig"), delimiter="\t"))
    systems = list(pred[0].keys())
    if not pred[0][systems[0]].startswith("#meta"):
        raise SystemExit("expected a #meta first row in predictions.tsv; the format changed")
    pred = pred[1:]
    if len(gold) != len(pred):
        raise SystemExit(f"{len(gold)} gold rows against {len(pred)} prediction rows")

    words = []
    dropped = 0
    for g, p in zip(gold, pred):
        want = parse_label(g["Label"])
        tokens = g["Text"].split()
        if max(want, default=-1) >= len(tokens):
            dropped += 1
            continue
        got = {s: prediction_targets(p.get(s, ""), want) for s in systems}
        for i, ipa in want.items():
            words.append(
                {
                    "cat": g["Category"],
                    "text": g["Text"],
                    "tok": tokens[i],
                    "gold": ipa,
                    **{s: got[s][i] for s in systems},
                }
            )
    return words, systems, len(gold), dropped


def accuracy(group, system) -> float:
    return 100 * sum(w[system] == w["gold"] for w in group) / len(group) if group else 0.0


def main() -> None:
    gold_path, pred_path, out_path = (Path(a) for a in sys.argv[1:4])
    words, systems, n_rows, dropped = load(gold_path, pred_path)
    cats = sorted({w["cat"] for w in words})
    by_cat = {c: [w for w in words if w["cat"] == c] for c in cats}

    L = [
        f"{n_rows} sentences ({dropped} dropped for a target index past the end of the "
        f"sentence), {len(words)} target words, {len(systems)} systems",
        "",
        "MACRO is the mean of the eleven per-category accuracies and is the number to quote:",
        f"ilspeech-v2 alone is {len(by_cat['ilspeech-v2'])} of {len(words)} target words, so a "
        "micro-average is half one control category.",
        "",
        f"{'system':34s}  {'MACRO':>6s}  {'micro':>6s}  " + "  ".join(f"{c[:9]:>9s}" for c in cats),
    ]

    def macro(system):
        return sum(accuracy(by_cat[c], system) for c in cats) / len(cats)

    for s in sorted(systems, key=lambda s: -macro(s)):
        cells = "  ".join(f"{accuracy(by_cat[c], s):9.1f}" for c in cats)
        L.append(f"{s:34s}  {macro(s):6.1f}  {accuracy(words, s):6.1f}  " + cells)
    L.append(
        f"{'target words':34s}  {'':6s}  {len(words):6d}  "
        + "  ".join(f"{len(by_cat[c]):9d}" for c in cats)
    )

    best = max((s for s in systems if s != BASE), key=macro)
    L += [
        "",
        f"COMPLEMENTARITY per category: {BASE} against {best}",
        f"{'category':22s}  {'n':>5s}  {BASE[:9]:>9s}  {'other':>9s}  {'both':>6s}"
        f"  {'base only':>9s}  {'other only':>10s}  {'oracle':>7s}",
    ]
    for c in cats + ["__ALL__"]:
        group = words if c == "__ALL__" else by_cat[c]
        a = [w[BASE] == w["gold"] for w in group]
        b = [w[best] == w["gold"] for w in group]
        both = sum(x and y for x, y in zip(a, b))
        a_only = sum(x and not y for x, y in zip(a, b))
        b_only = sum(y and not x for x, y in zip(a, b))
        L.append(
            f"{c:22s}  {len(group):5d}  {100 * sum(a) / len(group):9.1f}  {100 * sum(b) / len(group):9.1f}"
            f"  {both:6d}  {a_only:9d}  {b_only:10d}  {100 * (both + a_only + b_only) / len(group):7.1f}"
        )

    pool = sum(any(w[s] == w["gold"] for s in systems) for w in words)
    never = [w for w in words if not any(w[s] == w["gold"] for s in systems)]
    L += [
        "",
        f"oracle over all {len(systems)} systems: {100 * pool / len(words):.1f}%",
        f"target words no system gets right: {len(never)} ({100 * len(never) / len(words):.1f}%)",
        "  "
        + ", ".join(
            f"{k} {v}" for k, v in collections.Counter(w["cat"] for w in never).most_common()
        ),
        "",
        f"Where {BASE} alone is right and {best} is wrong -- what a gate has to catch:",
    ]
    base_only = [w for w in words if w[BASE] == w["gold"] and w[best] != w["gold"]]
    L.append(
        "  "
        + ", ".join(
            f"{k} {v}" for k, v in collections.Counter(w["cat"] for w in base_only).most_common()
        )
    )
    L.append("")
    for w in [x for x in base_only if x["cat"] in ("colloquial", "slang")][:10]:
        L.append(
            f"  [{w['cat']:10s}] {w['tok']}  gold={w['gold']}  {BASE}={w[BASE]}  other={w[best]}"
        )
        L.append(f"      {w['text']}")

    text = "\n".join(L)
    out_path.write_text(text + "\n", encoding="utf-8")
    # Not print(): a Windows console is cp1252 and raises on the Hebrew and the IPA.
    sys.stdout.buffer.write(text.encode("utf-8") + b"\n")


if __name__ == "__main__":
    main()
