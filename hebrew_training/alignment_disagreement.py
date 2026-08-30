"""Rank utterances by how much two aligners disagree, and emit them for hand correction.

We have no Hebrew word-alignment ground truth -- every ivrit.ai corpus, Knesset and
CrowdRecital alike, carries timings from the same family of tools, so measuring one against
another reports agreement rather than accuracy. A gold set has to be made by hand, and
hand-annotation is the expensive part. This makes it cheap by choosing what to annotate.

Where two independently-trained aligners agree, there is little to learn and a human mark
would probably land in the same place. Where they disagree, the ranking between them is
decided, and a human mark is worth the minute it costs. So: align the same utterances twice
with different models, rank by disagreement, annotate the top of the list.

It compares two ALREADY-ALIGNED manifests rather than running the aligners itself, so it has
no ML dependencies and does not care which aligners produced them. Suggested pair, chosen to
share as little as possible:

    A  imvladikon/wav2vec2-xls-r-300m-hebrew   via data_prep/align_hebrew.py
                                               CTC over native Hebrew characters
    B  MahmoudAshraf/mms-300m-1130-forced-aligner  via the ctc-forced-aligner package
                                               CTC over uroman romanization, 1130+ languages

    python -m hebrew_training.alignment_disagreement \
        --a aligned_imvladikon.jsonl --b aligned_mms.jsonl \
        --out disagreement.jsonl --textgrid-dir annotate --top 30

Each TextGrid carries three tiers: the two aligners, and an empty `gold` tier for the
annotator to fill in Praat. Correcting a pre-filled boundary is far faster than placing one,
and having both machines visible shows exactly where the question is.

Reported per utterance: median and p90 of |A - B| on word starts and on word ends. Ends are
reported separately because `last_word_end` drives the trailing-silence trim, which Kyutai
name as the cause of a model that never emits EOS.
"""

from __future__ import annotations

import argparse
import json
import statistics
import unicodedata
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--a", type=Path, required=True, help="First aligned manifest.")
    parser.add_argument("--b", type=Path, required=True, help="Second aligned manifest.")
    parser.add_argument("--name-a", default="A", help="Tier name for --a in the TextGrids.")
    parser.add_argument("--name-b", default="B", help="Tier name for --b in the TextGrids.")
    parser.add_argument("--out", type=Path, required=True, help="Ranked jsonl, worst first.")
    parser.add_argument(
        "--textgrid-dir", type=Path, help="Write Praat TextGrids for the --top worst utterances."
    )
    parser.add_argument(
        "--top", type=int, default=30, help="How many utterances to emit for annotation."
    )
    parser.add_argument(
        "--min-words",
        type=int,
        default=3,
        help="Skip very short utterances; their statistics are noise.",
    )
    return parser.parse_args()


def norm(text: str) -> str:
    return unicodedata.normalize("NFC", text).strip()


def load(path: Path) -> dict[tuple[str, float], dict]:
    """Manifest keyed by (path, start), which is what identifies an utterance."""
    rows: dict[tuple[str, float], dict] = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            rows[(row["path"], round(float(row.get("start", 0.0)), 3))] = row
    return rows


def timed_words(row: dict) -> list[dict]:
    return [
        w
        for w in (row.get("words") or [])
        if w.get("start") is not None and w.get("end") is not None
    ]


def compare(row_a: dict, row_b: dict, min_words: int) -> dict | None:
    """Per-word boundary differences, or None when the two are not comparable.

    The word SEQUENCES must match. Two aligners given the same transcript normally produce
    the same words, but not always -- one may drop a token it could not align. Comparing
    index-by-index across differing sequences would silently pair up unrelated words, so
    mismatches are reported rather than measured.
    """
    words_a, words_b = timed_words(row_a), timed_words(row_b)
    if len(words_a) < min_words or len(words_b) < min_words:
        return None
    if len(words_a) != len(words_b):
        return {"mismatch": f"word count {len(words_a)} vs {len(words_b)}"}
    if [norm(w["word"]) for w in words_a] != [norm(w["word"]) for w in words_b]:
        return {"mismatch": "word sequences differ"}

    starts = [abs(float(a["start"]) - float(b["start"])) for a, b in zip(words_a, words_b)]
    ends = [abs(float(a["end"]) - float(b["end"])) for a, b in zip(words_a, words_b)]
    worst = max(range(len(ends)), key=lambda i: max(starts[i], ends[i]))
    return {
        "words": len(words_a),
        "start_median": statistics.median(starts),
        "start_p90": sorted(starts)[min(len(starts) - 1, int(0.9 * len(starts)))],
        "end_median": statistics.median(ends),
        "end_p90": sorted(ends)[min(len(ends) - 1, int(0.9 * len(ends)))],
        "end_max": max(ends),
        "worst_word": norm(words_a[worst]["word"]),
        "worst_delta": max(starts[worst], ends[worst]),
        # The trailing-silence trim uses only this one, so a disagreement here is worth
        # more than the same disagreement mid-utterance.
        "last_word_end_delta": abs(float(words_a[-1]["end"]) - float(words_b[-1]["end"])),
    }


def escape(text: str) -> str:
    return text.replace('"', '""')


def interval_tier(name: str, words: list[dict], xmax: float, index: int) -> list[str]:
    """One IntervalTier. Praat requires intervals to tile [0, xmax] with no gaps."""
    intervals: list[tuple[float, float, str]] = []
    cursor = 0.0
    for word in words:
        start, end = max(0.0, float(word["start"])), min(xmax, float(word["end"]))
        # Aligners can emit a word that starts before the previous one ended -- Praat rejects
        # overlapping intervals, so the later word is clipped to begin where the last ended.
        # Only the display is clipped; the measurements above use the raw times.
        start = max(start, cursor)
        if end <= start + 1e-6:
            continue
        if start > cursor + 1e-6:
            intervals.append((cursor, start, ""))
        intervals.append((start, end, norm(word["word"])))
        cursor = end
    if cursor < xmax - 1e-6:
        intervals.append((cursor, xmax, ""))
    if not intervals:
        intervals = [(0.0, xmax, "")]

    lines = [
        f"    item [{index}]:",
        '        class = "IntervalTier"',
        f'        name = "{escape(name)}"',
        "        xmin = 0",
        f"        xmax = {xmax:.6f}",
        f"        intervals: size = {len(intervals)}",
    ]
    for position, (start, end, text) in enumerate(intervals, 1):
        lines += [
            f"        intervals [{position}]:",
            f"            xmin = {start:.6f}",
            f"            xmax = {end:.6f}",
            f'            text = "{escape(text)}"',
        ]
    return lines


def write_textgrid(path: Path, duration: float, tiers: list[tuple[str, list[dict]]]) -> None:
    lines = [
        'File type = "ooTextFile"',
        'Object class = "TextGrid"',
        "",
        "xmin = 0",
        f"xmax = {duration:.6f}",
        "tiers? <exists>",
        f"size = {len(tiers)}",
        "item []:",
    ]
    for index, (name, words) in enumerate(tiers, 1):
        lines += interval_tier(name, words, duration, index)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    rows_a, rows_b = load(args.a), load(args.b)
    shared = sorted(set(rows_a) & set(rows_b))
    print(
        f"{len(rows_a)} rows in {args.a.name}, {len(rows_b)} in {args.b.name}, "
        f"{len(shared)} in both"
    )
    if not shared:
        raise SystemExit(
            "no utterances in common. Both manifests must key on the same (path, start); "
            "if one was rebuilt after a transcode, its paths changed."
        )

    scored, mismatched = [], 0
    for key in shared:
        result = compare(rows_a[key], rows_b[key], args.min_words)
        if result is None:
            continue
        if "mismatch" in result:
            mismatched += 1
            continue
        scored.append(
            {
                "path": key[0],
                "start": key[1],
                "transcript": rows_a[key].get("transcript", ""),
                **result,
            }
        )

    if not scored:
        raise SystemExit(f"nothing comparable ({mismatched} had mismatched word sequences)")

    # Rank by p90 rather than median: one badly-placed boundary is what makes an utterance
    # worth a human minute, and a median hides it.
    scored.sort(key=lambda r: r["end_p90"], reverse=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8", newline="\n") as handle:
        for row in scored:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    ends = [r["end_median"] for r in scored]
    last = [r["last_word_end_delta"] for r in scored]
    print(
        f"\ncomparable utterances : {len(scored)}"
        + (f"   ({mismatched} skipped, word sequences differ)" if mismatched else "")
    )
    print(
        f"per-utterance median end disagreement: median {statistics.median(ends) * 1000:.0f} ms, "
        f"p90 {sorted(ends)[int(0.9 * len(ends))] * 1000:.0f} ms"
    )
    print(
        f"last_word_end disagreement           : median {statistics.median(last) * 1000:.0f} ms, "
        f"p90 {sorted(last)[int(0.9 * len(last))] * 1000:.0f} ms"
    )
    print(f"wrote {args.out} (worst first)")

    if not args.textgrid_dir:
        print("\npass --textgrid-dir to emit the worst utterances for annotation in Praat")
        return

    args.textgrid_dir.mkdir(parents=True, exist_ok=True)
    written = 0
    for rank, row in enumerate(scored[: args.top], 1):
        key = (row["path"], row["start"])
        source_a, source_b = rows_a[key], rows_b[key]
        duration = float(source_a.get("duration") or 0.0)
        if duration <= 0:
            continue
        # Times in the manifest are relative to `start`, so the TextGrid spans [0, duration]
        # and lines up with an extracted clip rather than the whole recording.
        write_textgrid(
            args.textgrid_dir / f"{rank:03d}_{Path(row['path']).stem}_{row['start']:.0f}.TextGrid",
            duration,
            [
                (args.name_a, timed_words(source_a)),
                (args.name_b, timed_words(source_b)),
                ("gold", []),  # empty tier: the annotator fills this one
            ],
        )
        written += 1
    print(f"wrote {written} TextGrids to {args.textgrid_dir}")
    print("open each beside its audio in Praat, and fill the empty 'gold' tier")


if __name__ == "__main__":
    main()
