"""Add vocalized text to a manifest, so training can swap some words for their niqqud forms.

    python -m hebrew_training.diacritize_official_manifest /data/plenum \\
        --model <dicta model> --revision <pinned>

Reads `train.jsonl` and `valid.jsonl` (build_official_manifest) and writes `train_diac.jsonl`
and `valid_diac.jsonl`. Nothing is modified in place. A row that gets vocalized is written as
it was with one more key, `nikud`: a list with one string per entry of `words`, the same word
with its marks. `words`, `transcript`, the times and `alt_words` are untouched, so the aligner
can run before or after this pass. Rows that are not vocalized -- which is most of them -- are
copied byte for byte, so an old reader and a row without `nikud` see no difference. The loader
treats a missing `nikud` as "not eligible"; it strips marks by default and keeps them on some
words only by a draw at load time (not part of this pass).

WHICH ROWS
----------
1. Eligible: has a Hebrew letter and has no marks already. There is no length rule: vocalized
   text is longer, but a row of audio that fits the model always has text that fits it too, so
   any row that can take marks may.
2. Of the eligible rows, `--rate` (default 0.3) are picked, by a hash of the seed and the
   row's audio position, so a rebuild picks the same rows. Validation uses `--valid-rate`
   (default 1.0): every eligible validation row is vocalized, so validation can measure the
   vocalized case and the loader can still show it a mix like training's. Digits need nothing
   here -- `alt_words` already comes from the manifest builder, for both splits.
3. After the diacritizer has run, a row is kept vocalized only if the result strips back to its
   `words` exactly and has the same word count. Anything else stays as it was and is counted in
   the summary.

Long rows are cut at word boundaries before going to the diacritizer, which truncates past
about 400 characters.

The run is resumable (`train_diac.progress.json`) and the output is the same row for row as
the input, so line N of the output is line N of the input.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

from hebrew_training.build_official_manifest import stable_bucket
from hebrew_training.diacritizer import (
    FAKE,
    DiacritizerSpec,
    add_diacritizer_args,
    open_diacritizer,
    spec_from_args,
)
from hebrew_training.manifest_files import BUILT, DIACRITIZED, SPLITS, check_manifest_head
from hebrew_training.nikud import has_hebrew_letter, has_marks, vocalize_word_lists

DEFAULT_RATE = 0.3
DEFAULT_VALID_RATE = 1.0
CHECKPOINT_ROWS = 2048

STAT_KEYS = (
    "rows",
    "ineligible_no_hebrew",
    "ineligible_has_marks",
    "eligible",
    "not_picked",
    "picked",
    "failed_word_count_changed",
    "failed_content_changed",
    "diacritized",
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("folder", type=Path, help="Folder holding train.jsonl and valid.jsonl")
    parser.add_argument("--splits", nargs="+", choices=SPLITS, default=["valid", "train"])
    parser.add_argument(
        "--rate",
        type=float,
        default=DEFAULT_RATE,
        help=f"Share of eligible train rows to vocalize (default {DEFAULT_RATE}).",
    )
    parser.add_argument(
        "--valid-rate",
        type=float,
        default=DEFAULT_VALID_RATE,
        help=f"The same for valid (default {DEFAULT_VALID_RATE}: all eligible rows).",
    )
    parser.add_argument("--seed", type=int, default=0, help="Seeds which rows are picked.")
    parser.add_argument(
        "--batch-rows",
        type=int,
        help="Rows handed to the diacritizer at a time (default: enough to keep every GPU busy, "
        "2 x --batch-size x the number of GPUs, at least 128).",
    )
    parser.add_argument("--limit", type=int, help="Process at most this many rows per split.")
    parser.add_argument("--overwrite", action="store_true")
    add_diacritizer_args(parser)
    args = parser.parse_args(argv)
    for name in ("rate", "valid_rate"):
        if not 0 <= getattr(args, name) <= 1:
            parser.error(f"--{name.replace('_', '-')} must be between 0 and 1")
    if (args.batch_rows is not None and args.batch_rows < 1) or args.batch_size < 1:
        parser.error("--batch-rows and --batch-size must be at least 1")
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be at least 1")
    return args


def is_picked(row: dict, seed: int, rate: float) -> bool:
    # `original_start` is what the aligner leaves behind when it moves `start`, so the same
    # row is picked whether this pass runs before or after alignment.
    start = row.get("original_start", row["start"])
    return stable_bucket(f"{seed}:{row['path']}:{start}") < rate


def triage(row: dict) -> tuple[str, list[str]]:
    """`("eligible", words)`, or the reason the row is left alone."""
    words = [item["word"] for item in row["words"]]
    text = " ".join(words)
    if not has_hebrew_letter(text):
        return "ineligible_no_hebrew", words
    if has_marks(text):
        return "ineligible_has_marks", words
    return "eligible", words


def process_batch(
    lines: list[bytes], *, rate: float, seed: int, predict, batch_size: int, stats: dict[str, int]
) -> list[bytes]:
    """The output line for each input line, in order. Unchanged rows are returned as read."""
    rows = [json.loads(line) for line in lines]
    wanted: list[int] = []
    word_lists: list[list[str]] = []
    for index, row in enumerate(rows):
        stats["rows"] += 1
        status, words = triage(row)
        if status != "eligible":
            stats[status] += 1
            continue
        stats["eligible"] += 1
        if not is_picked(row, seed, rate):
            stats["not_picked"] += 1
            continue
        stats["picked"] += 1
        wanted.append(index)
        word_lists.append(words)

    changed: dict[int, list[str]] = {}
    if word_lists:
        for index, words, result in zip(
            wanted, word_lists, vocalize_word_lists(word_lists, predict, batch_size), strict=True
        ):
            if isinstance(result, str):
                stats[f"failed_{result}"] += 1
            else:
                stats["diacritized"] += 1
                changed[index] = result

    out = []
    for index, (line, row) in enumerate(zip(lines, rows, strict=True)):
        if index in changed:
            row["nikud"] = changed[index]
            line = (json.dumps(row, ensure_ascii=False) + "\n").encode("utf-8")
        out.append(line if line.endswith(b"\n") else line + b"\n")
    return out


def save_progress(path: Path, state: dict) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(state) + "\n", encoding="utf-8")
    temporary.replace(path)


def open_output(output: Path, overwrite: bool, configuration: dict):
    """`(handle, input_lines_done, stats)`, resuming a run with the same configuration."""
    progress = output.with_suffix(".progress.json")
    if overwrite:
        for path in (output, progress):
            path.unlink(missing_ok=True)
    if output.exists() and not progress.exists():
        raise FileExistsError(f"{output} already exists; use --overwrite to replace it")
    state = {
        "input_lines": 0,
        "output_bytes": 0,
        "stats": dict.fromkeys(STAT_KEYS, 0),
        "configuration": configuration,
    }
    if progress.exists():
        saved = json.loads(progress.read_text(encoding="utf-8"))
        if saved.get("configuration") != configuration:
            raise SystemExit(
                f"{progress} was started with a different configuration; use --overwrite to "
                "start again"
            )
        state.update(saved)
    output.parent.mkdir(parents=True, exist_ok=True)
    handle = output.open("r+b" if output.exists() else "w+b")
    handle.truncate(state["output_bytes"])
    handle.seek(0, os.SEEK_END)
    return handle, state["input_lines"], state["stats"]


def run_split(
    args: argparse.Namespace, split: str, *, source: Path, predict, configuration: dict
) -> dict[str, int] | None:
    output = args.folder / DIACRITIZED.format(split=split)
    rate = args.rate if split == "train" else args.valid_rate
    handle, done, stats = open_output(output, args.overwrite, configuration[split])
    progress = output.with_suffix(".progress.json")
    started = time.monotonic()
    processed = last_report = 0

    def checkpoint(line_number: int) -> None:
        handle.flush()
        os.fsync(handle.fileno())
        save_progress(
            progress,
            {
                "input_lines": line_number,
                "output_bytes": handle.tell(),
                "stats": stats,
                "configuration": configuration[split],
            },
        )

    line_number = done
    pending: list[bytes] = []
    try:
        with source.open("rb") as reader:
            for number, line in enumerate(reader, 1):
                if number <= done:
                    continue
                if not line.strip():
                    line_number = number
                    continue
                pending.append(line)
                line_number = number
                limit_hit = args.limit is not None and processed + len(pending) >= args.limit
                if len(pending) < args.batch_rows and not limit_hit:
                    continue
                handle.write(
                    b"".join(
                        process_batch(
                            pending,
                            rate=rate,
                            seed=args.seed,
                            predict=predict,
                            batch_size=args.batch_size,
                            stats=stats,
                        )
                    )
                )
                processed += len(pending)
                pending = []
                if processed - last_report >= CHECKPOINT_ROWS:
                    checkpoint(line_number)
                    last_report = processed
                    elapsed = time.monotonic() - started
                    print(
                        f"{split}: {processed} rows, {processed / elapsed:.1f} rows/s, "
                        f"{stats['diacritized']} vocalized",
                        flush=True,
                    )
                if limit_hit:
                    break
            if pending:
                handle.write(
                    b"".join(
                        process_batch(
                            pending,
                            rate=rate,
                            seed=args.seed,
                            predict=predict,
                            batch_size=args.batch_size,
                            stats=stats,
                        )
                    )
                )
        checkpoint(line_number)
    finally:
        handle.close()
    if args.limit is None:
        progress.unlink(missing_ok=True)
    return stats


def describe(split: str, rate: float, stats: dict[str, int]) -> str:
    rows, eligible = stats["rows"], stats["eligible"]
    lines = [
        (
            f"{split}: {rows} rows; eligible {eligible} (no Hebrew "
            f"{stats['ineligible_no_hebrew']}, already marked {stats['ineligible_has_marks']})"
        ),
        (
            f"{split}: picked {stats['picked']} of the eligible at rate {rate}; vocalized "
            f"{stats['diacritized']} ({stats['diacritized'] / max(1, rows):.1%} of all rows)"
        ),
    ]
    failed = {k: v for k, v in stats.items() if k.startswith("failed_") and v}
    if failed:
        lines.append(f"{split}: left plain after vocalizing: {failed}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    args.folder = args.folder.expanduser().resolve()
    spec: DiacritizerSpec = spec_from_args(args)
    real = args.diacritizer != FAKE

    sources = {}
    for split in args.splits:
        source = args.folder / BUILT.format(split=split)
        if not source.is_file():
            raise FileNotFoundError(source)
        check_manifest_head(source, expected_from="build_official_manifest", forbidden=("nikud",))
        sources[split] = source

    configuration = {
        split: {
            "rate": args.rate if split == "train" else args.valid_rate,
            "seed": args.seed,
            "diacritizer": {
                "backend": args.diacritizer,
                "model": spec.model,
                "revision": spec.revision,
                "dtype": spec.dtype,
                "preserve_matres": spec.preserve_matres,
            },
            "source": {"name": sources[split].name, "bytes": sources[split].stat().st_size},
        }
        for split in args.splits
    }

    # Everything that can fail cheaply has failed by now; only then load the model.
    runnable = []
    for split in args.splits:
        output = args.folder / DIACRITIZED.format(split=split)
        progress = output.with_suffix(".progress.json")
        if output.exists() and not progress.exists() and not args.overwrite:
            print(f"{split}: {output.name} already exists; use --overwrite to replace it")
        else:
            runnable.append(split)
    if not runnable:
        return
    if args.diacritizer == FAKE:
        print("NOTE: the fake diacritizer is for mechanics tests; its output is not real niqqud.")
    with open_diacritizer(args) as predict:
        if args.batch_rows is None:
            args.batch_rows = max(128, 2 * args.batch_size * len(predict.devices))
        for split in runnable:
            print(
                f"{split}: vocalizing {sources[split].name} into {DIACRITIZED.format(split=split)}"
            )
            stats = run_split(
                args, split, source=sources[split], predict=predict, configuration=configuration
            )
            rate = args.rate if split == "train" else args.valid_rate
            print(describe(split, rate, stats))
            summary = args.folder / f"{split}_diac.summary.json"
            summary.write_text(
                json.dumps(
                    {
                        "stats": stats,
                        "configuration": configuration[split],
                        "synthetic": not real,
                        "complete": args.limit is None,
                    },
                    ensure_ascii=False,
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )


if __name__ == "__main__":
    main(sys.argv[1:])
