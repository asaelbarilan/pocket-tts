"""Bring the `alt_words` of existing manifests up to date, touching nothing else.

`alt_words` used to hold only the written-with-digits form of spelled-out numbers (label
`digits`). It now holds the corpus's own form of EVERY word the normalizer rewrote (label
`norms`, see the docstring of build_official_manifest.py). The words, their times, the
`nikud` and everything else in a row are unchanged by that, so a manifest that was already
built, diacritized and aligned does not need to go through any of those passes again. This
script recomputes `alt_words` from the corpus and writes it into the existing rows:

    uv run python -m hebrew_training.migrate_alt_words \\
        --corpus /data/knesset/plenum --normalize-text \\
        --normalizer-dir ../hebrew-tts-data-tools/normalizer \\
        --manifest /data/plenum/train_aligned.jsonl /data/plenum/valid_aligned.jsonl \\
        --out-dir /data/plenum_norms --workers 16

Give it the same corpus options the manifest was built with (`--min-duration`,
`--merge-target`, `--merge-gap`, `--merge-jitter`, `--seed`, `--min-quality`, `--audio-glob`,
...; they are the builder's flags). That makes the same rows, with the same words. Each
existing row is matched to the rebuilt one by recording and `start` (the aligner moves
`start` and leaves the old one in `original_start`, which is used when present) and checked
word for word. A row that does not match -- different options, or a normalizer that has
changed since the build -- stops the run with the row named (`--on-mismatch fail`, the
default), or keeps the row, minus its now-stale `alt_words`, and counts it (`skip`).

Everything else about a row is carried over byte for byte: only the `alt_words` value changes,
in place when the row had one. Rows stream through in order, one recording at a time, so
memory stays at a few recordings however large the manifest is. Nothing is aligned,
diacritized, transcoded or read from audio: the cost is the normalizer pass over the
transcripts, and `--workers` spreads it over processes. Output is the same for any worker
count.

`--out-dir` receives one file per `--manifest`, under the same name. Writing over the input
(`--out-dir` the manifest's own folder) needs `--in-place`; either way the new file is
written beside the target and renamed into place last, so an interrupted run leaves the old
file intact.

The summary lists how many spans each kind of rewrite produced (the `k` tags in
build_official_manifest.NORM_KINDS).
"""

from __future__ import annotations

import argparse
import itertools
import json
import time
from collections import Counter, deque
from collections.abc import Iterator, Sequence
from concurrent.futures import Future, ProcessPoolExecutor
from pathlib import Path

from hebrew_training import build_official_manifest as builder
from hebrew_training.manifest_files import check_manifest_head

PROGRESS_SECONDS = 10.0
# Recordings read ahead of the one being written, per worker.
LOOKAHEAD_PER_WORKER = 2

_ARGS: argparse.Namespace | None = None
_NORMALIZE = None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Rewrite the `alt_words` of existing manifests from the corpus, without "
        "re-aligning or re-diacritizing them.",
        parents=[builder.build_parser(add_help=False)],
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        nargs="+",
        required=True,
        help="Manifests to migrate: any stage's output (train.jsonl, train_diac.jsonl, "
        "train_aligned.jsonl, ...), each one from a single corpus. --out-dir is where the "
        "migrated copies go.",
    )
    parser.add_argument(
        "--in-place",
        action="store_true",
        help="Allow --out-dir to be the manifest's own folder, replacing the file.",
    )
    parser.add_argument(
        "--on-mismatch",
        choices=("fail", "skip"),
        default="fail",
        help="A row that is not found in the rebuild, or whose words differ: stop (default), "
        "or keep the row without `alt_words` and count it.",
    )
    return parser


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = build_parser()
    args = parser.parse_args(argv)
    builder.check_merge_args(parser, args)
    if not args.normalize_text:
        parser.error(
            "pass --normalize-text, as the manifest was built with: without the normalizer "
            "there is nothing to recompute"
        )
    if args.workers < 1:
        parser.error("--workers must be at least 1")
    if not args.corpus.is_dir():
        parser.error(f"--corpus is not a directory: {args.corpus}")
    for manifest in args.manifest:
        if not manifest.is_file():
            parser.error(f"--manifest not found: {manifest}")
        if (args.out_dir / manifest.name).resolve() == manifest.resolve() and not args.in_place:
            parser.error(
                f"{manifest} would be overwritten by its own migrated copy; pass --in-place "
                "or a different --out-dir"
            )
    return args


def recording_name(row: dict, transcoded: bool) -> str:
    """The corpus folder a row came from: `<recording>.wav` in the audio folder when the audio
    was transcoded, else the folder holding the source audio file."""
    path = Path(row["path"])
    return path.stem if transcoded else path.parent.name


def row_key(row: dict) -> float:
    """Where the row starts in the recording as the builder wrote it: the aligner moves `start`
    and keeps the original in `original_start`."""
    return round(float(row.get("original_start", row["start"])), 4)


def _init_worker(args: argparse.Namespace) -> None:
    global _ARGS, _NORMALIZE
    _ARGS = args
    _NORMALIZE = builder.load_normalizer(args.normalizer_dir)


def expected_rows(name: str) -> dict[float, tuple[list[str], dict | None]]:
    """What the builder makes of recording `name`: `{start: (word texts, alt_words)}`."""
    assert _ARGS is not None
    recording = _ARGS.corpus / name
    if not recording.is_dir():
        raise RuntimeError(
            f"the manifest has rows from recording {name!r}, which is not in {_ARGS.corpus}; "
            "is --corpus the corpus it was built from?"
        )
    rows, _ = builder.segment_rows(
        recording, _ARGS, _NORMALIZE, builder._recording_rng(_ARGS.seed, recording)
    )
    return {
        row["start"]: ([word["word"] for word in row["words"]], row.get("alt_words"))
        for row in rows
    }


class _Immediate:
    """A finished result with a Future's interface, for the single-process path."""

    def __init__(self, value) -> None:
        self.value = value

    def result(self):
        return self.value


def with_alt_words(row: dict, alt_words: dict | None) -> dict:
    """`row` with `alt_words` replaced, in the same key position when it had one."""
    out: dict = {}
    placed = False
    for key, value in row.items():
        if key == "alt_words":
            if alt_words:
                out[key] = alt_words
            placed = True
        else:
            out[key] = value
    if alt_words and not placed:
        out["alt_words"] = alt_words
    return out


class Stats:
    def __init__(self) -> None:
        self.rows = 0
        self.with_spans = 0
        self.spans = 0
        self.mismatched = Counter()
        self.kinds: Counter[str] = Counter()

    def add(self, alt_words: dict | None) -> None:
        self.rows += 1
        spans = (alt_words or {}).get(builder.NORMS, [])
        if spans:
            self.with_spans += 1
        self.spans += len(spans)
        self.kinds.update(span.get("k", "?") for span in spans)

    def lines(self, title: str) -> list[str]:
        skipped = sum(self.mismatched.values())
        head = (
            f"{title}: {self.rows} rows, {self.with_spans} with alternates "
            f"({self.spans} spans), {skipped} left without alt_words"
        )
        out = [head]
        for reason, count in sorted(self.mismatched.items()):
            out.append(f"    mismatch {reason}: {count}")
        for tag, count in self.kinds.most_common():
            out.append(f"    {tag}  {count:>10}  {builder.NORM_KINDS.get(tag, 'unknown tag')}")
        return out


def mismatch_reason(
    row: dict, found: tuple[list[str], dict | None] | None
) -> tuple[str, str] | None:
    """(category, detail) when the rebuilt row is not the manifest's row, else None."""
    if found is None:
        return "no such row in the rebuild", "the rebuild has no row starting there"
    words = [word["word"] for word in row["words"]]
    if found[0] == words:
        return None
    if len(found[0]) != len(words):
        return "different word count", f"{len(words)} words here, {len(found[0])} rebuilt"
    index = next(i for i, (a, b) in enumerate(zip(words, found[0])) if a != b)
    return "different word text", f"word {index} is {words[index]!r}, rebuilt {found[0][index]!r}"


def row_groups(handle, transcoded: bool) -> Iterator[tuple[str, list[dict]]]:
    rows = (json.loads(line) for line in handle if line.strip())
    for name, group in itertools.groupby(rows, key=lambda row: recording_name(row, transcoded)):
        yield name, list(group)


def migrate_manifest(source: Path, target: Path, submit, window: int, args, stats: Stats) -> None:
    check_manifest_head(source, expected_from="build, diacritize or align_official_manifest")
    partial = target.with_name(f".{target.name}.migrating")
    target.parent.mkdir(parents=True, exist_ok=True)
    last_report = time.monotonic()
    started = last_report
    pending: deque[tuple[str, list[dict], Future | _Immediate]] = deque()

    def finish(output, name: str, rows: list[dict], future) -> None:
        expected = future.result()
        for row in rows:
            found = expected.get(row_key(row))
            reason = mismatch_reason(row, found)
            if reason is None:
                alt_words = found[1]
            elif args.on_mismatch == "skip":
                stats.mismatched[reason[0]] += 1
                alt_words = None
            else:
                raise SystemExit(
                    f"{source}: recording {name}, row at start {row_key(row)}: {reason[1]}.\n"
                    "The manifest was not built from this corpus with these options (or the "
                    "normalizer changed since). Pass the builder's flags it was made with, or "
                    "--on-mismatch skip."
                )
            stats.add(alt_words)
            output.write(json.dumps(with_alt_words(row, alt_words), ensure_ascii=False) + "\n")

    try:
        with (
            source.open(encoding="utf-8") as handle,
            partial.open("w", encoding="utf-8", newline="\n") as output,
        ):
            for name, rows in row_groups(handle, args.transcode_audio):
                pending.append((name, rows, submit(name)))
                if len(pending) >= window:
                    finish(output, *pending.popleft())
                now = time.monotonic()
                if now - last_report >= PROGRESS_SECONDS:
                    print(
                        f"progress : {source.name}: {stats.rows} rows, {now - started:.0f}s",
                        flush=True,
                    )
                    last_report = now
            while pending:
                finish(output, *pending.popleft())
        partial.replace(target)
    except BaseException:
        partial.unlink(missing_ok=True)
        raise


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    worker_args = argparse.Namespace(**{**vars(args), "transcode_audio": False})
    if args.workers == 1:
        _init_worker(worker_args)

        def submit(name: str):
            return _Immediate(expected_rows(name))

        pool = None
        window = 1
    else:
        pool = ProcessPoolExecutor(
            max_workers=args.workers, initializer=_init_worker, initargs=(worker_args,)
        )

        def submit(name: str):
            return pool.submit(expected_rows, name)

        window = args.workers * LOOKAHEAD_PER_WORKER
    try:
        totals = Stats()
        for manifest in args.manifest:
            stats = Stats()
            migrate_manifest(manifest, args.out_dir / manifest.name, submit, window, args, stats)
            print("\n".join(stats.lines(manifest.name)))
            totals.rows += stats.rows
            totals.with_spans += stats.with_spans
            totals.spans += stats.spans
            totals.kinds.update(stats.kinds)
            totals.mismatched.update(stats.mismatched)
        if len(args.manifest) > 1:
            print("\n".join(totals.lines("total")))
    finally:
        if pool is not None:
            pool.shutdown(cancel_futures=True)


if __name__ == "__main__":
    main()
