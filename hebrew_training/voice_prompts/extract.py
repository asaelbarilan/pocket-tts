"""Cut 3-5 s voice prompts out of official manifests, only at natural seams.

A voice prompt is useful when it starts and ends like a recording of someone speaking, not like
an excerpt: no word clipped at either edge. The aligned manifests tell us where every word is,
so a *seam* is a gap between two consecutive words of at least `--min-gap-ms` (300 ms by
default). A cut goes in the middle of a seam, which leaves half the gap (>= 150 ms) between the
cut and the nearest word on each side, so word-timing error of that size still cannot clip a
word. A prompt is the audio between two seams that are `--min-duration`..`--max-duration` apart.

Where the seams come from, per row (`words` times are relative to the row's `start`):

* every gap between two consecutive words of the row that is >= the minimum;
* optionally (`--no-edge-cuts` turns it off) the gap before the first word / after the last
  word, when the row carries `lead_gap` / `tail_gap` (build_official_manifest: milliseconds to
  the neighbouring segment's speech) and it is >= the minimum. The audio there is outside the
  row but inside the recording, which is what lets a prompt start exactly at a row's first word.

Candidate windows are also checked for speech, so a prompt is not a long pause with a word on
each end: at least `--min-words` words, no gap between words longer than `--max-inner-gap-ms`,
and word time of at least `--min-speech-ratio` of the clip.

Selection is random and reproducible (`--seed`). Each row yields up to `--per-row` windows that
do not overlap, and each speaker keeps at most `--per-speaker` of its candidates (the speaker
key is `speaker`, else `user_id`, else the recording file; the Knesset corpora only have the
recording, so the cap there is per recording). The cap is applied to random priorities, so it
does not favour the start of a manifest. Rows whose word times are not measured
(`alignment_backend: original`, `timing: estimated`) are skipped: their times are not trustworthy
enough to cut on.

Output, next to each other:

    <output>/prompts/000000.wav ...     mono 16-bit WAV at --sample-rate (24 kHz by default)
    <output>/manifest.jsonl             one row per WAV, same numbering

Manifest row (times in seconds, gaps in milliseconds):

    id, audio (relative to <output>), dataset, split, stage, manifest, line (1-based, in that
    manifest), source_path, source_start / source_end (the cut, in the WAV at source_path),
    duration (of the WAV written), sample_rate, speaker / user_id / entry_id (when the row has
    them), text, num_words, gap_before_ms / gap_after_ms (silence between the clip edge and the
    first / last word's timestamps), seam_before_ms / seam_after_ms (the whole natural gap the
    cut was made in; the clip edge sits at its middle).
"""

from __future__ import annotations

import argparse
import heapq
import itertools
import json
import math
import os
import random
import shutil
import sys
from collections import Counter
from collections.abc import Callable, Iterator, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from pathlib import Path

from hebrew_training.data import read_jsonl
from hebrew_training.manifest_files import ALIGNED, BUILT, DIACRITIZED, SPLITS

STAGE_TEMPLATES = {"aligned": ALIGNED, "diac": DIACRITIZED, "built": BUILT}
PROMPTS_DIR = "prompts"
MANIFEST = "manifest.jsonl"
# Rows whose word times were not measured by an aligner: see ALIGNMENT.md.
UNMEASURED = ({"alignment_backend": "original"}, {"timing": "estimated"})


@dataclass
class Window:
    """One chosen prompt, before any audio is read."""

    dataset: str
    split: str
    stage: str
    manifest: str
    line: int
    source_path: str
    source_start: float
    source_end: float
    speaker: str | None
    user_id: str | None
    entry_id: str | None
    text: str
    num_words: int
    gap_before_ms: int
    gap_after_ms: int
    seam_before_ms: int
    seam_after_ms: int


# ---------------------------------------------------------------------------------------------
# Planning: where can a prompt be cut? No audio is read here.
# ---------------------------------------------------------------------------------------------


def usable_words(row: dict) -> list[tuple[float, float, str]] | None:
    """Absolute `(start, end, word)` for a row with measured word times, else None."""
    words = row.get("words")
    if not isinstance(words, list) or not words or not isinstance(row.get("path"), str):
        return None
    if any(row.get(key) == value for marks in UNMEASURED for key, value in marks.items()):
        return None
    try:
        origin = float(row["start"])
        timed = [
            (origin + float(w["start"]), origin + float(w["end"]), str(w["word"])) for w in words
        ]
    except (KeyError, TypeError, ValueError):
        return None
    previous_start = -math.inf
    for start, end, _ in timed:
        if (
            not (math.isfinite(start) and math.isfinite(end))
            or end < start
            or start < previous_start
        ):
            return None
        previous_start = start
    return timed


def find_seams(
    row: dict, timed: list[tuple[float, float, str]], min_gap: float, edge_cuts: bool
) -> list[tuple[float, float, int, int]]:
    """Seams as `(cut_time, gap_seconds, index_of_word_before, index_of_word_after)`.

    The edge seams use -1 and `len(timed)` as the missing word's index."""
    seams = []
    if edge_cuts:
        lead = row.get("lead_gap")
        if isinstance(lead, int | float) and lead / 1000 >= min_gap:
            gap = min(lead / 1000, timed[0][0])  # the recording does not start before 0
            if gap >= min_gap:
                seams.append((timed[0][0] - gap / 2, gap, -1, 0))
    for index in range(len(timed) - 1):
        gap = timed[index + 1][0] - timed[index][1]
        if gap >= min_gap:
            seams.append(((timed[index][1] + timed[index + 1][0]) / 2, gap, index, index + 1))
    if edge_cuts:
        tail = row.get("tail_gap")
        if isinstance(tail, int | float) and tail / 1000 >= min_gap:
            seams.append((timed[-1][1] + tail / 2000, tail / 1000, len(timed) - 1, len(timed)))
    return seams


def plan_row(
    row: dict,
    args: argparse.Namespace,
    rng: random.Random,
    avoid: Sequence[tuple[float, float]] = (),
) -> list[tuple[Window, float]]:
    """Up to `--per-row` non-overlapping windows for one row, each with a random priority.

    `avoid` lists spans of this recording that earlier runs already extracted."""
    timed = usable_words(row)
    if timed is None:
        return []
    min_gap = args.min_gap_ms / 1000
    seams = find_seams(row, timed, min_gap, args.edge_cuts)
    candidates = []
    for i, left in enumerate(seams):
        for right in seams[i + 1 :]:
            duration = right[0] - left[0]
            if duration > args.max_duration:
                break
            if duration < args.min_duration:
                continue
            first, last = left[3], right[2]  # words inside the clip, inclusive
            if last < first or last - first + 1 < args.min_words:
                continue
            inside = timed[first : last + 1]
            inner = max((b[0] - a[1] for a, b in itertools.pairwise(inside)), default=0.0)
            if inner * 1000 > args.max_inner_gap_ms:
                continue
            if sum(end - start for start, end, _ in inside) < args.min_speech_ratio * duration:
                continue
            candidates.append((left, right, first, last))
    rng.shuffle(candidates)

    chosen: list[tuple[float, float]] = list(avoid)
    plans = []
    for left, right, first, last in candidates:
        if len(plans) == args.per_row:
            break
        if any(left[0] < end and start < right[0] for start, end in chosen):
            continue
        chosen.append((left[0], right[0]))
        plans.append(
            (
                Window(
                    dataset="",
                    split="",
                    stage="",
                    manifest="",
                    line=0,
                    source_path=row["path"],
                    source_start=left[0],
                    source_end=right[0],
                    speaker=_text_or_none(row.get("speaker")),
                    user_id=_text_or_none(row.get("user_id")),
                    entry_id=_text_or_none(row.get("entry_id")),
                    text=" ".join(word for _, _, word in timed[first : last + 1]),
                    num_words=last - first + 1,
                    gap_before_ms=round((timed[first][0] - left[0]) * 1000),
                    gap_after_ms=round((right[0] - timed[last][1]) * 1000),
                    seam_before_ms=round(left[1] * 1000),
                    seam_after_ms=round(right[1] * 1000),
                ),
                rng.random(),
            )
        )
    return plans


def _text_or_none(value: object) -> str | None:
    return None if value is None or value == "" else str(value)


def speaker_key(window: Window | dict) -> str:
    """`speaker`, else `user_id`, else the recording file; for a Window or a manifest row."""
    get = window.get if isinstance(window, dict) else lambda key: getattr(window, key)
    return get("speaker") or get("user_id") or Path(get("source_path")).stem


class Cancelled(Exception):
    """The caller asked a running extraction to stop."""


@dataclass
class Existing:
    """What earlier runs left in the output folder, so that a new run adds to it."""

    next_id: int = 0
    per_speaker: Counter = field(default_factory=Counter)
    spans: dict[str, list[tuple[float, float]]] = field(default_factory=dict)
    sample_rates: set[int] = field(default_factory=set)


def load_existing(output: Path) -> Existing:
    existing = Existing()
    manifest = output / MANIFEST
    if not manifest.is_file():
        return existing
    for row in read_jsonl(manifest):
        existing.next_id = max(existing.next_id, int(row["id"]) + 1)
        existing.per_speaker[f"{row['dataset']}\0{speaker_key(row)}"] += 1
        existing.spans.setdefault(row["source_path"], []).append(
            (row["source_start"], row["source_end"])
        )
        existing.sample_rates.add(row["sample_rate"])
    return existing


def resolve_manifest(folder: Path, split: str, stage: str) -> tuple[Path, str] | None:
    names = list(STAGE_TEMPLATES) if stage == "auto" else [stage]
    for name in names:
        path = folder / STAGE_TEMPLATES[name].format(split=split)
        if path.is_file():
            return path, name
    return None


def parse_datasets(specs: list[str]) -> list[tuple[str, Path]]:
    datasets = []
    for spec in specs:
        name, separator, location = spec.partition("=")
        folder = Path(location if separator else spec).expanduser()
        if not separator:
            name = folder.resolve().name
        if not folder.is_dir():
            raise SystemExit(f"{folder} is not a directory")
        if any(name == taken for taken, _ in datasets):
            raise SystemExit(f"two datasets are called {name!r}; name one with NAME=DIR")
        datasets.append((name, folder))
    return datasets


def plan(
    args: argparse.Namespace,
    datasets: list[tuple[str, Path]],
    existing: Existing | None = None,
    progress: Callable[[str, int, int | None], None] | None = None,
    should_stop: Callable[[], bool] | None = None,
) -> tuple[list[Window], Counter]:
    """Scan every manifest once; keep each speaker's `--per-speaker` lowest-priority windows.

    With `existing`, spans already extracted are avoided and the per-speaker cap counts what is
    there. `progress(phase, done, total)` and `should_stop()` serve a caller that shows a bar."""
    existing = existing or Existing()
    stats: Counter = Counter()
    heaps: dict[str, list] = {}
    counter = 0
    for name, folder in datasets:
        found = False
        for split in args.splits:
            resolved = resolve_manifest(folder, split, args.stage)
            if resolved is None:
                print(f"[{name}] no {split} manifest ({args.stage}); skipping", file=sys.stderr)
                continue
            found = True
            manifest, stage = resolved
            print(f"[{name}] scanning {manifest} ({stage})", file=sys.stderr)
            for line, row in enumerate(_rows(manifest), 1):
                stats["rows"] += 1
                if stats["rows"] % 2000 == 0:
                    if should_stop and should_stop():
                        raise Cancelled
                    if progress:
                        progress(f"scanning {name}/{split}", stats["rows"], None)
                if usable_words(row) is None:
                    stats["rows_unusable"] += 1
                    continue
                rng = random.Random(f"{args.seed}:{name}:{row['path']}:{row['start']}")
                audio = _resolve_audio(row["path"], folder)
                plans = plan_row(row, args, rng, existing.spans.get(audio, ()))
                if not plans:
                    stats["rows_without_window"] += 1
                for window, priority in plans:
                    window.dataset, window.split, window.stage = name, split, stage
                    window.manifest, window.line = str(manifest), line
                    window.source_path = audio
                    stats["windows"] += 1
                    counter += 1
                    key = f"{name}\0{speaker_key(window)}"
                    room = (
                        args.per_speaker - existing.per_speaker[key] if args.per_speaker else None
                    )
                    if room is not None and room <= 0:
                        stats["windows_over_speaker_cap"] += 1
                        continue
                    heap = heaps.setdefault(key, [])
                    heapq.heappush(heap, (-priority, counter, window))
                    if room is not None and len(heap) > room:
                        heapq.heappop(heap)
        if not found:
            raise SystemExit(f"{folder} has none of the requested manifests")
    stats["speakers"] = len(heaps)
    kept = sorted(
        (
            (-negative, number, window)
            for heap in heaps.values()
            for negative, number, window in heap
        ),
        key=lambda item: item[:2],
    )
    if args.max_prompts:
        kept = kept[: args.max_prompts]
    stats["selected"] = len(kept)
    return [window for _, _, window in kept], stats


def _rows(manifest: Path) -> Iterator[dict]:
    for row in read_jsonl(manifest):
        yield row if isinstance(row, dict) else {}


def _resolve_audio(path: str, folder: Path) -> str:
    if os.path.isabs(path) or Path(path).exists():
        return path
    return str(folder / path)


# ---------------------------------------------------------------------------------------------
# Extraction: read the audio for the chosen windows.
# ---------------------------------------------------------------------------------------------


def read_window(window: Window, sample_rate: int):
    """Mono float32 samples at `sample_rate` for the window, or None when the source is short."""
    import numpy as np
    import soundfile as sf
    from scipy.signal import resample_poly

    with sf.SoundFile(window.source_path) as source:
        rate = source.samplerate
        first = max(0, round(window.source_start * rate))
        last = round(window.source_end * rate)
        source.seek(first)
        audio = source.read(last - first, dtype="float32", always_2d=True)
    if audio.shape[0] < last - first:
        return None  # the manifest points past the end of the file
    mono = audio.mean(axis=1)
    if rate != sample_rate:
        divisor = math.gcd(rate, sample_rate)
        mono = resample_poly(mono, sample_rate // divisor, rate // divisor).astype(np.float32)
    return np.clip(mono, -1.0, 1.0), first / rate, last / rate


def extract(
    args: argparse.Namespace,
    windows: list[Window],
    stats: Counter,
    next_id: int = 0,
    progress: Callable[[str, int, int | None], None] | None = None,
    should_stop: Callable[[], bool] | None = None,
) -> None:
    """Write the WAVs and append their rows to the manifest, numbering from `next_id`.

    A row is written after its WAV, and each chunk is flushed, so a stopped or failed run leaves a
    manifest that is consistent with the files."""
    import soundfile as sf

    output = Path(args.output)
    prompts = output / PROMPTS_DIR
    prompts.mkdir(parents=True, exist_ok=True)

    def work(window: Window):
        try:
            return read_window(window, args.sample_rate)
        except (OSError, RuntimeError) as error:  # soundfile.LibsndfileError is a RuntimeError
            return error

    written = 0
    with (
        (output / MANIFEST).open("a", encoding="utf-8", newline="\n") as handle,
        ThreadPoolExecutor(args.workers) as pool,
    ):
        step = max(1, args.workers * 8)
        for offset in range(0, len(windows), step):
            if should_stop and should_stop():
                raise Cancelled
            chunk = windows[offset : offset + step]
            for window, result in zip(chunk, pool.map(work, chunk), strict=True):
                if result is None or isinstance(result, Exception):
                    stats["audio_failed"] += 1
                    if isinstance(result, Exception) and stats["audio_failed"] <= 5:
                        print(f"skipped {window.source_path}: {result}", file=sys.stderr)
                    continue
                samples, start, end = result
                name = f"{next_id:06d}.wav"
                sf.write(prompts / name, samples, args.sample_rate, subtype="PCM_16")
                row = {"id": next_id, "audio": f"{PROMPTS_DIR}/{name}"}
                row.update(asdict(window))
                row["source_start"], row["source_end"] = round(start, 4), round(end, 4)
                row["duration"] = round(len(samples) / args.sample_rate, 4)
                row["sample_rate"] = args.sample_rate
                row = {key: value for key, value in row.items() if value is not None}
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                next_id += 1
                written += 1
                stats["written"] = written
            handle.flush()
            if progress:
                progress("extracting", min(offset + step, len(windows)), len(windows))
            else:
                print(f"  {written}/{len(windows)} written", file=sys.stderr, end="\r")
    if not progress:
        print(file=sys.stderr)


# ---------------------------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------------------------


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "datasets",
        nargs="+",
        metavar="[NAME=]DIR",
        help="official manifest folders (train*.jsonl / valid*.jsonl); NAME defaults to the folder",
    )
    parser.add_argument("--output", required=True, help="folder for prompts/ and manifest.jsonl")
    parser.add_argument("--splits", nargs="+", choices=SPLITS, default=list(SPLITS))
    parser.add_argument(
        "--stage",
        choices=["auto", *STAGE_TEMPLATES],
        default="auto",
        help="manifest stage to read; auto takes aligned, else diac, else built",
    )
    parser.add_argument("--min-duration", type=float, default=3.0)
    parser.add_argument("--max-duration", type=float, default=5.0)
    parser.add_argument("--min-gap-ms", type=float, default=300.0, help="smallest seam")
    parser.add_argument(
        "--edge-cuts",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="also cut in the lead_gap / tail_gap beside a row",
    )
    parser.add_argument("--min-words", type=int, default=3)
    parser.add_argument("--max-inner-gap-ms", type=float, default=1000.0)
    parser.add_argument("--min-speech-ratio", type=float, default=0.6)
    parser.add_argument("--per-row", type=int, default=1)
    parser.add_argument("--per-speaker", type=int, default=10, help="0 for no cap")
    parser.add_argument("--max-prompts", type=int, default=0, help="0 for no limit")
    parser.add_argument("--sample-rate", type=int, default=24_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 1))
    parser.add_argument(
        "--append",
        action="store_true",
        help="add to an earlier extraction: continue its numbering, avoid its spans, and count "
        "its prompts against --per-speaker (--max-prompts then limits this run's new prompts)",
    )
    parser.add_argument("--overwrite", action="store_true", help="replace an earlier extraction")


def check_arguments(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    if not 0 < args.min_duration <= args.max_duration:
        parser.error("need 0 < --min-duration <= --max-duration")
    if args.min_gap_ms <= 0 or args.per_row < 1 or args.workers < 1 or args.sample_rate < 8000:
        parser.error("--min-gap-ms, --per-row, --workers and --sample-rate must be positive")


def run(
    args: argparse.Namespace,
    progress: Callable[[str, int, int | None], None] | None = None,
    should_stop: Callable[[], bool] | None = None,
) -> Counter:
    output = Path(args.output).expanduser()
    args.output = str(output)
    if args.append and args.overwrite:
        raise SystemExit("--append and --overwrite do not combine")
    if (output / MANIFEST).exists() or (output / PROMPTS_DIR).exists():
        if args.overwrite:
            shutil.rmtree(output / PROMPTS_DIR, ignore_errors=True)
            (output / MANIFEST).unlink(missing_ok=True)
        elif not args.append:
            raise SystemExit(
                f"{output} already has an extraction; pass --append to add to it or "
                "--overwrite to replace it"
            )
    existing = load_existing(output)
    if existing.sample_rates - {args.sample_rate}:
        raise SystemExit(
            f"{output} holds {sorted(existing.sample_rates)} Hz prompts; --sample-rate "
            f"{args.sample_rate} would mix rates"
        )
    datasets = parse_datasets(args.datasets)
    windows, stats = plan(args, datasets, existing, progress, should_stop)
    if not windows:
        raise SystemExit(f"no new prompt window found: {dict(stats)}")
    random.Random(args.seed).shuffle(windows)  # numbering must not follow dataset order
    extract(args, windows, stats, existing.next_id, progress, should_stop)
    return stats


def report(stats: Counter, args: argparse.Namespace) -> None:
    for key in ("rows", "rows_unusable", "rows_without_window", "windows", "speakers", "selected"):
        print(f"{key:>20}: {stats[key]}")
    print(f"{'audio_failed':>20}: {stats['audio_failed']}")
    print(f"{'written':>20}: {stats['written']}  -> {args.output}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    add_arguments(parser)
    args = parser.parse_args(argv)
    check_arguments(parser, args)
    report(run(args), args)


if __name__ == "__main__":
    main()
