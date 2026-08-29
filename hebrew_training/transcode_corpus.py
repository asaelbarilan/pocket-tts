"""Transcode a corpus to 24 kHz mono wav and repoint a manifest at it.

This exists because reading a window out of a long m4a is catastrophically slow, and it is
the single largest cost in training on the Knesset corpora.

`training/dataloader.py` reads two windows per sample -- the voice prompt and the target.
For a wav, `sphn.read(path, start_sec=...)` is byte arithmetic over PCM and costs the same
wherever it lands. For AAC-in-MP4 there is no such shortcut: an exact seek has to decode
forward from the previous keyframe, so the cost grows with the offset. Measured on one
51-minute plenum file, a 5-second window:

    offset      0 s     300 s    1500 s    3000 s
    m4a        54 ms    168 ms    634 ms   1229 ms      (exact seek, what a decoder must do)
    wav       0.5 ms    0.5 ms    0.5 ms    0.5 ms      (sphn, O(1))

Plenum recordings average 4.66 h and reach 22 h, so real offsets are far past the right of
that table. At batch_size 64 that is 128 reads per step per GPU, and the run stalls at tens
of seconds per step while the GPUs idle.

24 kHz mono is not a compromise: Mimi runs at 24 kHz and the loader downmixes to mono, so
this is exactly what training consumes. Budget ~165 MB per audio-hour.

    python -m hebrew_training.transcode_corpus \
        --manifest data/hebrew_official/train_aligned.jsonl \
        --out-dir data/wav24 \
        --out-manifest data/hebrew_official/train_aligned_wav.jsonl

Point the training config's train_jsonl/valid_jsonl at the new manifests. `start` and
`duration` are unchanged -- only `path` moves -- so alignments stay valid.

Resumable: an existing output of non-zero size is skipped, so this can be interrupted.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

SAMPLE_RATE = 24000


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "--manifest",
        type=Path,
        required=True,
        action="append",
        help="Repeat for train and valid; they share the audio cache.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        required=True,
        help="Where the wavs go. Needs ~165 MB per audio-hour.",
    )
    parser.add_argument(
        "--out-manifest",
        type=Path,
        action="append",
        help="One per --manifest, in order. Defaults to <manifest>_wav.jsonl",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=8,
        help="Parallel ffmpeg processes. Each is roughly one core.",
    )
    parser.add_argument("--sample-rate", type=int, default=SAMPLE_RATE)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Report how many files and how much disk, transcode nothing.",
    )
    return parser.parse_args()


def output_for(source: Path, out_dir: Path) -> Path:
    """A flat, collision-free name: recordings from different corpora can share a basename."""
    digest = hashlib.sha256(str(source).encode("utf-8")).hexdigest()[:12]
    return out_dir / f"{source.stem}_{digest}.wav"


def probe_seconds(path: Path) -> float:
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(path),
        ],
        capture_output=True,
        text=True,
    )
    try:
        return float(result.stdout.strip())
    except ValueError:
        return 0.0


def transcode(source: Path, target: Path, sample_rate: int) -> tuple[Path, str | None]:
    if target.exists() and target.stat().st_size > 0:
        return source, None
    # Write to a partial file first: an interrupted run must not leave a truncated wav that
    # the resume check would then accept as complete. The name keeps the .wav suffix -- with
    # only ".part" ffmpeg cannot infer the container and fails with "Invalid argument" -- and
    # -f wav states it outright.
    partial = target.with_name(target.name + ".part.wav")
    result = subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-y",
            "-i",
            str(source),
            "-ar",
            str(sample_rate),
            "-ac",
            "1",
            "-c:a",
            "pcm_s16le",
            "-f",
            "wav",
            str(partial),
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0 or not partial.exists():
        partial.unlink(missing_ok=True)
        return source, (result.stderr or "ffmpeg failed").strip().splitlines()[-1:][0]
    partial.replace(target)
    return source, None


def main() -> None:
    args = parse_args()
    if shutil.which("ffmpeg") is None:
        raise SystemExit("ffmpeg is not on PATH; it does the decoding")
    outputs = args.out_manifest or [m.with_name(m.stem + "_wav.jsonl") for m in args.manifest]
    if len(outputs) != len(args.manifest):
        raise SystemExit("--out-manifest must be given once per --manifest, in the same order")

    rows_by_manifest = []
    sources: dict[Path, None] = {}
    for manifest in args.manifest:
        rows = []
        with manifest.open(encoding="utf-8") as handle:
            for line in handle:
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                rows.append(row)
                sources.setdefault(Path(row["path"]), None)
        rows_by_manifest.append(rows)
        print(f"{manifest}: {len(rows)} rows")

    todo = [s for s in sources if not output_for(s, args.out_dir).exists()]
    print(f"{len(sources)} distinct recordings, {len(todo)} still to transcode")
    if args.dry_run:
        seconds = sum(probe_seconds(s) for s in list(todo)[:20])
        if seconds:
            per_hour_mb = args.sample_rate * 2 * 3600 / 2**20
            hours = seconds / 3600 * len(todo) / min(20, len(todo))
            print(f"  ~{hours:.0f} audio-hours estimated -> ~{hours * per_hour_mb / 1024:.1f} TB")
        return

    args.out_dir.mkdir(parents=True, exist_ok=True)
    failed = 0
    if todo:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = {
                pool.submit(transcode, s, output_for(s, args.out_dir), args.sample_rate): s
                for s in todo
            }
            for index, future in enumerate(as_completed(futures), 1):
                _, error = future.result()
                if error:
                    failed += 1
                    print(f"  failed {futures[future].name}: {error}", file=sys.stderr)
                if index % 25 == 0 or index == len(todo):
                    print(f"  {index}/{len(todo)} ({failed} failed)", flush=True)

    # Rewrite paths. start/duration/words are untouched: the transcode preserves the
    # timeline exactly, so every alignment stays valid.
    for manifest, rows, output in zip(args.manifest, rows_by_manifest, outputs, strict=True):
        kept = 0
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("w", encoding="utf-8", newline="\n") as handle:
            for row in rows:
                target = output_for(Path(row["path"]), args.out_dir)
                if not (target.exists() and target.stat().st_size > 0):
                    continue  # its recording failed to transcode
                handle.write(
                    json.dumps({**row, "path": str(target.resolve())}, ensure_ascii=False) + "\n"
                )
                kept += 1
        dropped = len(rows) - kept
        print(f"wrote {output}: {kept} rows" + (f", {dropped} dropped" if dropped else ""))

    if failed:
        print(f"\n{failed} recordings failed to transcode; their rows were dropped.")
    print("\nPoint train_jsonl/valid_jsonl at the new manifests.")


if __name__ == "__main__":
    main()
