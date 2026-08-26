"""Download a subset of an ivrit-ai Knesset corpus into a directory-per-recording tree.

`ivrit-ai/knesset-plenums` and `ivrit-ai/knesset-committees` are 340-400 GiB each, and
nobody needs all of it: 1,000 usable hours is already past Kyutai's "strong model" line.
This selects a subset first and downloads only that, the way
training/scripts/prepare_data.py subsets HiFiTTS-2 before touching archive.org.

Selection reads `manifest.csv` from the repo root -- one 147 KiB file carrying
`quality_score`, `duration` and `segments_count` for all 1,551 plenum recordings -- so
choosing a subset costs one HTTP request rather than 1,551.

    python -m hebrew_training.fetch_knesset --hours 1200 --out data/knesset_plenums

Layout written, which is what build_official_manifest.py reads:

    <out>/<recording_id>/audio.m4a
                        /transcript.aligned.json
                        /metadata.json

Both repos are gated (one checkbox, auto-approved). Accept at
https://huggingface.co/datasets/ivrit-ai/knesset-plenums first, then `hf auth login`.

Resumable: a recording whose three files are all present is skipped.
"""

from __future__ import annotations

import argparse
import csv
import io
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

FILES = ("audio.m4a", "transcript.aligned.json", "metadata.json")

# Measured on ivrit-ai/knesset-plenums: 44.1 kHz stereo AAC at 128 kbps.
BYTES_PER_SECOND = 128_000 / 8


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--repo", default="ivrit-ai/knesset-plenums")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--hours", type=float, default=1200.0,
                        help="Budget in SPEECH hours (time inside transcript segments), not "
                             "wall-clock. Plenary audio is only ~62%% speech.")
    parser.add_argument("--min-quality", type=float, default=0.8,
                        help="Recording-level quality_score gate. The corpus median is 0.894; "
                             "0.8 keeps 1,331 of 1,551 recordings.")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--dry-run", action="store_true",
                        help="Print the selection and the download size, fetch nothing.")
    return parser.parse_args()


def number(row: dict, key: str) -> float | None:
    try:
        return float(row.get(key, ""))
    except (TypeError, ValueError):
        return None


def stable_bucket(key: str) -> float:
    """Knuth multiplicative hash, as in training/scripts/prepare_data.py.

    Deterministic and nested: the recordings chosen for 500 h are a subset of those
    chosen for 1200 h, so raising --hours later is an incremental download.
    """
    return ((int(key) if key.isdigit() else hash(key)) * 2654435761 & 0xFFFFFFFF) / 2**32


def select(repo: str, hours: float, min_quality: float) -> tuple[list[str], float, float]:
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(repo, "manifest.csv", repo_type="dataset")
    with open(path, encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    eligible = []
    for row in rows:
        quality = number(row, "quality_score")
        duration = number(row, "duration")
        recording = row.get("source_entry_id") or ""
        if quality is None or duration is None or not recording:
            continue
        if quality < min_quality:
            continue
        # Speech seconds, not wall-clock: what survives into manifest rows.
        speech = (number(row, "segments_count") or 0) * (number(row, "avg_segment_duration") or 0)
        eligible.append((recording, duration, speech))

    eligible.sort(key=lambda item: stable_bucket(item[0]))
    chosen: list[str] = []
    speech_total = wall_total = 0.0
    for recording, duration, speech in eligible:
        if speech_total / 3600 >= hours:
            break
        chosen.append(recording)
        speech_total += speech
        wall_total += duration
    print(
        f"{len(rows)} recordings in {repo}; {len(eligible)} at quality >= {min_quality}; "
        f"selected {len(chosen)}"
    )
    return chosen, speech_total / 3600, wall_total / 3600


def fetch(repo: str, recording: str, out: Path) -> str | None:
    from huggingface_hub import hf_hub_download

    target = out / recording
    if all((target / name).exists() for name in FILES):
        return None
    target.mkdir(parents=True, exist_ok=True)
    for name in FILES:
        if (target / name).exists():
            continue
        source = hf_hub_download(repo, f"{recording}/{name}", repo_type="dataset")
        # Copy rather than symlink: the HF cache is often on a different volume, and the
        # manifests written later hold absolute paths into this tree.
        (target / name).write_bytes(Path(source).read_bytes())
    return recording


def main() -> None:
    args = parse_args()
    chosen, speech_hours, wall_hours = select(args.repo, args.hours, args.min_quality)
    if not chosen:
        raise SystemExit("nothing selected -- lower --min-quality or raise --hours")

    gib = wall_hours * 3600 * BYTES_PER_SECOND / 2**30
    print(
        f"  {speech_hours:.0f} h of speech inside {wall_hours:.0f} h of audio, "
        f"about {gib:.0f} GiB to download"
    )
    if args.dry_run:
        return

    args.out.mkdir(parents=True, exist_ok=True)
    done = skipped = failed = 0
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(fetch, args.repo, r, args.out): r for r in chosen}
        for future in as_completed(futures):
            try:
                result = future.result()
            except Exception as exc:  # noqa: BLE001 -- one dead recording must not stop 1200 h
                failed += 1
                print(f"  failed {futures[future]}: {exc}", file=sys.stderr)
                continue
            if result is None:
                skipped += 1
            else:
                done += 1
            if (done + skipped + failed) % 25 == 0:
                print(f"  {done + skipped + failed}/{len(chosen)}  ({failed} failed)")

    print(f"downloaded {done}, already present {skipped}, failed {failed}")
    print(f"next: python -m hebrew_training.build_official_manifest --corpus {args.out} ...")


if __name__ == "__main__":
    main()
