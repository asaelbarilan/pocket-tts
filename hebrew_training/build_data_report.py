from __future__ import annotations

import argparse
import json
import random
import re
import statistics
from collections import Counter
from pathlib import Path

# Summarise a prepared dataset into one JSON the data dashboard reads. Kept separate from
# the page so the expensive pass over the manifests happens once, not on every page load.


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build data_report.json for the dashboard.")
    parser.add_argument("--artifacts", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--examples", type=int, default=40)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def histogram(values: list[float], bins: int, lo: float, hi: float) -> list[dict]:
    if hi <= lo:
        return []
    width = (hi - lo) / bins
    counts = [0] * bins
    for v in values:
        i = int((v - lo) / width)
        counts[min(max(i, 0), bins - 1)] += 1
    return [
        {"lo": round(lo + i * width, 2), "hi": round(lo + (i + 1) * width, 2), "n": c}
        for i, c in enumerate(counts)
    ]


def summarise(path: Path, example_count: int, rng: random.Random) -> dict | None:
    train = path / "train.jsonl"
    val = path / "validation.jsonl"
    if not train.exists():
        return None
    rows = {}
    for split, f in (("train", train), ("validation", val)):
        if f.exists():
            rows[split] = [json.loads(line) for line in f.open(encoding="utf-8")]
    everything = [r for rs in rows.values() for r in rs]
    if not everything:
        return None

    durations = sorted(r["duration"] for r in everything)
    texts = [r["text"] for r in everything]
    lengths = sorted(len(t) for t in texts)
    quality = sorted(r.get("quality_score", 0.0) for r in everything)
    speakers = Counter(r["speaker_id"] for r in everything)
    recordings = {r["source_id"] for r in everything}

    def pct(sorted_values, p):
        return sorted_values[int(p * (len(sorted_values) - 1))]

    # Audio only exists if the wavs were not cleaned up; the page hides players without it.
    wavs_present = (path / "wavs").is_dir()
    sample = rng.sample(everything, min(example_count, len(everything)))
    examples = [
        {
            "text": r["text"],
            "duration": round(r["duration"], 2),
            "speaker": r["speaker_id"][:8],
            "quality": round(r.get("quality_score", 0.0), 3),
            "audio": (
                f"artifacts/{path.name}/wavs/{Path(r['audio_path']).name}" if wavs_present else None
            ),
            "prompt_audio": (
                f"artifacts/{path.name}/wavs/{Path(r['prompt_audio_path']).name}"
                if wavs_present and r.get("prompt_audio_path")
                else None
            ),
        }
        for r in sample
    ]

    return {
        "name": path.name,
        "wavs_present": wavs_present,
        "clips": len(everything),
        "hours": round(sum(durations) / 3600, 2),
        "train_clips": len(rows.get("train", [])),
        "validation_clips": len(rows.get("validation", [])),
        "train_speakers": len({r["speaker_id"] for r in rows.get("train", [])}),
        "validation_speakers": len({r["speaker_id"] for r in rows.get("validation", [])}),
        "recordings": len(recordings),
        "duration": {
            "mean": round(statistics.mean(durations), 2),
            "median": round(statistics.median(durations), 2),
            "min": round(min(durations), 2),
            "max": round(max(durations), 2),
            "p90": round(pct(durations, 0.9), 2),
            "hist": histogram(durations, 30, min(durations), max(durations)),
        },
        "text_length": {
            "mean": round(statistics.mean(lengths), 1),
            "median": statistics.median(lengths),
            "max": max(lengths),
            "hist": histogram([float(x) for x in lengths], 30, 0, max(lengths)),
        },
        "quality": {
            "mean": round(statistics.mean(quality), 3),
            "median": round(statistics.median(quality), 3),
            "min": round(min(quality), 3),
            "p10": round(pct(quality, 0.1), 3),
            "hist": histogram(quality, 20, min(quality), max(quality)),
        },
        "speakers": {
            "count": len(speakers),
            "top": [{"id": s[:8], "clips": n} for s, n in speakers.most_common(15)],
            "clips_per_speaker_median": statistics.median(speakers.values()),
        },
        "text_flags": {
            "with_digits": sum(1 for t in texts if re.search(r"[0-9]", t)),
            "with_latin": sum(1 for t in texts if re.search(r"[A-Za-z]", t)),
            "empty": sum(1 for t in texts if not t.strip()),
            "total": len(texts),
        },
        "prompt_pairing": {
            "cross_recording": sum(1 for r in everything if r.get("prompt_is_cross_recording")),
            "same_recording": sum(
                1
                for r in everything
                if r.get("prompt_audio_path") and not r.get("prompt_is_cross_recording")
            ),
        },
        "examples": examples,
    }


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)
    datasets = [d for d in (summarise(p, args.examples, rng) for p in args.artifacts) if d]
    if not datasets:
        raise SystemExit("no datasets with a train.jsonl were found")
    args.output.write_text(
        json.dumps({"datasets": datasets}, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    for d in datasets:
        print(
            f"{d['name']}: {d['clips']} clips, {d['hours']} h, "
            f"{d['speakers']['count']} speakers, wavs={d['wavs_present']}"
        )
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
