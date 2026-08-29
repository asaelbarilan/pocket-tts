from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

from hebrew_training.data import normalize_hebrew_text, write_jsonl


def _stable_bucket(key: str) -> float:
    """Deterministic [0, 1) bucket for a key, stable across runs and machines."""
    digest = hashlib.sha256(key.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") / 2**64


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Convert a hebrew-tts-data-tools dataset into Pocket TTS manifests with a "
            "speaker-grouped split and cross-clip prompt pairing."
        )
    )
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--validation-hours",
        type=float,
        default=1.2,
        help="Target validation size in hours. Speakers are assigned whole.",
    )
    parser.add_argument(
        "--prompt-seconds",
        type=float,
        default=3.0,
        help="Prompt clips shorter than this are used only as a fallback.",
    )
    parser.add_argument(
        "--max-speaker-share",
        type=float,
        default=0.15,
        help=(
            "Cap on any single speaker's share of the validation budget. Forces breadth "
            "so the metric is not dominated by one voice."
        ),
    )
    parser.add_argument("--min-seconds", type=float, default=4.0)
    parser.add_argument("--max-seconds", type=float, default=16.0)
    return parser.parse_args()


def load_rows(dataset_dir: Path, wavs_dir: Path) -> list[dict]:
    """Materialize dataset audio to wav files and return manifest rows."""
    from datasets import load_from_disk

    dataset = load_from_disk(str(dataset_dir))
    wavs_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    for index, record in enumerate(dataset):
        metadata = record["metadata"]
        audio_path = wavs_dir / f"{metadata['entry_id']}_{index:06d}.wav"
        if not audio_path.exists():
            audio_path.write_bytes(record["audio"]["bytes"])
        rows.append(
            {
                "audio_path": str(audio_path.resolve()),
                "text": normalize_hebrew_text(str(record["text"])),
                "duration": float(metadata["duration"]),
                # source_id stays the recording so existing tooling keeps working,
                # but speaker_id is the key that governs splitting and pairing.
                "source_id": str(metadata["entry_id"]),
                "speaker_id": str(metadata["user_id"]),
                "quality_score": float(metadata["quality_score"]),
            }
        )
    return rows


def assign_validation_speakers(
    rows: list[dict], validation_hours: float, max_speaker_share: float = 0.15
) -> set[str]:
    """
    Pick whole speakers for validation, deterministically.

    Speakers are walked in a stable hashed order and taken until the hours target is met.

    No single speaker may contribute more than `max_speaker_share` of the budget. Without
    that cap, a hashed walk over this corpus picks a couple of large speakers and stops:
    the first attempt produced a 2-speaker validation set where one voice held 552 of 615
    clips, which measures that one person rather than generalization to unseen speakers.
    The corpus is heavily concentrated (top 10 speakers hold ~48% of hours), so breadth
    has to be forced rather than hoped for.

    If the cap is so tight that the budget cannot be met, it is relaxed step by step
    rather than silently returning a short validation set.
    """
    speaker_seconds: dict[str, float] = defaultdict(float)
    for row in rows:
        speaker_seconds[row["speaker_id"]] += row["duration"]

    target_seconds = validation_hours * 3600
    ordered = sorted(speaker_seconds, key=_stable_bucket)

    validation_speakers: set[str] = set()
    for share in (max_speaker_share, 0.25, 0.5, 1.5):
        if share < max_speaker_share:
            continue
        cap = target_seconds * share
        validation_speakers = set()
        accumulated = 0.0
        for speaker in ordered:
            if accumulated >= target_seconds:
                break
            if speaker_seconds[speaker] > cap:
                continue
            if accumulated + speaker_seconds[speaker] > target_seconds * 1.5:
                continue
            validation_speakers.add(speaker)
            accumulated += speaker_seconds[speaker]
        if accumulated >= target_seconds * 0.9:
            break

    if not validation_speakers:
        raise RuntimeError("No speaker fit the validation budget. Raise --validation-hours.")
    return validation_speakers


def pair_prompts(rows: list[dict], prompt_seconds: float) -> tuple[list[dict], Counter]:
    """
    Give every row a prompt clip recorded by the same speaker but from a different clip.

    Preference order:
      1. a clip from a different recording by that speaker, long enough for the prompt
      2. any other clip by that speaker that is long enough
      3. any other clip by that speaker
    Rows whose speaker has no second clip are dropped; they cannot be prompted without
    reintroducing the same-clip leakage this function exists to remove.
    """
    by_speaker: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_speaker[row["speaker_id"]].append(row)

    paired: list[dict] = []
    dropped = Counter()
    for speaker, speaker_rows in by_speaker.items():
        if len(speaker_rows) < 2:
            dropped["speaker_has_one_clip"] += len(speaker_rows)
            continue

        for row in speaker_rows:
            others = [
                candidate
                for candidate in speaker_rows
                if candidate["audio_path"] != row["audio_path"]
            ]
            long_enough = [c for c in others if c["duration"] >= prompt_seconds]
            cross_recording = [c for c in long_enough if c["source_id"] != row["source_id"]]
            pool = cross_recording or long_enough or others

            # Deterministic choice, but varied across rows.
            choice = pool[int(_stable_bucket(row["audio_path"]) * len(pool)) % len(pool)]
            paired_row = dict(row)
            paired_row["prompt_audio_path"] = choice["audio_path"]
            paired_row["prompt_duration"] = choice["duration"]
            paired_row["prompt_is_cross_recording"] = choice["source_id"] != row["source_id"]
            paired.append(paired_row)

    return paired, dropped


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    rows = load_rows(args.dataset, args.output_dir / "wavs")

    skipped = Counter()
    kept: list[dict] = []
    for row in rows:
        if not row["text"]:
            skipped["empty_text"] += 1
        elif not any("֐" <= char <= "׿" for char in row["text"]):
            skipped["no_hebrew"] += 1
        elif not args.min_seconds <= row["duration"] <= args.max_seconds:
            skipped["duration"] += 1
        elif not row["speaker_id"]:
            skipped["missing_speaker_id"] += 1
        else:
            kept.append(row)

    paired, dropped = pair_prompts(kept, args.prompt_seconds)
    validation_speakers = assign_validation_speakers(
        paired, args.validation_hours, args.max_speaker_share
    )

    train = [r for r in paired if r["speaker_id"] not in validation_speakers]
    validation = [r for r in paired if r["speaker_id"] in validation_speakers]

    # A prompt must never cross the split boundary, or validation leaks into training.
    train_speakers = {r["speaker_id"] for r in train}
    assert not (train_speakers & validation_speakers), "speaker leaked across split"

    write_jsonl(args.output_dir / "train.jsonl", train)
    write_jsonl(args.output_dir / "validation.jsonl", validation)

    with (args.output_dir / "tokenizer_corpus.txt").open(
        "w", encoding="utf-8", newline="\n"
    ) as handle:
        for row in train:
            handle.write(row["text"] + "\n")

    stats = {
        "train_samples": len(train),
        "validation_samples": len(validation),
        "train_hours": round(sum(r["duration"] for r in train) / 3600, 3),
        "validation_hours": round(sum(r["duration"] for r in validation) / 3600, 3),
        "train_speakers": len(train_speakers),
        "validation_speakers": len(validation_speakers),
        "train_recordings": len({r["source_id"] for r in train}),
        "validation_recordings": len({r["source_id"] for r in validation}),
        # Largest single-speaker share of validation. If this creeps back toward 1.0 the
        # metric is measuring one voice again.
        "validation_largest_speaker_share": round(
            max(
                sum(r["duration"] for r in validation if r["speaker_id"] == s)
                for s in validation_speakers
            )
            / max(sum(r["duration"] for r in validation), 1e-9),
            3,
        ),
        "cross_recording_prompts": sum(1 for r in paired if r["prompt_is_cross_recording"]),
        "same_recording_prompts": sum(1 for r in paired if not r["prompt_is_cross_recording"]),
        "skipped": dict(skipped),
        "dropped": dict(dropped),
        "dataset": str(args.dataset.resolve()),
    }
    (args.output_dir / "stats.json").write_text(
        json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
