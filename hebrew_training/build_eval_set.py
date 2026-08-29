from __future__ import annotations

import argparse
import json
import random
import re
import unicodedata
from collections import defaultdict
from pathlib import Path

_WS = re.compile(r"\s+")
_LATIN_OR_DIGIT = re.compile(r"[A-Za-z0-9]")
_HEBREW = re.compile(r"[\u0590-\u05ff]")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build fixed controlled Hebrew evaluation groups.")
    parser.add_argument("--artifacts", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sentences-per-group", type=int, default=8)
    parser.add_argument("--speakers-per-group", type=int, default=4)
    parser.add_argument("--min-chars", type=int, default=45)
    parser.add_argument("--max-chars", type=int, default=110)
    parser.add_argument("--prompt-min-seconds", type=float, default=4.0)
    parser.add_argument("--prompt-max-seconds", type=float, default=10.0)
    parser.add_argument(
        "--asr-floor-clips", type=int, default=0, help="0 matches the total generated clip count."
    )
    parser.add_argument("--seed", type=int, default=1234)
    return parser.parse_args()


def clean(text: str) -> str:
    return _WS.sub(" ", unicodedata.normalize("NFC", text)).strip()


def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(path)
    return [json.loads(line) for line in path.open(encoding="utf-8")]


def is_usable_text(text: str, min_chars: int, max_chars: int) -> bool:
    return (
        min_chars <= len(text) <= max_chars
        and _HEBREW.search(text) is not None
        and _LATIN_OR_DIGIT.search(text) is None
    )


def choose_prompts(
    rows: list[dict],
    speaker_ids: set[str],
    count: int,
    minimum_seconds: float,
    maximum_seconds: float,
) -> list[dict]:
    best: dict[str, dict] = {}
    for row in rows:
        speaker = row["speaker_id"]
        audio_path = Path(row["audio_path"])
        if (
            speaker not in speaker_ids
            or not minimum_seconds <= row["duration"] <= maximum_seconds
            or not audio_path.exists()
        ):
            continue
        if speaker not in best or abs(row["duration"] - 6.0) < abs(best[speaker]["duration"] - 6.0):
            best[speaker] = row
    selected = sorted(
        best.values(), key=lambda row: (abs(row["duration"] - 6.0), row["speaker_id"])
    )[:count]
    if len(selected) < count:
        raise ValueError(f"only {len(selected)} of {count} requested speakers have usable prompts")
    return [
        {
            "speaker_id": row["speaker_id"],
            "prompt_audio_path": row["audio_path"],
            "prompt_duration": round(row["duration"], 3),
        }
        for row in selected
    ]


def choose_texts(
    rows: list[dict],
    count: int,
    min_chars: int,
    max_chars: int,
    rng: random.Random,
    *,
    reject_blob: str | None = None,
) -> list[str]:
    candidates = {
        clean(row["text"])
        for row in rows
        if is_usable_text(clean(row["text"]), min_chars, max_chars)
    }
    if reject_blob is not None:
        candidates = {text for text in candidates if text not in reject_blob}
    candidates = sorted(candidates)
    rng.shuffle(candidates)
    if len(candidates) < count:
        raise ValueError(f"only {len(candidates)} of {count} requested texts are usable")
    return sorted(candidates[:count])


def choose_asr_floor(rows: list[dict], count: int, rng: random.Random) -> list[dict]:
    by_speaker: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        if Path(row["audio_path"]).exists() and clean(row["text"]):
            by_speaker[row["speaker_id"]].append(row)
    for speaker_rows in by_speaker.values():
        rng.shuffle(speaker_rows)
    speakers = sorted(by_speaker)
    selected = []
    offset = 0
    while len(selected) < count:
        added = False
        for speaker in speakers:
            if offset >= len(by_speaker[speaker]):
                continue
            row = by_speaker[speaker][offset]
            selected.append(
                {"speaker_id": speaker, "audio_path": row["audio_path"], "text": clean(row["text"])}
            )
            added = True
            if len(selected) == count:
                break
        if not added:
            break
        offset += 1
    if len(selected) < count:
        raise ValueError(f"only {len(selected)} of {count} requested ASR-floor clips exist")
    return selected


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)
    train_rows = []
    validation_rows = []
    for artifact in args.artifacts:
        train_rows.extend(load_jsonl(artifact / "train.jsonl"))
        validation_rows.extend(load_jsonl(artifact / "validation.jsonl"))

    train_speakers = {row["speaker_id"] for row in train_rows}
    validation_speakers = {row["speaker_id"] for row in validation_rows}
    unseen_speakers = validation_speakers - train_speakers
    if train_speakers & unseen_speakers:
        raise AssertionError("speaker leakage in controlled evaluation split")

    seen_prompts = choose_prompts(
        train_rows,
        train_speakers,
        args.speakers_per_group,
        args.prompt_min_seconds,
        args.prompt_max_seconds,
    )
    unseen_prompts = choose_prompts(
        validation_rows,
        unseen_speakers,
        args.speakers_per_group,
        args.prompt_min_seconds,
        args.prompt_max_seconds,
    )
    train_blob = "\n".join(clean(row["text"]) for row in train_rows)
    seen_texts = choose_texts(
        train_rows, args.sentences_per_group, args.min_chars, args.max_chars, rng
    )
    unseen_texts = choose_texts(
        validation_rows,
        args.sentences_per_group,
        args.min_chars,
        args.max_chars,
        rng,
        reject_blob=train_blob,
    )

    group_specs = (
        ("seen_speaker_seen_text", "seen", "seen", seen_prompts, seen_texts),
        ("seen_speaker_unseen_text", "seen", "unseen", seen_prompts, unseen_texts),
        ("unseen_speaker_seen_text", "unseen", "seen", unseen_prompts, seen_texts),
        ("unseen_speaker_unseen_text", "unseen", "unseen", unseen_prompts, unseen_texts),
    )
    groups = [
        {
            "name": name,
            "speaker_status": speaker_status,
            "text_status": text_status,
            "speakers": speakers,
            "sentences": sentences,
            "clips": len(speakers) * len(sentences),
        }
        for name, speaker_status, text_status, speakers, sentences in group_specs
    ]
    total_clips = sum(group["clips"] for group in groups)
    floor_count = args.asr_floor_clips or total_clips
    held_out_validation_rows = [
        row for row in validation_rows if row["speaker_id"] in unseen_speakers
    ]
    payload = {
        "schema_version": 2,
        "seed": args.seed,
        "artifacts": [str(path.resolve()) for path in args.artifacts],
        "groups": groups,
        "clips_per_group": groups[0]["clips"],
        "clips_per_checkpoint": total_clips,
        "asr_floor": choose_asr_floor(held_out_validation_rows, floor_count, rng),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote {args.output}")
    print(f"4 groups x {groups[0]['clips']} clips = {total_clips} clips/checkpoint")
    print(f"seen speakers={len(train_speakers)} unseen speakers={len(unseen_speakers)}")
    print(f"ASR floor={floor_count} genuine held-out clips")


if __name__ == "__main__":
    main()
