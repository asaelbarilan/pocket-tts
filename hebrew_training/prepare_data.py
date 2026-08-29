from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

from hebrew_training.data import (
    ManifestRow,
    normalize_hebrew_text,
    read_arrow_rows,
    resolve_audio_path,
    source_id_from_row,
    write_jsonl,
)


def split_is_validation(source_id: str, validation_fraction: float) -> bool:
    digest = hashlib.sha256(source_id.encode("utf-8")).digest()
    bucket = int.from_bytes(digest[:8], "big") / 2**64
    return bucket < validation_fraction


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert the existing F5 CrowdRecital Arrow data into Pocket TTS manifests."
    )
    parser.add_argument("--arrow", type=Path, required=True)
    parser.add_argument("--wavs-dir", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--validation-fraction", type=float, default=0.02)
    parser.add_argument("--min-seconds", type=float, default=1.0)
    parser.add_argument("--max-seconds", type=float, default=26.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not 0 < args.validation_fraction < 0.5:
        raise ValueError("--validation-fraction must be between 0 and 0.5")

    train: list[ManifestRow] = []
    validation: list[ManifestRow] = []
    skipped = Counter()
    source_counts = Counter()

    for row in read_arrow_rows(args.arrow):
        text = normalize_hebrew_text(str(row.get("text", "")))
        duration = float(row.get("duration", 0.0))
        audio_path = resolve_audio_path(row, args.wavs_dir)
        source_id = source_id_from_row(row)

        if not text:
            skipped["empty_text"] += 1
            continue
        if not args.min_seconds <= duration <= args.max_seconds:
            skipped["duration"] += 1
            continue
        if not audio_path.exists():
            skipped["missing_audio"] += 1
            continue
        if not any("\u0590" <= char <= "\u05ff" for char in text):
            skipped["no_hebrew"] += 1
            continue

        item = ManifestRow(str(audio_path), text, duration, source_id)
        target = validation if split_is_validation(source_id, args.validation_fraction) else train
        target.append(item)
        source_counts[source_id] += 1

    if not train or not validation:
        raise RuntimeError(
            f"Split produced train={len(train)}, validation={len(validation)}. "
            "Adjust --validation-fraction or inspect the dataset."
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.output_dir / "train.jsonl", train)
    write_jsonl(args.output_dir / "validation.jsonl", validation)

    with (args.output_dir / "tokenizer_corpus.txt").open(
        "w", encoding="utf-8", newline="\n"
    ) as handle:
        for row in train:
            handle.write(row.text + "\n")

    all_rows = train + validation
    stats = {
        "train_samples": len(train),
        "validation_samples": len(validation),
        "total_hours": round(sum(row.duration for row in all_rows) / 3600, 3),
        "source_count": len(source_counts),
        "skipped": dict(skipped),
        "arrow": str(args.arrow.resolve()),
        "wavs_dir": str(args.wavs_dir.resolve()) if args.wavs_dir else None,
    }
    (args.output_dir / "stats.json").write_text(
        json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
