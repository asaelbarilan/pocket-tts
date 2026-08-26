"""Convert Hebrew corpora into the manifest format Kyutai's trainer expects.

Replaces our own slicing pipeline for training. The official DataLoader does its own
windowing: given `words`, it cuts each utterance at a random word boundary, uses the audio
before the cut as voice conditioning and the audio after it as the target, paired with only
the remaining words as text. So we hand it whole aligned utterances and let it decide the
split, rather than pre-cutting clips and pairing prompts ourselves.

Row schema (see `get_entry` in training/dataloader.py):

    {"path": str, "start": float, "duration": float, "transcript": str,
     "words": [{"word": str, "start": float, "end": float}]}

`words` is optional but strongly preferred; without it the loader falls back to a random
window for the voice prompt. Word times are RELATIVE to `start`, because align_data.py
keys on (path, start) and aligns inside that window.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import statistics
import sys
import unicodedata
from pathlib import Path

_WS = re.compile(r"\s+")
_HEBREW = re.compile(r"[֐-׿]")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build Kyutai-format training manifests from an aligned Hebrew corpus."
    )
    parser.add_argument("--corpus", type=Path, required=True,
                        help="Root holding one directory per recording.")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--audio-glob", default="audio.wav")
    parser.add_argument("--align-glob", default="transcript.aligned.json")
    parser.add_argument("--metadata-glob", default="metadata.json")
    parser.add_argument("--min-duration", type=float, default=1.0)
    parser.add_argument("--max-duration", type=float, default=30.0,
                        help="Should match data.max_duration_sec in the training config.")
    parser.add_argument("--min-quality", type=float, default=0.6,
                        help="Median word probability required to keep a segment.")
    parser.add_argument("--valid-hours", type=float, default=2.0)
    parser.add_argument("--normalize-text", action="store_true",
                        help="Apply the Hebrew TTS normalizer (number expansion etc).")
    parser.add_argument("--normalizer-dir", type=Path,
                        default=Path("../hebrew-tts-data-tools/normalizer"))
    parser.add_argument("--limit-recordings", type=int)
    return parser.parse_args()


def clean(text: str) -> str:
    return _WS.sub(" ", unicodedata.normalize("NFC", text)).strip()


def stable_bucket(key: str) -> float:
    return int.from_bytes(hashlib.sha256(key.encode("utf-8")).digest()[:8], "big") / 2**64


def load_normalizer(directory: Path):
    """Optional: the same Hebrew normalizer the earlier pipeline used."""
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory.resolve()))
    from hebrew_tts_normalizer import (  # type: ignore
        TTSNormalizeOptions,
        load_word_replacements,
        normalize_tts_text,
    )

    options = TTSNormalizeOptions(
        apply_word_replacements=True,
        expand_numbers=True,
        keep_punctuations=True,
        attach_punctuations_to_token=True,
        stt_compat_mode=False,
        remove_parentheses=False,
    )
    replacements = load_word_replacements()
    return lambda text: normalize_tts_text(text, options=options, word_replacements=replacements)


def segment_rows(recording: Path, args, normalize) -> tuple[list[dict], dict]:
    """One manifest row per aligned segment of one recording."""
    stats = {"no_words": 0, "low_quality": 0, "bad_duration": 0, "no_hebrew": 0, "kept": 0}
    audio = next(iter(recording.glob(args.audio_glob)), None)
    align = next(iter(recording.glob(args.align_glob)), None)
    if audio is None or align is None:
        return [], stats

    speaker = None
    meta_file = next(iter(recording.glob(args.metadata_glob)), None)
    if meta_file is not None:
        try:
            speaker = json.loads(meta_file.read_text(encoding="utf-8")).get("user_id")
        except (json.JSONDecodeError, OSError):
            speaker = None

    try:
        segments = json.loads(align.read_text(encoding="utf-8"))["segments"]
    except (json.JSONDecodeError, KeyError, OSError):
        return [], stats

    rows = []
    for segment in segments:
        words = segment.get("words") or []
        if not words:
            stats["no_words"] += 1
            continue

        probabilities = [w["probability"] for w in words if w.get("probability") is not None]
        if probabilities and statistics.median(probabilities) < args.min_quality:
            stats["low_quality"] += 1
            continue

        start = float(segment["start"])
        duration = float(segment["end"]) - start
        if not args.min_duration <= duration <= args.max_duration:
            stats["bad_duration"] += 1
            continue

        text = clean(segment.get("text", ""))
        if normalize is not None:
            text = clean(normalize(text))
        if not text or not _HEBREW.search(text):
            stats["no_hebrew"] += 1
            continue

        # Relative to `start`: the loader compares these against entry.duration, and
        # align_data.py aligns inside the (path, start) window.
        timed = [
            {
                "word": clean(word["word"]),
                "start": round(float(word["start"]) - start, 4),
                "end": round(float(word["end"]) - start, 4),
            }
            for word in words
            if word.get("start") is not None and word.get("end") is not None
        ]
        if not timed:
            stats["no_words"] += 1
            continue

        rows.append(
            {
                "path": str(audio.resolve()),
                "start": round(start, 4),
                "duration": round(duration, 4),
                "transcript": text,
                "words": timed,
                "speaker": speaker or "unknown",
            }
        )
        stats["kept"] += 1
    return rows, stats


def main() -> None:
    args = parse_args()
    normalize = load_normalizer(args.normalizer_dir) if args.normalize_text else None

    recordings = sorted(p for p in args.corpus.iterdir() if p.is_dir())
    if args.limit_recordings:
        recordings = recordings[: args.limit_recordings]

    all_rows: list[dict] = []
    totals = {"no_words": 0, "low_quality": 0, "bad_duration": 0, "no_hebrew": 0, "kept": 0}
    for recording in recordings:
        rows, stats = segment_rows(recording, args, normalize)
        all_rows.extend(rows)
        for key, value in stats.items():
            totals[key] += value
    if not all_rows:
        raise SystemExit("no usable utterances found")

    # Speaker-disjoint split, deterministic: whole speakers go to validation until the
    # hour budget is met, so no voice appears on both sides.
    seconds_by_speaker: dict[str, float] = {}
    for row in all_rows:
        seconds_by_speaker[row["speaker"]] = (
            seconds_by_speaker.get(row["speaker"], 0.0) + row["duration"]
        )
    budget = args.valid_hours * 3600
    valid_speakers: set[str] = set()
    accumulated = 0.0
    for speaker in sorted(seconds_by_speaker, key=stable_bucket):
        if accumulated >= budget:
            break
        if seconds_by_speaker[speaker] > budget * 1.5:
            continue
        valid_speakers.add(speaker)
        accumulated += seconds_by_speaker[speaker]

    train = [r for r in all_rows if r["speaker"] not in valid_speakers]
    valid = [r for r in all_rows if r["speaker"] in valid_speakers]
    assert not ({r["speaker"] for r in train} & valid_speakers), "speaker leaked across split"

    args.out_dir.mkdir(parents=True, exist_ok=True)
    for name, rows in (("train_aligned.jsonl", train), ("valid_aligned.jsonl", valid)):
        with (args.out_dir / name).open("w", encoding="utf-8", newline="\n") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    def hours(rows: list[dict]) -> float:
        return sum(r["duration"] for r in rows) / 3600

    print(f"recordings scanned : {len(recordings)}")
    print(f"utterances kept    : {totals['kept']}")
    print(
        f"  dropped: no_words {totals['no_words']}, low_quality {totals['low_quality']}, "
        f"duration {totals['bad_duration']}, no_hebrew {totals['no_hebrew']}"
    )
    print(
        f"train : {len(train):6d} utterances, {hours(train):6.2f} h, "
        f"{len({r['speaker'] for r in train})} speakers"
    )
    print(f"valid : {len(valid):6d} utterances, {hours(valid):6.2f} h, {len(valid_speakers)} speakers")
    print(f"wrote {args.out_dir}/train_aligned.jsonl and valid_aligned.jsonl")


if __name__ == "__main__":
    main()
