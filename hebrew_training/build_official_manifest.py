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

WHY SEGMENTS ARE MERGED
-----------------------
CrowdRecital ships utterance-length segments. The ivrit-ai Knesset corpora ship Whisper
DECODER segments: median 1.84 s, p90 4.40 s, measured over 702,646 segments in 120 random
plenum recordings. That length is not usable as a manifest row, because of `MIN_CUT_SEC`
in training/dataloader.py:

    a cut must leave >= 1.0 s of audio on BOTH sides

so a row shorter than 2.0 s has no eligible word-boundary cut at all -- 53.7% of raw
segments. Those rows take the loader's fallback branch, where the voice prompt is a window
read from `entry.start`, i.e. it OVERLAPS the target audio the model is asked to predict.
That is the same prompt leakage prepare_data_v2 was written to remove.

So consecutive segments are merged into utterances of ~12 s before a row is emitted, and
merging stops at a gap longer than --merge-gap (parliamentary audio is 63.7% speech; the
rest is gavel, procedure and dead air, and merging across it would train the model on
silence). Measured on the same 120 recordings, --merge-gap 1.5 --min-duration 4 yields a
12.64 s median row and retains ~5,494 h of the plenum corpus.
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

# Resolved from this file, not the working directory: the earlier relative default broke
# whenever the command was run from anywhere but the repo root. Assumes the two repos are
# checked out side by side, which is what the recipe tells you to do.
DEFAULT_NORMALIZER = (
    Path(__file__).resolve().parents[2] / "hebrew-tts-data-tools" / "normalizer"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build Kyutai-format training manifests from an aligned Hebrew corpus."
    )
    parser.add_argument("--corpus", type=Path, required=True,
                        help="Root holding one directory per recording.")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--audio-glob", default="audio.wav,audio.m4a",
                        help="Comma-separated candidates; the first that matches wins. "
                             "CrowdRecital ships audio.wav, the Knesset corpora audio.m4a.")
    parser.add_argument("--align-glob", default="transcript.aligned.json")
    parser.add_argument("--metadata-glob", default="metadata.json")
    parser.add_argument("--speaker-field", default="user_id",
                        help="metadata.json key identifying the speaker. Falls back to the "
                             "recording directory name, which for the Knesset corpora means "
                             "the split is recording-disjoint but not speaker-disjoint.")
    parser.add_argument("--min-duration", type=float, default=4.0,
                        help="Post-merge floor. Must stay above 2 x MIN_CUT_SEC (2.0 s) or "
                             "the loader cannot cut the row and leaks the prompt.")
    parser.add_argument("--max-duration", type=float, default=30.0,
                        help="Should match data.max_duration_sec in the training config.")
    parser.add_argument("--merge-target", type=float, default=12.0,
                        help="Stop merging once a group reaches this length.")
    parser.add_argument("--merge-gap", type=float, default=1.5,
                        help="Break a merge at a silence longer than this. 0 disables merging, "
                             "which is right only for already-utterance-length corpora.")
    parser.add_argument("--min-quality", type=float, default=0.6,
                        help="Median word probability required to keep a segment.")
    parser.add_argument("--valid-hours", type=float, default=2.0)
    parser.add_argument("--normalize-text", action="store_true",
                        help="Apply the Hebrew TTS normalizer (number expansion etc).")
    parser.add_argument("--normalizer-dir", type=Path, default=DEFAULT_NORMALIZER,
                        help="The normalizer package from the hebrew-tts-data-tools repo. "
                             "Defaults to a checkout sitting beside this one.")
    parser.add_argument("--limit-recordings", type=int)
    return parser.parse_args()


def clean(text: str) -> str:
    return _WS.sub(" ", unicodedata.normalize("NFC", text)).strip()


def stable_bucket(key: str) -> float:
    return int.from_bytes(hashlib.sha256(key.encode("utf-8")).digest()[:8], "big") / 2**64


def load_normalizer(directory: Path):
    """The same Hebrew normalizer the earlier pipeline used -- number expansion,
    word replacements, punctuation handling. Only the slicing was replaced; this was not."""
    if not directory.exists():
        raise SystemExit(
            f"--normalize-text needs the normalizer, and {directory} does not exist.\n"
            "Clone it beside this repo:\n"
            "    git clone https://github.com/asaelbarilan/hebrew-tts-data-tools\n"
            "or pass --normalizer-dir explicitly."
        )
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


def find_audio(recording: Path, globs: str) -> Path | None:
    """First matching audio file. Corpora differ: audio.wav here, audio.m4a there."""
    for pattern in (g.strip() for g in globs.split(",") if g.strip()):
        match = next(iter(recording.glob(pattern)), None)
        if match is not None:
            return match
    return None


def clean_segments(segments: list[dict], args, stats: dict) -> list[dict]:
    """Segments that pass the quality and language filters, times still absolute."""
    kept = []
    for segment in segments:
        words = segment.get("words") or []
        if not words:
            stats["no_words"] += 1
            continue

        probabilities = [w["probability"] for w in words if w.get("probability") is not None]
        if probabilities and statistics.median(probabilities) < args.min_quality:
            stats["low_quality"] += 1
            continue

        timed = [
            w for w in words
            if w.get("start") is not None and w.get("end") is not None
        ]
        if not timed:
            stats["no_words"] += 1
            continue

        text = clean(segment.get("text", ""))
        if not text or not _HEBREW.search(text):
            stats["no_hebrew"] += 1
            continue

        kept.append(
            {
                "start": float(segment["start"]),
                "end": float(segment["end"]),
                "text": text,
                "words": timed,
            }
        )
    kept.sort(key=lambda s: s["start"])
    return kept


def merge_segments(segments: list[dict], args) -> list[list[dict]]:
    """Group consecutive segments into utterance-length runs.

    A group closes when it reaches --merge-target, when the next segment would push it
    past --max-duration, or when the silence before the next segment exceeds --merge-gap.
    See the module docstring for why this is required and not a tuning knob.
    """
    if args.merge_gap <= 0:
        return [[segment] for segment in segments]

    groups: list[list[dict]] = []
    current: list[dict] = []
    for segment in segments:
        if not current:
            current = [segment]
            continue
        gap = segment["start"] - current[-1]["end"]
        span = segment["end"] - current[0]["start"]
        if gap <= args.merge_gap and span <= args.max_duration:
            current.append(segment)
            if current[-1]["end"] - current[0]["start"] >= args.merge_target:
                groups.append(current)
                current = []
        else:
            groups.append(current)
            current = [segment]
    if current:
        groups.append(current)
    return groups


def segment_rows(recording: Path, args, normalize) -> tuple[list[dict], dict]:
    """One manifest row per merged run of aligned segments."""
    stats = {"no_words": 0, "low_quality": 0, "bad_duration": 0, "no_hebrew": 0, "kept": 0}
    audio = find_audio(recording, args.audio_glob)
    align = next(iter(recording.glob(args.align_glob)), None)
    if audio is None or align is None:
        return [], stats

    # No speaker field in the Knesset corpora, so fall back to the recording id. Without
    # this every row would share one speaker key and the validation split would collapse.
    speaker = None
    meta_file = next(iter(recording.glob(args.metadata_glob)), None)
    if meta_file is not None:
        try:
            speaker = json.loads(meta_file.read_text(encoding="utf-8")).get(args.speaker_field)
        except (json.JSONDecodeError, OSError):
            speaker = None
    speaker = str(speaker) if speaker else recording.name

    try:
        segments = json.loads(align.read_text(encoding="utf-8"))["segments"]
    except (json.JSONDecodeError, KeyError, OSError):
        return [], stats

    rows = []
    for group in merge_segments(clean_segments(segments, args, stats), args):
        start = group[0]["start"]
        duration = group[-1]["end"] - start
        if not args.min_duration <= duration <= args.max_duration:
            stats["bad_duration"] += 1
            continue

        text = clean(" ".join(segment["text"] for segment in group))
        if normalize is not None:
            text = clean(normalize(text))
        if not text:
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
            for segment in group
            for word in segment["words"]
        ]

        rows.append(
            {
                "path": str(audio.resolve()),
                "start": round(start, 4),
                "duration": round(duration, 4),
                "transcript": text,
                "words": timed,
                "speaker": speaker,
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
    ordered = sorted(seconds_by_speaker, key=stable_bucket)
    valid_speakers: set[str] = set()
    accumulated = 0.0
    for speaker in ordered:
        if accumulated >= budget:
            break
        if seconds_by_speaker[speaker] > budget * 1.5:
            continue
        valid_speakers.add(speaker)
        accumulated += seconds_by_speaker[speaker]

    # A whole Knesset sitting runs several hours, so every candidate can exceed the
    # oversize guard above and leave validation empty -- which the trainer only reports
    # much later, as "no entries for rank 0" out of load_entries. Take the smallest
    # speaker rather than shipping an empty valid split.
    if not valid_speakers:
        smallest = min(ordered, key=lambda s: seconds_by_speaker[s])
        valid_speakers.add(smallest)
        accumulated = seconds_by_speaker[smallest]
        print(
            f"note: no speaker fit under {args.valid_hours} h; validation is the single "
            f"smallest ({smallest}, {accumulated / 3600:.2f} h)"
        )

    train = [r for r in all_rows if r["speaker"] not in valid_speakers]
    valid = [r for r in all_rows if r["speaker"] in valid_speakers]
    assert not ({r["speaker"] for r in train} & valid_speakers), "speaker leaked across split"
    assert train and valid, "one side of the split is empty"

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

    # The check that matters: rows shorter than 2 x MIN_CUT_SEC cannot be cut by the
    # loader and fall back to a prompt window that overlaps the target. If this line
    # shows a low median or a non-zero uncuttable count, the merge settings are wrong.
    lengths = sorted(r["duration"] for r in all_rows)
    uncuttable = sum(1 for d in lengths if d < 2.0)
    print(
        f"row duration : median {statistics.median(lengths):.2f} s, "
        f"p10 {lengths[len(lengths) // 10]:.2f}, p90 {lengths[9 * len(lengths) // 10]:.2f}, "
        f"max {lengths[-1]:.2f}"
    )
    print(
        f"uncuttable (<2.0 s, prompt would overlap target): {uncuttable} "
        f"({100 * uncuttable / len(lengths):.2f}%)"
    )
    print(f"wrote {args.out_dir}/train_aligned.jsonl and valid_aligned.jsonl")


if __name__ == "__main__":
    main()
