#!/usr/bin/env python3
"""Sample segments from Knesset committee sessions into a pocket-tts manifest.

Source dataset:
    /mlspeech/hebrew/normalized/knesset/committee/<session_id>/
        audio.m4a
        metadata.json
        transcript.refined.json
        transcript.refined.map.json
        speakers.txt
        speakers.segments.txt
        ...

Target dataset:
    /mlspeech/hebrew/normalized/knesset/committee_sample/
        audio/<session_id>.wav
        valid_aligned.jsonl
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np

HEBREW_RE = re.compile(r"[\u0590-\u05FF]")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sample Knesset committee segments into pocket-tts manifest format."
    )
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=Path("/mlspeech/hebrew/normalized/knesset/committee"),
        help="Path to source committee dataset root",
    )
    parser.add_argument(
        "--dest-dir",
        type=Path,
        default=Path("/mlspeech/hebrew/normalized/knesset/committee_sample"),
        help="Path to destination directory",
    )
    parser.add_argument(
        "-n",
        "--num-samples",
        type=int,
        default=None,
        help="Number of samples to produce (prompted if omitted)",
    )
    parser.add_argument(
        "--max-audio-files",
        type=int,
        default=20,
        help="Maximum unique audio files to transcode (default: 20)",
    )
    parser.add_argument(
        "--min-quality",
        type=float,
        default=0.7,
        help="Minimum segment quality score (default: 0.7)",
    )
    parser.add_argument(
        "--min-duration",
        type=float,
        default=0.5,
        help="Minimum segment duration in seconds (default: 0.5)",
    )
    parser.add_argument(
        "--max-duration",
        type=float,
        default=30.0,
        help="Maximum segment duration in seconds (default: 30.0)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible sampling (default: 42)",
    )
    parser.add_argument(
        "--ffmpeg-threads",
        type=int,
        default=8,
        help="Number of threads for ffmpeg transcoding (default: 8)",
    )
    return parser.parse_args()


def transcode_to_wav(m4a_path: Path, wav_path: Path, threads: int = 8) -> None:
    """Transcode source m4a audio to mono 24kHz 16-bit PCM wav."""
    if wav_path.is_file() and wav_path.stat().st_size > 44:
        return

    wav_path.parent.mkdir(parents=True, exist_ok=True)
    partial = wav_path.with_suffix(".wav.partial")
    cmd = [
        "ffmpeg",
        "-nostdin",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(m4a_path),
        "-map",
        "0:a:0",
        "-map_metadata",
        "-1",
        "-vn",
        "-ac",
        "1",
        "-ar",
        "24000",
        "-c:a",
        "pcm_s16le",
        "-threads",
        str(threads),
        "-f",
        "wav",
        str(partial),
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        partial.unlink(missing_ok=True)
        raise RuntimeError(f"ffmpeg failed for {m4a_path}: {res.stderr.strip()}")
    partial.replace(wav_path)


def compute_segment_quality(seg: dict, meta_scores: list[dict]) -> float:
    """Compute segment quality score using metadata scores or word probabilities."""
    seg_start = float(seg.get("start", 0.0))
    seg_end = float(seg.get("end", 0.0))

    # Match overlapping intervals from metadata.json's per_segment_quality_scores
    overlaps = [
        s["probability"]
        for s in meta_scores
        if "probability" in s
        and not (s.get("end", 0.0) <= seg_start or s.get("start", 0.0) >= seg_end)
    ]
    if overlaps:
        return float(np.mean(overlaps))

    # Fallback to word-level probabilities from the segment
    words = seg.get("words", [])
    word_probs = [w["probability"] for w in words if w.get("probability") is not None]
    if word_probs:
        return float(np.median(word_probs))

    return 0.0


def extract_candidate_segments(
    session_dir: Path,
    min_quality: float,
    min_duration: float,
    max_duration: float,
) -> tuple[list[dict[str, Any]], list[dict], list[dict]] | None:
    """Extract valid candidate segments from a session directory."""
    ref_file = session_dir / "transcript.refined.json"
    map_file = session_dir / "transcript.refined.map.json"
    meta_file = session_dir / "metadata.json"
    audio_file = session_dir / "audio.m4a"

    if not (ref_file.is_file() and map_file.is_file() and meta_file.is_file() and audio_file.is_file()):
        return None

    try:
        with open(ref_file, "r", encoding="utf-8") as f:
            refined_data = json.load(f)
        with open(map_file, "r", encoding="utf-8") as f:
            refined_map = json.load(f)
        with open(meta_file, "r", encoding="utf-8") as f:
            meta_data = json.load(f)
    except Exception as e:
        print(f"Warning: failed reading files in {session_dir.name}: {e}")
        return None

    segments = refined_data.get("segments", [])
    meta_scores = meta_data.get("per_segment_quality_scores", [])
    candidates = []

    for idx, seg in enumerate(segments):
        start = float(seg.get("start", 0.0))
        end = float(seg.get("end", 0.0))
        duration = end - start
        if not (min_duration <= duration <= max_duration):
            continue

        raw_text = seg.get("text", "").strip()
        if not raw_text or not HEBREW_RE.search(raw_text):
            continue

        words = [w for w in seg.get("words", []) if w.get("word", "").strip()]
        if not words:
            continue

        quality = compute_segment_quality(seg, meta_scores)
        if quality <= min_quality:
            continue

        speaker_ids = refined_map[idx].get("speaker_ids", []) if idx < len(refined_map) else []
        speaker = str(speaker_ids[0]) if speaker_ids else session_dir.name

        candidates.append({
            "idx": idx,
            "seg": seg,
            "quality": quality,
            "speaker": speaker,
        })

    if not candidates:
        return None

    return candidates, refined_map, meta_scores


def format_sample_row(
    wav_path: Path,
    seg: dict,
    quality: float,
    speaker: str,
) -> dict:
    """Format one sample into the pocket-tts jsonl row structure."""
    seg_start = float(seg["start"])
    seg_end = float(seg["end"])
    seg_duration = round(seg_end - seg_start, 2)

    timed_words = []
    for w in seg.get("words", []):
        word_text = w.get("word", "").strip()
        if not word_text:
            continue
        w_start = w.get("start")
        w_end = w.get("end")
        if w_start is None or w_end is None:
            continue
        rel_start = round(max(0.0, float(w_start) - seg_start), 2)
        rel_end = round(max(rel_start, float(w_end) - seg_start), 2)
        rel_end = min(seg_duration, rel_end)
        timed_words.append({
            "word": word_text,
            "start": rel_start,
            "end": rel_end,
        })

    transcript = seg.get("text", "").strip()
    if not transcript:
        transcript = " ".join(w["word"] for w in timed_words)

    return {
        "path": str(wav_path.resolve()),
        "start": round(seg_start, 2),
        "duration": seg_duration,
        "transcript": transcript,
        "words": timed_words,
        "speaker": speaker,
        "quality": round(quality, 4),
    }


def main() -> None:
    args = parse_args()

    num_samples = args.num_samples
    if num_samples is None:
        if sys.stdin.isatty():
            val = input("Enter number of samples to produce [default: 100]: ").strip()
            num_samples = int(val) if val else 100
        else:
            num_samples = 100

    print(f"Target samples count: {num_samples}")
    print(f"Source directory: {args.source_dir}")
    print(f"Destination directory: {args.dest_dir}")
    print(f"Max audio files: {args.max_audio_files}")
    print(f"Min quality threshold: {args.min_quality}")

    if shutil.which("ffmpeg") is None:
        raise SystemExit("ffmpeg is required on PATH")

    if not args.source_dir.is_dir():
        raise SystemExit(f"Source directory does not exist: {args.source_dir}")

    dest_audio_dir = args.dest_dir / "audio"
    dest_audio_dir.mkdir(parents=True, exist_ok=True)
    dest_manifest = args.dest_dir / "valid_aligned.jsonl"

    rng = random.Random(args.seed)

    print("Scanning available session folders...")
    all_session_dirs = [p for p in args.source_dir.iterdir() if p.is_dir()]
    rng.shuffle(all_session_dirs)
    print(f"Found {len(all_session_dirs)} session folders.")

    # Pool of active sessions up to args.max_audio_files
    # Each entry: {
    #   "session_id": str,
    #   "dir": Path,
    #   "wav_path": Path,
    #   "candidates": list[dict],
    # }
    active_pool: list[dict[str, Any]] = []
    active_pool_ids: set[str] = set()

    session_dir_iter = iter(all_session_dirs)
    used_segments: set[tuple[str, int]] = set()
    samples: list[dict] = []

    print(f"\nStarting sampling loop (limit: {args.max_audio_files} audio files)...")

    while len(samples) < num_samples:
        # Determine which session to sample from
        if len(active_pool) < args.max_audio_files:
            # Find next valid session to transcode and add to pool
            session_entry = None
            while len(active_pool) < args.max_audio_files:
                try:
                    s_dir = next(session_dir_iter)
                except StopIteration:
                    print("Ran out of session folders to check.")
                    break

                if s_dir.name in active_pool_ids:
                    continue

                candidates_res = extract_candidate_segments(
                    s_dir,
                    args.min_quality,
                    args.min_duration,
                    args.max_duration,
                )
                if candidates_res is None:
                    continue

                candidates, _, _ = candidates_res
                wav_path = dest_audio_dir / f"{s_dir.name}.wav"
                print(f"[{len(active_pool) + 1}/{args.max_audio_files}] Transcoding audio for session {s_dir.name}...")
                transcode_to_wav(s_dir / "audio.m4a", wav_path, threads=args.ffmpeg_threads)

                entry = {
                    "session_id": s_dir.name,
                    "dir": s_dir,
                    "wav_path": wav_path,
                    "candidates": candidates,
                }
                active_pool.append(entry)
                active_pool_ids.add(s_dir.name)
                session_entry = entry
                break

            if session_entry is None and not active_pool:
                raise SystemExit("Error: Could not find any valid sessions.")
            if session_entry is None:
                # If no more new sessions, pick from active pool
                session_entry = rng.choice(active_pool)
        else:
            # Pool has reached max_audio_files (e.g. 20), pick exclusively from active pool
            session_entry = rng.choice(active_pool)

        # Draw a candidate from chosen session
        avail_candidates = [
            c for c in session_entry["candidates"]
            if (session_entry["session_id"], c["idx"]) not in used_segments
        ]

        if not avail_candidates:
            # If all candidates used from this session, pick another active session
            other_entries = [
                e for e in active_pool
                if any((e["session_id"], c["idx"]) not in used_segments for c in e["candidates"])
            ]
            if not other_entries:
                print("All candidate segments across active sessions have been used.")
                break
            session_entry = rng.choice(other_entries)
            avail_candidates = [
                c for c in session_entry["candidates"]
                if (session_entry["session_id"], c["idx"]) not in used_segments
            ]

        chosen = rng.choice(avail_candidates)
        used_segments.add((session_entry["session_id"], chosen["idx"]))

        row = format_sample_row(
            wav_path=session_entry["wav_path"],
            seg=chosen["seg"],
            quality=chosen["quality"],
            speaker=chosen["speaker"],
        )
        samples.append(row)

        if len(samples) % 10 == 0 or len(samples) == num_samples:
            print(
                f"Generated {len(samples)}/{num_samples} samples "
                f"(from {len(active_pool)} transcoded audio files)..."
            )

    print(f"\nWriting {len(samples)} samples to {dest_manifest}...")
    with open(dest_manifest, "w", encoding="utf-8") as f:
        for row in samples:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print("Done! Summary:")
    print(f"  Total samples written: {len(samples)}")
    print(f"  Total audio files:     {len(active_pool)}")
    print(f"  Output manifest:       {dest_manifest}")
    print(f"  Output audio dir:      {dest_audio_dir}")


if __name__ == "__main__":
    main()
