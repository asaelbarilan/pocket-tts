"""The audio-v2 per-row feature table and the rule for which words belong to a row.

Kept free of torch and `pocket_tts` on purpose: the rating tool (`audio_v2_tagger`) runs in
its own small environment and imports this, while `audio_v2_rows` (which writes the table
with the builder's own merge) lives in the main one. `tests/test_fetch_audio_v2.py` pins
`row_segments` to `build_official_manifest.clean_segments`, so the two cannot drift.
"""

from __future__ import annotations

import re
import statistics
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from hebrew_training import fetch_audio_v2 as av2

FEATURES = (
    "duration",
    "n_segments",
    "n_words",
    "words_per_sec",
    "lp_mean",
    "lp_min",
    "lp_max",
    "lp_median",
    "cr_mean",
    "cr_min",
    "cr_max",
    "cr_median",
    "wp_mean",
    "wp_median",
    "wp_min",
    "wp_max",
    "wp_low_frac",
    "wd_mean",
    "wd_median",
    "max_gap",
    "lead_gap_ms",
    "tail_gap_ms",
)

# build_official_manifest's --min-quality default: median word probability a segment needs.
MIN_SEGMENT_QUALITY = 0.6
_HEBREW = re.compile(r"[֐-׿]")


def row_segments(segments: list[dict], start: float, end: float) -> list[dict]:
    """Raw Whisper segments the builder would keep (`clean_segments`) inside [start, end]."""
    kept = []
    for segment in segments:
        words = segment.get("words") or []
        if not words:
            continue
        probabilities = [w["probability"] for w in words if w.get("probability") is not None]
        if probabilities and statistics.median(probabilities) < MIN_SEGMENT_QUALITY:
            continue
        timed = [w for w in words if w.get("start") is not None and w.get("end") is not None]
        if not timed or not _HEBREW.search(segment.get("text") or ""):
            continue
        if segment["start"] >= start - 1e-3 and segment["end"] <= end + 1e-3:
            kept.append({**segment, "words": timed})
    kept.sort(key=lambda s: s["start"])
    return kept


@dataclass
class Table:
    x: np.ndarray  # (rows, len(FEATURES)) float32
    episode: np.ndarray  # index into `episode_ids`
    source: np.ndarray  # index into `sources`
    start: np.ndarray
    end: np.ndarray
    episode_ids: list[str]
    sources: list[str]

    @property
    def duration(self) -> np.ndarray:
        return self.end - self.start

    def __len__(self) -> int:
        return len(self.start)


def order(work: Path) -> tuple[list[dict], list[str]]:
    """Episodes in `sorted(id)` order (what the shards index into) and the sorted sources."""
    listing = {e["id"]: e for e in av2.load_listing(work)}
    return [listing[i] for i in sorted(listing)], sorted({e["source"] for e in listing.values()})


def load_table(work: Path) -> Table:
    episodes, sources = order(work)
    shards = sorted((work / "rows").glob("shard_*.npz"))
    if not shards:
        raise SystemExit(f"no shards in {work / 'rows'} -- run audio_v2_rows first")
    parts = [np.load(path) for path in shards]
    return Table(
        x=np.concatenate([p["x"] for p in parts]),
        episode=np.concatenate([p["episode"] for p in parts]),
        source=np.concatenate([p["source"] for p in parts]),
        start=np.concatenate([p["start"] for p in parts]),
        end=np.concatenate([p["end"] for p in parts]),
        episode_ids=[e["id"] for e in episodes],
        sources=sources,
    )
