"""Pass 2 of the audio-v2 survey: one feature row per candidate training row.

`fetch_audio_v2 survey` keeps only histograms, which cannot say how a *training row* (the
~12 s merge the builder emits) scores. This pass re-reads the transcripts and cuts rows with
the builder's own `clean_segments` + `merge_segments` (same defaults, same per-recording seed,
no Whisper-confidence filters), then records per-row features:

    Whisper:  lp_*  (avg_logprob over the row's segments, duration-weighted mean/min/max/median)
              cr_*  (compression_ratio, same statistics)
    words:    wp_*  (word probability: mean/median/min/max, share below 0.5)
              wd_*  (word duration: mean/median)
    shape:    duration, n_segments, n_words, words_per_sec, max_gap (inside the row),
              lead_gap_ms / tail_gap_ms (silence to the neighbouring raw segments)

plus `episode`, `source`, `start`, `end` to find the audio again. Only transcripts are read
(~8 GB through the Hub, nothing kept); the table is a few hundred MB.

    python -m hebrew_training.audio_v2_rows --work W --workers 8

Resumable in shards of --shard-size episodes (`W/rows/shard_NNNN.npz`). The episode order is
`sorted(ids)` of listing.json, so shard contents do not depend on download timing.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from hebrew_training import fetch_audio_v2 as av2
from hebrew_training.build_official_manifest import (
    _recording_rng,
    build_parser,
    clean_segments,
    merge_segments,
    neighbour_gaps,
)

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


def builder_args() -> argparse.Namespace:
    """The builder's defaults, with every Whisper-confidence filter off."""
    return build_parser().parse_args(["--corpus", "-", "--out-dir", "-"])


def _nan_stats(values: np.ndarray, weights: np.ndarray | None = None):
    ok = ~np.isnan(values)
    if not ok.any():
        return (np.nan,) * 4
    v = values[ok]
    mean = float(np.average(v, weights=weights[ok])) if weights is not None else float(v.mean())
    return mean, float(v.min()), float(v.max()), float(np.median(v))


def row_features(group: list[dict], neighbours) -> list[float]:
    start, end = group[0]["start"], group[-1]["end"]
    duration = max(end - start, 1e-3)
    seg_seconds = np.array([max(s["end"] - s["start"], 1e-3) for s in group])
    lp = np.array([s.get("avg_logprob", np.nan) for s in group], dtype=float)
    cr = np.array([s.get("compression_ratio", np.nan) for s in group], dtype=float)
    words = [w for s in group for w in s["words"]]
    probabilities = np.array(
        [w["probability"] if w.get("probability") is not None else np.nan for w in words],
        dtype=float,
    )
    word_seconds = np.array([max(w["end"] - w["start"], 0.0) for w in words])
    gaps = [max(0.0, b["start"] - a["end"]) for a, b in zip(group, group[1:])]
    lead, tail = neighbours(group)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        lp_mean, lp_min, lp_max, lp_median = _nan_stats(lp, seg_seconds)
        cr_mean, cr_min, cr_max, cr_median = _nan_stats(cr, seg_seconds)
        wp_mean, wp_min, wp_max, wp_median = _nan_stats(probabilities)
        low = float(np.nanmean(probabilities < 0.5)) if np.isfinite(probabilities).any() else np.nan
    return [
        duration,
        len(group),
        len(words),
        len(words) / duration,
        lp_mean,
        lp_min,
        lp_max,
        lp_median,
        cr_mean,
        cr_min,
        cr_max,
        cr_median,
        wp_mean,
        wp_median,
        wp_min,
        wp_max,
        low,
        float(word_seconds.mean()) if len(words) else np.nan,
        float(np.median(word_seconds)) if len(words) else np.nan,
        max(gaps) if gaps else 0.0,
        np.nan if lead is None else float(lead),
        np.nan if tail is None else float(tail),
    ]


def candidate_rows(segments: list[dict], episode_id: str, args) -> list[tuple[float, float, list]]:
    """(start, end, features) of every row the builder would emit for this recording."""
    cleaned = clean_segments(segments, args, defaultdict(int))
    groups = merge_segments(cleaned, args, _recording_rng(args.seed, Path(episode_id)))
    neighbours = neighbour_gaps(segments)
    rows = []
    for group in groups:
        start, end = group[0]["start"], group[-1]["end"]
        if not args.min_duration <= end - start <= args.max_duration:
            continue
        rows.append((start, end, row_features(group, neighbours)))
    return rows


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


def _order(work: Path) -> tuple[list[dict], list[str]]:
    listing = {e["id"]: e for e in av2.load_listing(work)}
    ids = sorted(listing)
    return [listing[i] for i in ids], sorted({e["source"] for e in listing.values()})


def load_table(work: Path) -> Table:
    episodes, sources = _order(work)
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


def build(args: argparse.Namespace) -> None:
    episodes, sources = _order(args.work)
    source_index = {s: i for i, s in enumerate(sources)}
    out_dir = args.work / "rows"
    out_dir.mkdir(parents=True, exist_ok=True)
    builder = builder_args()
    total = len(episodes) if not args.limit_episodes else min(args.limit_episodes, len(episodes))
    shard_count = (total + args.shard_size - 1) // args.shard_size
    started = time.monotonic()
    failed = 0

    def one(index: int):
        episode = episodes[index]
        segments = av2.fetch_transcript(episode).get("segments") or []
        rows = candidate_rows(segments, episode["id"], builder)
        return index, rows

    for shard in range(shard_count):
        path = out_dir / f"shard_{shard:04d}.npz"
        if path.exists():
            continue
        low, high = shard * args.shard_size, min(total, (shard + 1) * args.shard_size)
        collected: dict[int, list] = {}
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = {pool.submit(one, i): i for i in range(low, high)}
            for future in as_completed(futures):
                try:
                    index, rows = future.result()
                    collected[index] = rows
                except Exception as exc:  # noqa: BLE001 -- retried by re-running; shard not saved
                    failed += 1
                    print(f"  failed {episodes[futures[future]]['id']}: {exc}", file=sys.stderr)
        if len(collected) != high - low:
            print(f"shard {shard}: {high - low - len(collected)} failed, not saved; rerun")
            continue
        xs, eps, srcs, starts, ends = [], [], [], [], []
        for index in range(low, high):
            for start, end, features in collected[index]:
                xs.append(features)
                eps.append(index)
                srcs.append(source_index[episodes[index]["source"]])
                starts.append(start)
                ends.append(end)
        np.savez_compressed(
            path,
            x=np.asarray(xs, dtype=np.float32).reshape(-1, len(FEATURES)),
            episode=np.asarray(eps, dtype=np.int32),
            source=np.asarray(srcs, dtype=np.int16),
            start=np.asarray(starts, dtype=np.float64),
            end=np.asarray(ends, dtype=np.float64),
        )
        rate = (high) / (time.monotonic() - started)
        print(f"shard {shard + 1}/{shard_count}: {len(xs)} rows ({rate:.1f} episodes/s overall)")
    (out_dir / "features.json").write_text(json.dumps(list(FEATURES)), encoding="utf-8")
    print(f"done; {failed} transient failures")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--shard-size", type=int, default=500)
    parser.add_argument("--limit-episodes", type=int, help="First N episodes (testing).")
    build(parser.parse_args())


if __name__ == "__main__":
    main()
