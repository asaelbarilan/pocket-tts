"""Transcode completed audio-v2 episodes while the detached fetch is still running.

The fetcher writes metadata.json last; only then is a recording eligible. WAVs are
published by an atomic rename, so an interrupted run can safely reuse finished WAVs.
Downloaded MP3/M4A files remain until the manifest builder verifies the entire dataset.
"""

from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path

from hebrew_training.build_official_manifest import transcode_audio


def ready_audio(episode: dict, corpus: Path) -> Path | None:
    folder = corpus / episode["id"]
    source = folder / f"audio{episode['ext']}"
    if not (folder / "metadata.json").is_file():
        return None
    if not (folder / "transcript.aligned.json").is_file():
        return None
    if not source.is_file() or source.stat().st_size != episode["audio_bytes"]:
        return None
    return source


def transcode_ready(
    episodes: list[dict], corpus: Path, wavs: Path, workers: int = 16, interval: float = 2
) -> int:
    if workers < 1:
        raise ValueError("workers must be positive")
    pending = {e["id"]: e for e in episodes}
    for episode_id in list(pending):
        wav = wavs / f"{episode_id}.wav"
        if wav.is_file() and wav.stat().st_size > 44:
            del pending[episode_id]
    reused = len(episodes) - len(pending)
    print(f"transcode: {reused:,} WAVs reused, {len(pending):,} awaiting download", flush=True)
    made = 0
    in_flight = {}
    with ThreadPoolExecutor(max_workers=workers) as pool:
        while pending or in_flight:
            if len(in_flight) < workers * 2:
                for folder in corpus.iterdir():
                    episode = pending.get(folder.name)
                    if episode is None:
                        continue
                    source = ready_audio(episode, corpus)
                    if source is None:
                        continue
                    del pending[folder.name]
                    in_flight[pool.submit(transcode_audio, source, folder, wavs)] = folder.name
                    if len(in_flight) >= workers * 2:
                        break
            if not in_flight:
                time.sleep(interval)
                continue
            finished, _ = wait(in_flight, timeout=interval, return_when=FIRST_COMPLETED)
            for future in finished:
                episode_id = in_flight.pop(future)
                target, _ = (
                    future.result()
                )  # Stop on a failed transcode; keep its source for a retry.
                if not target.is_file() or target.stat().st_size <= 44:
                    raise RuntimeError(f"missing finished WAV for {episode_id}")
                made += 1
                if made % 100 == 0 or made + reused == len(episodes):
                    print(
                        f"transcode: {made + reused:,}/{len(episodes):,} WAVs ready "
                        f"({len(pending):,} awaiting download, {len(in_flight)} converting)",
                        flush=True,
                    )
    return made


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--wavs", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=16)
    args = parser.parse_args()
    episodes = json.loads(args.selection.read_text(encoding="utf-8"))["episodes"]
    args.wavs.mkdir(parents=True, exist_ok=True)
    transcode_ready(episodes, args.corpus, args.wavs, workers=args.workers)


if __name__ == "__main__":
    main()
