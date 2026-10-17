"""Build a self-contained HiFiTTS-2 (English) manifest for Kyutai's trainer.

Purpose: keep some of the base model's English while it learns Hebrew. HiFiTTS-2
(nvidia/hifitts-2, 44 kHz subset: 31.7 kh of LibriVox audiobooks, ~5k speakers) is the
diverse half of the English mix; build_libritts_r_manifest.py is the clean half.

    python -m hebrew_training.build_hifitts2_manifest --out-dir data/english/hifitts2 --hours 750

Everything training needs ends up in --out-dir; nothing is fetched while training:

    train.jsonl, valid.jsonl   rows in the format described in build_official_manifest.py
    audio/                     one 24 kHz mono FLAC (or WAV) per utterance, `start` is 0
    meta/                      the downloaded metadata (~6 GiB, kept so a bigger rerun is
                               offline up to the audio fetch; safe to delete afterwards)
    build_state.json           what the audio was built with (see prepare_output)

Where each part comes from
    chapter list + URLs     nvidia/hifitts-2 44khz/chapters_44khz.json (1.7 GB, once)
    transcripts + words     kyutai/hifitts2-aligned, the word alignments Kyutai trained pocket-tts
                            with (4.2 GiB, once). They are CTC-aligned on the 24 kHz utterance cuts
                            this script makes, so the times are valid as they are.
    validation              kyutai/hifitts2-aligned eval_aligned.jsonl.gz (1000 held-out
                            utterances). Its speakers are NOT guaranteed disjoint from train;
                            the overlap is printed.
    audio                   LibriVox chapter MP3s on archive.org, cut at NVIDIA's offsets

Choosing the hours
    --hours is the target for TRAIN, in utterance hours after the filters below. A utterance
    set is whole chapters taken in a fixed pseudo-random order (stable_bucket of the chapter),
    so every smaller selection is a prefix of every larger one: raising --hours downloads only
    the extra chapters and reuses every utterance file already in audio/. The shards of the
    alignment repo are shuffled, so choosing chapters (not utterances) is what keeps the download
    small: an utterance-level sample would touch nearly every chapter.

    Estimated cost, from 311 sampled chapters (chapter audio is ~1.16x the utterance hours, ~123
    kbps MP3) and a measured FLAC size: about 64 MB of MP3 fetched per utterance hour (deleted
    chapter by chapter, so it never sits on disk together) and ~75 MB of FLAC kept. 750 h is
    ~48 GB of MP3 traffic and ~56 GB kept, plus ~7 GB of traffic for 400 validation chapters.
    The script prints its own estimate before it downloads any audio.

Cleanliness filters (applied identically when counting hours and when building rows)
    --max-wer 0.1       Kyutai/NVIDIA's ASR word error rate of the transcript. Half the corpus is 0,
                        the 90th percentile is 0.11.
    --min-duration 2    a row under 2 x MIN_CUT_SEC has no word-boundary cut for the loader.
    --max-duration 30   data.max_duration_sec of the training config.
    Words the aligner could not time are punctuation-only tokens (a dash, a quote): 9% of rows.
    They are kept as zero-width words at the boundary between their neighbours; see
    fill_untimed_words.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from collections import Counter
from functools import partial
from pathlib import Path

from hebrew_training.official_manifest_common import (
    AUDIO_FORMATS,
    SAMPLE_RATE,
    audio_suffix,
    describe_split,
    fill_untimed_words,
    format_seconds,
    make_row,
    normalize_words,
    parallel,
    prepare_output,
    print_duration_report,
    stable_bucket,
    write_audio,
    write_manifests,
)

NVIDIA_REPO = "nvidia/hifitts-2"
CHAPTERS_REMOTE = "44khz/chapters_44khz.json"
KYUTAI_REPO = "kyutai/hifitts2-aligned"
EVAL_REMOTE = "eval_aligned.jsonl.gz"
LAYOUT_VERSION = 1
DEFAULT_HOURS = 750.0
# Rows are collected for this much more than --hours, so chapters that fail to download (dead
# LibriVox links) can be replaced from the same ordered list without scanning the alignments again.
POOL_MARGIN = 1.25
MP3_MB_PER_HOUR = 55.0
FLAC_MB_PER_HOUR = 75.0
PROGRESS_SECONDS = 15.0

META_COMPLETE = ".complete"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a self-contained HiFiTTS-2 manifest (audio + alignments) for training."
    )
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument(
        "--hours",
        type=float,
        default=DEFAULT_HOURS,
        help="Target hours of TRAIN utterances (default: %(default)s). Raising it later reuses "
        "everything already downloaded.",
    )
    parser.add_argument(
        "--valid-utterances",
        type=int,
        default=400,
        help="How many of Kyutai's 1000 held-out utterances to use as valid (default: "
        "%(default)s, ~1.2 h). Each sits in its own chapter, so each costs one chapter download.",
    )
    parser.add_argument("--max-wer", type=float, default=0.1)
    parser.add_argument("--min-duration", type=float, default=2.0)
    parser.add_argument("--max-duration", type=float, default=30.0)
    parser.add_argument(
        "--max-chapter-hours",
        type=float,
        default=1.5,
        help="Skip chapters longer than this: each is decoded whole (float32, 44.1 kHz stereo, "
        "~1.3 GB per hour) in one worker. Skipped chapters are replaced from the pool.",
    )
    parser.add_argument("--audio-format", choices=AUDIO_FORMATS, default="flac")
    parser.add_argument(
        "--workers",
        type=int,
        default=min(8, os.cpu_count() or 1),
        help="Processes for scanning the alignments and for fetching/cutting chapters.",
    )
    parser.add_argument(
        "--meta-dir", type=Path, help="Downloaded metadata (default: OUT_DIR/meta)."
    )
    parser.add_argument(
        "--yes",
        action="store_true",
        help="Do not ask before deleting an incompatible earlier build in --out-dir.",
    )
    args = parser.parse_args(argv)
    if args.hours <= 0:
        parser.error("--hours must be positive")
    if args.workers < 1:
        parser.error("--workers must be at least 1")
    if args.min_duration < 2.0:
        parser.error("--min-duration below 2.0 s leaves rows the loader cannot cut")
    args.meta_dir = args.meta_dir or args.out_dir / "meta"
    return args


def chapter_key(audio_filepath: str) -> str:
    """The chapter an utterance id belongs to: `<chapter stem>_<n>.flac` -> `<chapter stem>`."""
    return audio_filepath.rsplit("_", 1)[0]


def chapter_key_of_path(chapter_filepath: str) -> str:
    return chapter_filepath.rsplit(".", 1)[0]


def usable_row(row: dict, settings: dict) -> tuple[dict | None, str | None]:
    """The row to build from, or the reason it is dropped.

    Used both to count hours and to build rows, so the hours that were promised are the hours
    that arrive."""
    duration = float(row["duration"])
    if duration < settings["min_duration"]:
        return None, "too_short"
    if duration > settings["max_duration"]:
        return None, "too_long"
    wer = row.get("src_wer")
    if wer is not None and wer > settings["max_wer"]:
        return None, "wer"
    if not row.get("words"):
        return None, "no_words"
    try:
        words = normalize_words(fill_untimed_words(row["words"], duration), duration)
    except ValueError:
        return None, "bad_words"
    return {
        "audio_filepath": row["audio_filepath"],
        "duration": duration,
        "speaker": str(row["speaker"]),
        "words": words,
    }, None


def scan_shard(path: str, settings: dict) -> tuple[dict[str, float], dict[str, int]]:
    """Seconds of usable utterances per chapter in one alignment shard."""
    seconds: dict[str, float] = {}
    dropped: Counter = Counter()
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            parsed, reason = usable_row(json.loads(line), settings)
            if parsed is None:
                dropped[reason] += 1
                continue
            key = chapter_key(parsed["audio_filepath"])
            seconds[key] = seconds.get(key, 0.0) + parsed["duration"]
    return seconds, dict(dropped)


def collect_shard(path: str, settings: dict, wanted: frozenset) -> dict[str, dict]:
    """The usable rows of the `wanted` chapters in one alignment shard."""
    rows: dict[str, dict] = {}
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if chapter_key(row["audio_filepath"]) not in wanted:
                continue
            parsed, _ = usable_row(row, settings)
            if parsed is not None:
                rows[parsed["audio_filepath"]] = parsed
    return rows


def ensure_meta(meta_dir: Path) -> tuple[Path, list[Path], Path]:
    """Download the metadata once. A `.complete` marker makes every later run offline."""
    from huggingface_hub import HfApi, hf_hub_download

    nvidia, kyutai = meta_dir / "nvidia", meta_dir / "kyutai"
    chapters = nvidia / CHAPTERS_REMOTE
    eval_path = kyutai / EVAL_REMOTE
    marker = meta_dir / META_COMPLETE
    if not marker.is_file():
        print(
            "downloading metadata once (~6 GiB: chapter list 1.7 GB, alignments 4.2 GiB) "
            f"into {meta_dir}",
            flush=True,
        )
        names = sorted(
            name
            for name in HfApi().list_repo_files(KYUTAI_REPO, repo_type="dataset")
            if name.startswith("train/") and name.endswith(".jsonl.gz")
        )
        if not names:
            raise SystemExit(f"no train shards found in {KYUTAI_REPO}")
        hf_hub_download(NVIDIA_REPO, CHAPTERS_REMOTE, repo_type="dataset", local_dir=str(nvidia))
        for name in [EVAL_REMOTE, *names]:
            hf_hub_download(KYUTAI_REPO, name, repo_type="dataset", local_dir=str(kyutai))
        marker.write_text("ok\n")
    shards = sorted((kyutai / "train").glob("*.jsonl.gz"))
    if not chapters.is_file() or not eval_path.is_file() or not shards:
        raise SystemExit(f"{meta_dir} is incomplete; delete {marker} to download it again")
    return chapters, shards, eval_path


def chapter_hours(shards: list[Path], settings: dict, meta_dir: Path, workers: int):
    """Usable seconds per chapter over the whole corpus, cached next to the metadata."""
    fingerprint = hashlib.sha256(
        json.dumps(
            [settings, [(s.name, s.stat().st_size) for s in shards]], sort_keys=True
        ).encode()
    ).hexdigest()[:16]
    cache = meta_dir / f"chapter_hours-{fingerprint}.json"
    if cache.is_file():
        data = json.loads(cache.read_text())
        return data["seconds"], data["dropped"]
    print(f"scanning {len(shards)} alignment shards for per-chapter hours ...", flush=True)
    seconds: dict[str, float] = {}
    dropped: Counter = Counter()
    started = time.monotonic()
    scan = partial(scan_shard, settings=settings)
    for done, (part, part_dropped) in enumerate(
        parallel(scan, [str(s) for s in shards], workers), 1
    ):
        for key, value in part.items():
            seconds[key] = seconds.get(key, 0.0) + value
        dropped.update(part_dropped)
        print(
            f"  shard {done}/{len(shards)} scanned, elapsed "
            f"{format_seconds(time.monotonic() - started)}",
            flush=True,
        )
    cache.write_text(json.dumps({"seconds": seconds, "dropped": dict(dropped)}))
    return seconds, dict(dropped)


def chapter_order(seconds: dict[str, float]) -> list[str]:
    """Every chapter, in the fixed order selections are prefixes of."""
    return sorted(seconds, key=stable_bucket)


def take_hours(order: list[str], seconds: dict[str, float], hours: float, skip=()) -> list[str]:
    """The shortest prefix of `order` (minus `skip`) that reaches `hours`, or all of it."""
    chosen, total = [], 0.0
    for key in order:
        if key in skip:
            continue
        chosen.append(key)
        total += seconds[key] / 3600
        if total >= hours:
            break
    return chosen


def load_eval(path: Path, settings: dict, count: int) -> list[dict]:
    rows = []
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            parsed, _ = usable_row(json.loads(line), settings)
            if parsed is not None:
                rows.append(parsed)
    rows.sort(key=lambda row: stable_bucket(row["audio_filepath"]))
    return rows[:count]


def read_chapters(path: Path, wanted: set[str]) -> dict[str, dict]:
    """url + utterance offsets of the `wanted` chapters, from the 1.7 GB chapter list."""
    found: dict[str, dict] = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            chapter = json.loads(line)
            key = chapter_key_of_path(chapter["chapter_filepath"])
            if key in wanted:
                found[key] = {
                    "url": chapter["url"],
                    "duration": float(chapter["duration"]),
                    "offsets": {
                        u["audio_filepath"]: (float(u["offset"]), float(u["duration"]))
                        for u in chapter["utterances"]
                    },
                }
                if len(found) == len(wanted):
                    break
    return found


def decode_chapter(mp3: Path):
    """The chapter as 24 kHz mono float32, decoded the way Kyutai's own downloader does."""
    import sphn

    try:
        wav, rate = sphn.read(str(mp3))
        wav = wav.mean(axis=0)
    except Exception:  # noqa: BLE001 -- any decoder failure falls back to ffmpeg
        # a few LibriVox MP3s have broken headers that symphonia rejects; ffmpeg copes
        converted = mp3.with_suffix(".wav")
        subprocess.run(
            ["ffmpeg", "-nostdin", "-y", "-loglevel", "error", "-i", str(mp3)]
            + ["-ac", "1", "-ar", str(SAMPLE_RATE), str(converted)],
            check=True,
            timeout=900,
        )
        try:
            wav, rate = sphn.read(str(converted))
            wav = wav.mean(axis=0)
        finally:
            converted.unlink(missing_ok=True)
    if rate != SAMPLE_RATE:
        wav = sphn.resample(wav, src_sample_rate=rate, dst_sample_rate=SAMPLE_RATE)
    return wav


def fetch_audio(url: str, destination: Path) -> None:
    from training.scripts.prepare_data import download

    download(url.replace("http://", "https://"), destination)


def utterance_path(audio_dir: Path, audio_filepath: str, audio_format: str) -> Path:
    return audio_dir / Path(audio_filepath).with_suffix(audio_suffix(audio_format))


def process_chapter(job: dict) -> dict:
    """Make sure every utterance of one chapter exists as its own audio file.

    Utterance files already on disk are kept. The chapter MP3 is fetched only when something
    is missing, and deleted as soon as it has been cut."""
    import soundfile

    audio_dir, fmt = Path(job["audio_dir"]), job["audio_format"]
    targets = {name: utterance_path(audio_dir, name, fmt) for name in job["utterances"]}
    missing = [name for name, target in targets.items() if not target.is_file()]
    if missing:
        mp3 = Path(job["tmp_dir"]) / (hashlib.sha1(job["key"].encode()).hexdigest() + ".mp3")
        mp3.parent.mkdir(parents=True, exist_ok=True)
        try:
            fetch_audio(job["url"], mp3)
            wav = decode_chapter(mp3)
            for name in missing:
                offset, length = job["utterances"][name]
                first = int(offset * SAMPLE_RATE)
                segment = wav[first : first + int(length * SAMPLE_RATE)]
                if len(segment) >= SAMPLE_RATE // 4:
                    write_audio(segment, targets[name], fmt)
        except Exception as error:  # noqa: BLE001 -- dead LibriVox links happen
            return {
                "key": job["key"],
                "status": "failed",
                "error": f"{type(error).__name__}: {error}",
            }
        finally:
            mp3.unlink(missing_ok=True)
    durations = {}
    for name, target in targets.items():
        if target.is_file():
            info = soundfile.info(str(target))
            durations[name] = info.frames / info.samplerate
    return {"key": job["key"], "status": "ok" if missing else "reused", "durations": durations}


def run_jobs(jobs: list[dict], workers: int, label: str) -> list[dict]:
    results: list[dict] = []
    if not jobs:
        return results
    started = time.monotonic()
    last = started
    failures = 0
    for done, result in enumerate(parallel(process_chapter, jobs, workers), 1):
        results.append(result)
        failures += result["status"] == "failed"
        now = time.monotonic()
        if done == len(jobs) or now - last >= PROGRESS_SECONDS:
            last = now
            elapsed = now - started
            eta = elapsed * (len(jobs) - done) / done
            print(
                f"{label}: {done}/{len(jobs)} chapters ({100 * done / len(jobs):.1f}%), "
                f"{failures} failed, elapsed {format_seconds(elapsed)}, "
                f"ETA {format_seconds(eta)}",
                flush=True,
            )
    return results


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    settings = {
        "min_duration": args.min_duration,
        "max_duration": args.max_duration,
        "max_wer": args.max_wer,
    }
    prepare_output(
        args.out_dir,
        {
            "builder": "hifitts2",
            "layout": LAYOUT_VERSION,
            "audio_format": args.audio_format,
            "sample_rate": SAMPLE_RATE,
        },
        assume_yes=args.yes,
    )
    audio_dir = args.out_dir / "audio"
    tmp_dir = args.out_dir / "tmp"
    shutil.rmtree(tmp_dir, ignore_errors=True)

    chapters_path, shards, eval_path = ensure_meta(args.meta_dir)
    seconds, dropped = chapter_hours(shards, settings, args.meta_dir, args.workers)
    pool_hours = sum(seconds.values()) / 3600
    print(
        f"corpus after filters: {pool_hours:.0f} h in {len(seconds)} chapters "
        f"(dropped rows: {dict(dropped)})"
    )
    if args.hours > pool_hours:
        print(f"note: --hours {args.hours:g} exceeds the {pool_hours:.0f} h available; using all")

    order = chapter_order(seconds)
    pool = take_hours(order, seconds, args.hours * POOL_MARGIN)
    valid_rows = load_eval(eval_path, settings, args.valid_utterances)
    valid_chapters = {chapter_key(row["audio_filepath"]) for row in valid_rows}
    wanted = frozenset(pool) | valid_chapters
    print(
        f"collecting alignments of {len(pool)} chapters (+{len(valid_chapters - set(pool))} "
        "validation-only) ...",
        flush=True,
    )
    rows_by_name: dict[str, dict] = {}
    collect = partial(collect_shard, settings=settings, wanted=wanted)
    for part in parallel(collect, [str(s) for s in shards], args.workers):
        rows_by_name.update(part)
    for row in valid_rows:
        rows_by_name.setdefault(row["audio_filepath"], row)
    valid_names = {row["audio_filepath"] for row in valid_rows}
    train_by_chapter: dict[str, list[str]] = {}
    for name in rows_by_name:
        if name not in valid_names:
            train_by_chapter.setdefault(chapter_key(name), []).append(name)

    chapter_info = read_chapters(chapters_path, wanted)
    missing_chapters = wanted - set(chapter_info)
    too_long = {
        key
        for key, info in chapter_info.items()
        if info["duration"] > args.max_chapter_hours * 3600
    }
    # Chapters we will not fetch count as failed, so the pool replaces them.
    unavailable = missing_chapters | too_long
    if unavailable:
        print(
            f"note: skipping {len(missing_chapters)} chapters missing from the chapter list and "
            f"{len(too_long)} longer than {args.max_chapter_hours:g} h (decoded whole in memory)"
        )

    def job_for(key: str, names: list[str]) -> dict | None:
        info = chapter_info.get(key)
        if info is None:
            return None
        utterances = {n: info["offsets"][n] for n in names if n in info["offsets"]}
        if not utterances:
            return None
        return {
            "key": key,
            "url": info["url"],
            "utterances": utterances,
            "audio_dir": str(audio_dir),
            "audio_format": args.audio_format,
            "tmp_dir": str(tmp_dir),
        }

    valid_by_chapter: dict[str, list[str]] = {}
    for name in valid_names:
        valid_by_chapter.setdefault(chapter_key(name), []).append(name)

    def jobs_for(keys: list[str]) -> list[dict]:
        jobs = []
        for key in keys:
            names = train_by_chapter.get(key, []) + valid_by_chapter.get(key, [])
            job = job_for(key, names)
            if job is not None:
                jobs.append(job)
        return jobs

    # What this run will fetch, before it fetches any audio.
    chosen = take_hours(pool, seconds, args.hours, skip=unavailable)
    first_jobs = jobs_for(list(dict.fromkeys([*chosen, *sorted(valid_chapters)])))
    todo = [
        job
        for job in first_jobs
        if any(
            not utterance_path(audio_dir, name, args.audio_format).is_file()
            for name in job["utterances"]
        )
    ]
    new_hours = sum(chapter_info[job["key"]]["duration"] for job in todo) / 3600
    new_utterance_hours = (
        sum(rows_by_name[name]["duration"] for job in todo for name in job["utterances"]) / 3600
    )
    need_gb = new_utterance_hours * FLAC_MB_PER_HOUR / 1000
    free_gb = shutil.disk_usage(args.out_dir).free / 1e9
    print(
        f"plan: {len(todo)} of {len(first_jobs)} chapters still to fetch ({new_hours:.0f} h of "
        f"chapter audio, ~{new_hours * MP3_MB_PER_HOUR / 1000:.0f} GB of MP3 traffic, deleted "
        f"after cutting); ~{need_gb:.0f} GB of new {args.audio_format.upper()}; "
        f"{free_gb:.0f} GB free in {args.out_dir}",
        flush=True,
    )
    if args.audio_format == "wav":
        print("note: wav is ~2.5x the size of flac; the estimate above is for flac")
    if need_gb > free_gb:
        print("WARNING: not enough free space for the new audio", file=sys.stderr)

    # Fetch, replacing chapters that fail from the same ordered pool until the target is met.
    # A chapter is done once the utterances asked of it have been produced; a validation-only
    # chapter that a later round also needs for train is simply asked again for the rest.
    done: dict[str, dict] = {}
    asked: dict[str, set[str]] = {}
    failed: set[str] = set(unavailable)
    while True:
        selected = take_hours(pool, seconds, args.hours, skip=failed)
        keys = [
            key for key in dict.fromkeys([*selected, *sorted(valid_chapters)]) if key not in failed
        ]
        jobs = [
            job
            for job in jobs_for(keys)
            if not set(job["utterances"]) <= asked.get(job["key"], set())
        ]
        if not jobs:
            break
        for job in jobs:
            asked[job["key"]] = asked.get(job["key"], set()) | set(job["utterances"])
        for result in run_jobs(jobs, args.workers, "chapters"):
            if result["status"] == "failed":
                failed.add(result["key"])
                print(f"  skipped {result['key']}: {result['error']}", file=sys.stderr)
            else:
                done.setdefault(result["key"], {"durations": {}})["durations"].update(
                    result["durations"]
                )
    shutil.rmtree(tmp_dir, ignore_errors=True)

    selected = take_hours(pool, seconds, args.hours, skip=failed)
    stats: Counter = Counter()

    def build(names: list[str], key: str) -> list[dict]:
        result = done.get(key)
        rows = []
        for name in names:
            length = result["durations"].get(name) if result else None
            if length is None:
                stats["no_audio"] += 1
                continue
            row = rows_by_name[name]
            try:
                rows.append(
                    make_row(
                        utterance_path(audio_dir, name, args.audio_format).resolve(),
                        length,
                        row["words"],
                        row["speaker"],
                        dataset="hifitts2",
                    )
                )
            except ValueError:
                stats["bad_row"] += 1
        return rows

    train: list[dict] = []
    for key in selected:
        train += build(train_by_chapter.get(key, []), key)
    valid: list[dict] = []
    for key in sorted(valid_chapters):
        valid += build(valid_by_chapter.get(key, []), key)
    if not train:
        raise SystemExit("no usable training utterances were produced")

    train_path, valid_path = write_manifests(args.out_dir, train, valid)
    print(
        f"chapters: {len(selected)} used, {len(failed)} unavailable (dead link, missing, or over the length cap)"
    )
    print(f"dropped at build: {dict(stats)}")
    describe_split("train", train)
    describe_split("valid", valid)
    train_hours = sum(row["duration"] for row in train) / 3600
    if train_hours < args.hours * 0.98:
        print(
            f"note: {train_hours:.1f} h of the requested {args.hours:g} h "
            "(the available pool or too many dead links)"
        )
    overlap = {r["speaker"] for r in train} & {r["speaker"] for r in valid}
    print(
        f"valid speakers also in train: {len(overlap)}/{len({r['speaker'] for r in valid})} "
        "(Kyutai's eval split is not speaker-disjoint)"
    )
    print_duration_report(train)
    print(f"wrote {train_path} and {valid_path}")


if __name__ == "__main__":
    main()
