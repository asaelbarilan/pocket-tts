"""Build a self-contained LibriTTS-R (English) manifest for Kyutai's trainer.

Purpose: keep some of the base model's English while it learns Hebrew. LibriTTS-R (585 h,
24 kHz, restored with Miipher) is the clean half of the English mix; build_hifitts2_manifest.py
is the diverse half.

    python -m hebrew_training.build_libritts_r_manifest --out-dir data/english/libritts_r --hours 250

Everything training needs ends up in --out-dir; nothing is fetched while training:

    train.jsonl, valid.jsonl   rows in the format described in build_official_manifest.py
    audio/                     one 24 kHz mono FLAC (or WAV) per utterance, `start` is 0
    rows/                      one small file per processed parquet shard (the reuse ledger)
    meta/                      shard sizes and the word alignments (~130 MB)
    build_state.json           what the audio was built with (see prepare_output)

Where each part comes from
    audio + text   mythicinfinity/libritts_r, LibriTTS-R as parquet shards per subset. Shards are
                   downloaded one at a time and deleted once their utterances are written out, so
                   only the output counts toward disk. `text_normalized` (cased, punctuated) is
                   what the rows are made from.
    word times     cdminix/libritts-r-aligned: Montreal Forced Aligner TextGrids per utterance.
                   MFA lowercases and strips punctuation, so its words are mapped back onto the
                   tokens of `text_normalized` (see map_words). A hyphenated word that MFA split in
                   two keeps both intervals. A row whose words do not line up exactly is dropped and
                   counted, never guessed.

Choosing the hours
    --hours is the target for TRAIN, in hours after the filters below. The clean subsets
    (train.clean.100 + train.clean.360, ~245 h before filtering) are the default pool; "other" is
    the noisy half of the corpus and is only used if you add it with --train-subsets. Shards are
    taken in a fixed pseudo-random order, so a smaller selection is a prefix of a larger one:
    raising --hours fetches only the extra shards. A shard is a handful of speakers, so this is a
    random sample of whole speakers. If the pool holds less than --hours after filtering, all of
    it is used and the shortfall is printed.

    Disk: 24 kHz audio is ~0.16 GB per hour as parquet (the transient download, one shard at a
    time per worker, ~0.5 GiB each) and ~0.09 GB per hour as FLAC (the output, measured): 250 h is
    ~23 GB. The script prints its own estimate before it downloads a shard.

Validation is --valid-subset (dev.clean), speaker-disjoint from train by the corpus's own split;
the script checks it.

Rows under --min-duration (2 s: the loader needs 1 s on each side of a cut) are dropped.
"""

from __future__ import annotations

import argparse
import io
import json
import os
import re
import shutil
import tarfile
import time
import unicodedata
from collections import Counter
from pathlib import Path

from hebrew_training.official_manifest_common import (
    AUDIO_FORMATS,
    SAMPLE_RATE,
    audio_suffix,
    describe_split,
    format_seconds,
    make_row,
    parallel,
    prepare_output,
    print_duration_report,
    stable_bucket,
    write_audio,
    write_manifests,
)

AUDIO_REPO = "mythicinfinity/libritts_r"
ALIGN_REPO = "cdminix/libritts-r-aligned"
LAYOUT_VERSION = 1
DEFAULT_HOURS = 250.0
DEFAULT_TRAIN_SUBSETS = "train.clean.100,train.clean.360"
WAV_BYTES_PER_HOUR = SAMPLE_RATE * 2 * 3600
FLAC_GB_PER_HOUR = 0.09
# What share of a shard's hours survives the filters. Only used to size the first batch of
# shards; every batch after it uses the yield actually measured.
FIRST_BATCH_YIELD = 0.9
ALIGN_TOLERANCE_SEC = 0.15
PROGRESS_SECONDS = 15.0

_DASHES = re.compile(r"[-\u2010-\u2015]+")
_INTERVAL = re.compile(
    r"xmin = ([0-9.eE+-]+)\s*\n\s*xmax = ([0-9.eE+-]+)\s*\n\s*text = \"(.*?)\"\s*\n", re.DOTALL
)
_XMAX = re.compile(r"xmax = ([0-9.eE+-]+)")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a self-contained LibriTTS-R manifest (audio + alignments) for training."
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
        "--valid-hours",
        type=float,
        default=2.0,
        help="Hours of validation, taken from --valid-subset in whole shards (default: %(default)s).",
    )
    parser.add_argument(
        "--train-subsets",
        default=DEFAULT_TRAIN_SUBSETS,
        help="Comma-separated LibriTTS-R subsets to draw train from (default: %(default)s; "
        "add train.other.500 for the noisier half).",
    )
    parser.add_argument("--valid-subset", default="dev.clean")
    parser.add_argument("--min-duration", type=float, default=2.0)
    parser.add_argument("--max-duration", type=float, default=30.0)
    parser.add_argument("--audio-format", choices=AUDIO_FORMATS, default="flac")
    parser.add_argument(
        "--workers",
        type=int,
        default=min(4, os.cpu_count() or 1),
        help="Shards processed in parallel; each holds one ~0.5 GiB shard in memory.",
    )
    parser.add_argument(
        "--meta-dir", type=Path, help="Shard sizes and alignments (default: OUT_DIR/meta)."
    )
    parser.add_argument(
        "--yes",
        action="store_true",
        help="Do not ask before deleting an incompatible earlier build in --out-dir.",
    )
    args = parser.parse_args(argv)
    if args.hours <= 0 or args.valid_hours <= 0:
        parser.error("--hours and --valid-hours must be positive")
    if args.workers < 1:
        parser.error("--workers must be at least 1")
    if args.min_duration < 2.0:
        parser.error("--min-duration below 2.0 s leaves rows the loader cannot cut")
    args.train_subsets = [s.strip() for s in args.train_subsets.split(",") if s.strip()]
    if args.valid_subset in args.train_subsets:
        parser.error("--valid-subset must not also be a train subset")
    args.meta_dir = args.meta_dir or args.out_dir / "meta"
    return args


def word_key(text: str) -> str:
    """The comparison form of a word: what MFA and the corpus text have in common."""
    text = unicodedata.normalize("NFKD", text.replace("\u2019", "'")).lower()
    return re.sub(r"[^a-z0-9']", "", text).strip("'")


def map_words(tokens: list[str], mfa: list[tuple[float, float, str]]) -> list[dict] | None:
    """Put MFA's word times on the corpus text's own tokens, or None if they do not line up.

    `tokens` are the whitespace-separated words of `text_normalized`, with their case and
    punctuation. `mfa` are (start, end, text) intervals; silence is "". A token made of several
    words joined by dashes takes one MFA interval per part and spans all of them. A token with
    no letters or digits (a lone dash) has nothing to align: it becomes a zero-width word at the
    end of the previous one, so the text is kept as written.
    """
    spoken = [interval for interval in mfa if interval[2] != ""]
    out: list[dict] = []
    position = 0
    for token in tokens:
        parts = [key for key in (word_key(part) for part in _DASHES.split(token)) if key]
        if not parts:
            at = out[-1]["end"] if out else 0.0
            out.append({"word": token, "start": at, "end": at})
            continue
        chunk = spoken[position : position + len(parts)]
        if [word_key(text) for _, _, text in chunk] != parts:
            return None
        out.append({"word": token, "start": chunk[0][0], "end": chunk[-1][1]})
        position += len(parts)
    return out if position == len(spoken) else None


def parse_textgrid(text: str) -> tuple[float, list[tuple[float, float, str]]]:
    """(file end time, the intervals of the `words` tier) of one MFA TextGrid."""
    header = _XMAX.search(text)
    start = text.find('name = "words"')
    stop = text.find('name = "phones"', start)
    if header is None or start < 0:
        raise ValueError("not a TextGrid with a words tier")
    tier = text[start : stop if stop >= 0 else len(text)]
    intervals = [(float(a), float(b), word) for a, b, word in _INTERVAL.findall(tier)]
    return float(header.group(1)), intervals


def read_textgrids(tarball: Path, wanted: set[str]) -> dict[str, tuple[float, list]]:
    """The parsed TextGrids of the `wanted` utterance ids, streamed out of one tarball."""
    found: dict[str, tuple[float, list]] = {}
    with tarfile.open(tarball, "r|gz") as archive:
        for member in archive:
            if not member.name.endswith(".TextGrid"):
                continue
            name = Path(member.name).name[: -len(".TextGrid")]
            if name not in wanted:
                continue
            handle = archive.extractfile(member)
            if handle is None:
                continue
            try:
                found[name] = parse_textgrid(handle.read().decode("utf-8"))
            except ValueError:
                continue
            if len(found) == len(wanted):
                break
    return found


def tarball_of(subset: str) -> str:
    return f"data/{subset.replace('.', '_')}.tar.gz"


def list_shards(subsets: list[str], meta_dir: Path) -> dict[str, list[tuple[str, int]]]:
    """(repo path, bytes) of every parquet shard of each subset, cached next to the alignments."""
    cache = meta_dir / "shards.json"
    known = json.loads(cache.read_text()) if cache.is_file() else {}
    missing = [s for s in subsets if s not in known]
    if missing:
        from huggingface_hub import HfApi

        api = HfApi()
        for subset in missing:
            entries = api.list_repo_tree(
                AUDIO_REPO, path_in_repo=f"data/{subset}", repo_type="dataset"
            )
            known[subset] = sorted((e.path, e.size) for e in entries if e.path.endswith(".parquet"))
            if not known[subset]:
                raise SystemExit(f"no parquet shards for subset {subset!r} in {AUDIO_REPO}")
        meta_dir.mkdir(parents=True, exist_ok=True)
        cache.write_text(json.dumps(known, indent=1))
    return {s: [tuple(e) for e in known[s]] for s in subsets}


def ensure_alignments(subsets: list[str], meta_dir: Path) -> dict[str, Path]:
    from huggingface_hub import hf_hub_download

    paths = {}
    for subset in subsets:
        path = meta_dir / "align" / tarball_of(subset)
        if not path.is_file():
            hf_hub_download(
                ALIGN_REPO,
                tarball_of(subset),
                repo_type="dataset",
                local_dir=str(meta_dir / "align"),
            )
        paths[subset] = path
    return paths


def shard_hours(size: int) -> float:
    return size / WAV_BYTES_PER_HOUR


def process_shard(job: dict) -> dict:
    """Turn one parquet shard into per-utterance audio files and manifest rows.

    The rows are kept in rows/<shard>.json. A shard whose file is already there, made with the
    same settings, is not downloaded again."""
    import soundfile

    out_dir = Path(job["out_dir"])
    name = job["shard"].replace("/", "__")
    ledger = out_dir / "rows" / f"{name}.json"
    signature = job["signature"]
    if ledger.is_file():
        data = json.loads(ledger.read_text(encoding="utf-8"))
        if data["signature"] == signature:
            return {**data, "status": "reused"}

    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download
    from huggingface_hub.utils import disable_progress_bars

    disable_progress_bars()
    tmp = out_dir / "tmp"
    local = Path(
        hf_hub_download(AUDIO_REPO, job["shard"], repo_type="dataset", local_dir=str(tmp / name))
    )
    try:
        parquet = pq.ParquetFile(local)
        ids = set(parquet.read(columns=["id"]).column("id").to_pylist())
        grids = read_textgrids(Path(job["tarball"]), ids)
        rows: list[dict] = []
        dropped: Counter = Counter()
        suffix = audio_suffix(signature["audio_format"])
        columns = ["id", "audio", "text_normalized", "speaker_id", "chapter_id"]
        for batch in parquet.iter_batches(batch_size=32, columns=columns):
            for record in batch.to_pylist():
                samples, rate = soundfile.read(
                    io.BytesIO(record["audio"]["bytes"]), dtype="float32"
                )
                if samples.ndim > 1:
                    samples = samples.mean(axis=1)
                if rate != SAMPLE_RATE:
                    dropped["bad_sample_rate"] += 1
                    continue
                duration = len(samples) / rate
                if duration < signature["min_duration"]:
                    dropped["too_short"] += 1
                    continue
                if duration > signature["max_duration"]:
                    dropped["too_long"] += 1
                    continue
                grid = grids.get(record["id"])
                if grid is None:
                    dropped["no_alignment"] += 1
                    continue
                if abs(grid[0] - duration) > ALIGN_TOLERANCE_SEC:
                    dropped["misaligned"] += 1
                    continue
                words = map_words(record["text_normalized"].split(), grid[1])
                if words is None:
                    dropped["text_mismatch"] += 1
                    continue
                relative = (
                    Path("audio")
                    / record["speaker_id"]
                    / record["chapter_id"]
                    / (record["id"] + suffix)
                )
                try:
                    row = make_row(
                        relative, duration, words, record["speaker_id"], dataset="libritts-r"
                    )
                except ValueError:
                    dropped["bad_words"] += 1
                    continue
                target = out_dir / relative
                if not target.is_file():
                    write_audio(samples, target, signature["audio_format"])
                rows.append(row)
    finally:
        shutil.rmtree(tmp / name, ignore_errors=True)
    data = {"signature": signature, "rows": rows, "dropped": dict(dropped), "seen": len(ids)}
    ledger.parent.mkdir(parents=True, exist_ok=True)
    partial = ledger.with_name(f".{ledger.name}.partial")
    partial.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    partial.replace(ledger)
    return {**data, "status": "built"}


def run_shards(jobs: list[dict], workers: int, label: str) -> list[dict]:
    results: list[dict] = []
    started = last = time.monotonic()
    for done, result in enumerate(parallel(process_shard, jobs, workers), 1):
        results.append(result)
        now = time.monotonic()
        if done == len(jobs) or now - last >= PROGRESS_SECONDS:
            last = now
            elapsed = now - started
            print(
                f"{label}: {done}/{len(jobs)} shards, elapsed {format_seconds(elapsed)}, "
                f"ETA {format_seconds(elapsed * (len(jobs) - done) / done)}",
                flush=True,
            )
    return results


def fill_target(order, hours_of, make_job, target: float, workers: int, label: str, dropped):
    """Process shards from `order` until `target` hours of rows exist, or the pool runs out.

    The first batch is sized by an assumed yield; each later batch by the yield measured so far.
    Returns the rows, and the number of shards used."""
    rows: list[dict] = []
    built_estimate = 0.0
    cursor = 0
    yield_ratio = FIRST_BATCH_YIELD
    while cursor < len(order) and sum(r["duration"] for r in rows) / 3600 < target:
        shortfall = target - sum(r["duration"] for r in rows) / 3600
        batch, estimate = [], 0.0
        while cursor < len(order) and estimate < shortfall / yield_ratio:
            batch.append(order[cursor])
            estimate += hours_of(order[cursor])
            cursor += 1
        for result in run_shards([make_job(shard) for shard in batch], workers, label):
            rows += result["rows"]
            dropped.update(result["dropped"])
        built_estimate += estimate
        yield_ratio = max(0.3, sum(r["duration"] for r in rows) / 3600 / built_estimate)
    return rows, cursor


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    signature = {
        "min_duration": args.min_duration,
        "max_duration": args.max_duration,
        "audio_format": args.audio_format,
    }
    prepare_output(
        args.out_dir,
        {
            "builder": "libritts-r",
            "layout": LAYOUT_VERSION,
            "audio_format": args.audio_format,
            "sample_rate": SAMPLE_RATE,
        },
        assume_yes=args.yes,
    )
    shutil.rmtree(args.out_dir / "tmp", ignore_errors=True)
    subsets = [*args.train_subsets, args.valid_subset]
    shards = list_shards(subsets, args.meta_dir)
    tarballs = ensure_alignments(subsets, args.meta_dir)

    sizes = {path: size for subset in subsets for path, size in shards[subset]}
    subset_of = {path: subset for subset in subsets for path, _ in shards[subset]}
    train_order = sorted(
        (path for subset in args.train_subsets for path, _ in shards[subset]), key=stable_bucket
    )
    valid_order = sorted((path for path, _ in shards[args.valid_subset]), key=stable_bucket)

    def job(path: str) -> dict:
        return {
            "shard": path,
            "tarball": str(tarballs[subset_of[path]]),
            "out_dir": str(args.out_dir),
            "signature": signature,
        }

    def hours_of(path: str) -> float:
        return shard_hours(sizes[path])

    pool_hours = sum(hours_of(p) for p in train_order)
    first_batch, covered = [], 0.0
    for path in train_order:
        if covered >= args.hours / FIRST_BATCH_YIELD:
            break
        first_batch.append(path)
        covered += hours_of(path)
    fresh = [
        p
        for p in first_batch
        if not (args.out_dir / "rows" / f"{p.replace('/', '__')}.json").is_file()
    ]
    download_gb = sum(sizes[p] for p in fresh) / 2**30
    print(
        f"plan: ~{pool_hours:.0f} h in the train pool ({len(train_order)} shards, before filters); "
        f"about {len(fresh)} shards (~{download_gb:.1f} GiB, deleted after each shard) still to "
        f"download, ~{sum(hours_of(p) for p in fresh) * FLAC_GB_PER_HOUR:.1f} GB of new "
        f"{args.audio_format.upper()}; {shutil.disk_usage(args.out_dir).free / 1e9:.0f} GB free",
        flush=True,
    )
    if args.audio_format == "wav":
        print("note: wav is ~2.5x the size of flac; the estimate above is for flac")

    dropped: Counter = Counter()
    train, used = fill_target(
        train_order, hours_of, job, args.hours, args.workers, "train", dropped
    )
    valid_dropped: Counter = Counter()
    valid, _ = fill_target(
        valid_order, hours_of, job, args.valid_hours, args.workers, "valid", valid_dropped
    )
    shutil.rmtree(args.out_dir / "tmp", ignore_errors=True)

    for rows in (train, valid):
        for row in rows:
            row["path"] = str((args.out_dir / row["path"]).resolve())
    if not train or not valid:
        raise SystemExit("no usable utterances were produced")
    overlap = {r["speaker"] for r in train} & {r["speaker"] for r in valid}
    if overlap:
        raise SystemExit(
            f"{len(overlap)} speakers are in both train and valid: {sorted(overlap)[:5]}"
        )

    train_path, valid_path = write_manifests(args.out_dir, train, valid)
    print(f"shards used: {used} of {len(train_order)} train shards")
    print(f"dropped (train): {dict(dropped)}")
    describe_split("train", train)
    describe_split("valid", valid)
    train_hours = sum(row["duration"] for row in train) / 3600
    if train_hours < args.hours * 0.98:
        print(
            f"note: the pool holds {train_hours:.1f} h after filtering, below the requested "
            f"{args.hours:g} h; add train.other.500 to --train-subsets for more"
        )
    print_duration_report(train)
    print(f"wrote {train_path} and {valid_path}")


if __name__ == "__main__":
    main()
