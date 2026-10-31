"""Build a manifest from ivrit-ai/crowd-transcribe-v5 (audio + text, no word times).

    python -m hebrew_training.build_crowd_transcribe_manifest --out-dir data/crowd --normalize-text
    python hebrew_training/align_official_manifest.py data/crowd --strict --batch-size 64 ...

The corpus is 205,491 distinct clips (299 h) of Israeli podcasts and lectures, cut at Whisper
segment boundaries, with the sentence a crowd worker corrected. Median clip 3.8 s, p90 9.2 s,
minimum ~2.1 s. It ships MP3s (32-48 kHz stereo), a transcript, and quality flags; no word
times, no speaker id, no offsets inside the episode.

What that means for training (measured on the metadata, see HEBREW_TRAINING_RECIPE.md)
    No word times. Rows are written with PLACEHOLDER times (`"timing": "estimated"`, spread over
    the clip by word length) so the aligner can read them; they are not a usable alignment. Run
    `align_official_manifest.py --strict`, which keeps only Viter results (no MMS, no retained
    "original" row) and also drops rows the loader could not cut. Train on `*_aligned.jsonl`.
    No speaker id. The id of a clip is `<show>/<episode>/<index>`. `speaker` is the episode (what
    Viter adapts to); the split is by SHOW, because hosts recur across a show's episodes. The
    dataset's own test split holds two shows (SmallBigHistory, Mindset) that appear nowhere in
    train, so it is used as validation (subsampled to --valid-hours).
    Short clips. training/dataloader.py cuts a row at a word boundary leaving >= 1.0 s on each
    side, so a row needs more than 2 s of aligned speech, and the training target is only what
    follows the cut. Neighbouring clips cannot be merged to make longer rows: only 5% of clips
    have an adjacent index in the sample. --min-duration 3 (the default) keeps the rows that are
    cuttable after Viter trims the silence at both ends; the aligner's --strict drops the rest.
    Duplicates. 17k clips were transcribed by more than one person. They are one row each, and by
    default (--duplicates agree) a clip whose transcribers disagree is dropped.

Everything training needs ends up in --out-dir (nothing is fetched while training):

    train.jsonl, valid.jsonl   rows in the format of build_official_manifest.py, with the extra
                               keys `clip` (the dataset uuid), `show` and `timing`
    audio/<episode hash>/<index>.flac   one 24 kHz mono FLAC per clip, `start` is 0
    meta/                      the scanned metadata (the shards are deleted after use unless
                               --keep-shards)
    build_state.json           what the audio was built with (see prepare_output)

The dataset is gated: accept its terms on Hugging Face and log in first. Pass --source-dir with
already downloaded shards to avoid downloading ~20 GB again.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
import os
import re
import shutil
import sys
import time
from collections import Counter
from pathlib import Path

from hebrew_training.build_official_manifest import (
    DEFAULT_NORMALIZER,
    load_normalizer,
    normalize_words_with_alternates,
    validate_alt_words,
)
from hebrew_training.official_manifest_common import (
    AUDIO_FORMATS,
    MIN_CUT_SEC,
    SAMPLE_RATE,
    audio_suffix,
    clean,
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
from pocket_tts.utils.text_fold import fold_characters

REPO = "ivrit-ai/crowd-transcribe-v5"
LAYOUT_VERSION = 1
DATASET = "crowd-transcribe-v5"
INDEX_FILE = "index.json.gz"
PROGRESS_SECONDS = 15.0
# extra_data flags the annotators set; 78 of 224,538 rows have one, but they are free to drop.
EXCLUDE_FLAGS = (
    "foreign_language",
    "multiple_speakers",
    "noisy",
    "skipped",
    "unintelligible",
    "too_long",
)

_HEBREW = re.compile(r"[֐-׿]")
_HAS_SPEECH = re.compile(r"[^\W_]")
# What align_official_manifest.clean_viter_word strips, and the alphabet it can then spell
# (split_graphemes). A word outside it is routed to MMS, which --strict turns off, so such a row
# is dropped here rather than after its audio has been exported. tests/ pins the two together.
_VITER_PUNCT = re.compile(r"[^\w֐-׿']+")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a manifest (audio + text, placeholder word times) from "
        "ivrit-ai/crowd-transcribe-v5."
    )
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument(
        "--source-dir",
        type=Path,
        help="Folder holding the dataset's parquet shards (any depth). Without it the shards "
        "are downloaded from Hugging Face one at a time.",
    )
    parser.add_argument(
        "--train-hours",
        type=float,
        help="Keep about this many hours of train, in whole episodes taken in a fixed "
        "pseudo-random order (a smaller selection is a prefix of a larger one). Default: all.",
    )
    parser.add_argument("--valid-hours", type=float, default=2.0)
    parser.add_argument(
        "--min-duration",
        type=float,
        default=3.0,
        help="Shortest clip kept. The loader needs >2 s of speech after Viter trims the ends, "
        "so clips under ~3 s mostly fail the aligner's cut check.",
    )
    parser.add_argument("--max-duration", type=float, default=30.0)
    parser.add_argument(
        "--duplicates",
        choices=("agree", "first"),
        default="agree",
        help="A clip transcribed by several people: drop it unless the transcripts agree "
        "(default), or just take one (a re-transcription when there is one).",
    )
    parser.add_argument(
        "--normalize-text",
        action="store_true",
        help="Apply the Hebrew TTS normalizer (numbers, abbreviations), as the other builders do.",
    )
    parser.add_argument("--normalizer-dir", type=Path, default=DEFAULT_NORMALIZER)
    parser.add_argument("--audio-format", choices=AUDIO_FORMATS, default="flac")
    parser.add_argument("--workers", type=int, default=min(4, os.cpu_count() or 1))
    parser.add_argument(
        "--limit-shards",
        type=int,
        help="Use only the first N train and N test shards (smoke test).",
    )
    parser.add_argument(
        "--keep-shards", action="store_true", help="Do not delete downloaded shards after use."
    )
    parser.add_argument(
        "--yes",
        action="store_true",
        help="Do not ask before deleting an incompatible earlier build in --out-dir.",
    )
    args = parser.parse_args(argv)
    if args.workers < 1:
        parser.error("--workers must be at least 1")
    if args.train_hours is not None and args.train_hours <= 0:
        parser.error("--train-hours must be positive")
    if args.min_duration < 2 * MIN_CUT_SEC:
        parser.error(
            f"--min-duration below {2 * MIN_CUT_SEC:g} s leaves rows the loader cannot cut"
        )
    return args


# ---------------------------------------------------------------- metadata


def split_of(shard: str) -> str:
    return "test" if Path(shard).name.startswith("test-") else "train"


def list_shards(args: argparse.Namespace) -> list[str]:
    if args.source_dir:
        shards = sorted(str(path.resolve()) for path in args.source_dir.rglob("*.parquet"))
    else:
        from huggingface_hub import HfApi

        shards = sorted(
            name
            for name in HfApi().list_repo_files(REPO, repo_type="dataset")
            if name.startswith("data/") and name.endswith(".parquet")
        )
    if not shards:
        raise SystemExit("no parquet shards found for the dataset")
    if args.limit_shards:
        shards = [
            shard
            for split in ("train", "test")
            for shard in [s for s in shards if split_of(s) == split][: args.limit_shards]
        ]
    return shards


def open_shard(shard: str):
    if Path(shard).is_absolute():
        return open(shard, "rb")
    from huggingface_hub import HfFileSystem

    return HfFileSystem().open(f"datasets/{REPO}/{shard}", "rb")


def scan_shard(shard: str) -> list[dict]:
    """One record per row of a shard, from the non-audio columns only.

    Reading a remote shard fetches just those column chunks, not the ~420 MB of audio."""
    import pyarrow.parquet as pq

    records = []
    with open_shard(shard) as handle:
        table = pq.ParquetFile(handle).read(
            columns=["uuid", "sentence", "is_retranscribe", "extra_data"]
        )
    split = split_of(shard)
    for row, (uuid, text, retranscribed, extra) in enumerate(
        zip(
            table.column("uuid").to_pylist(),
            table.column("sentence").to_pylist(),
            table.column("is_retranscribe").to_pylist(),
            table.column("extra_data").to_pylist(),
        )
    ):
        records.append(
            {
                "shard": shard,
                "row": row,
                "split": split,
                "uuid": uuid,
                "text": text or "",
                "retranscribed": bool(retranscribed),
                "duration": float(extra["duration"]),
                "flagged": any(extra.get(flag) for flag in EXCLUDE_FLAGS),
            }
        )
    return records


def load_records(shards: list[str], meta_dir: Path, workers: int) -> list[dict]:
    """Every row's metadata, scanned once and cached under `meta_dir`."""
    fingerprint = hashlib.sha256(json.dumps(shards).encode()).hexdigest()[:16]
    cache = meta_dir / f"{fingerprint}-{INDEX_FILE}"
    if cache.is_file():
        with gzip.open(cache, "rt", encoding="utf-8") as handle:
            return json.load(handle)
    print(f"scanning the metadata of {len(shards)} shards ...", flush=True)
    records: list[dict] = []
    started = time.monotonic()
    for done, part in enumerate(parallel(scan_shard, shards, workers), 1):
        records += part
        print(
            f"  shard {done}/{len(shards)} scanned, elapsed "
            f"{format_seconds(time.monotonic() - started)}",
            flush=True,
        )
    records.sort(key=lambda r: (r["shard"], r["row"]))
    meta_dir.mkdir(parents=True, exist_ok=True)
    partial_path = cache.with_name(f".{cache.name}.partial")
    with gzip.open(partial_path, "wt", encoding="utf-8") as handle:
        json.dump(records, handle, ensure_ascii=False)
    partial_path.replace(cache)
    return records


# --------------------------------------------------------------- selection


def clean_text(text: str) -> str:
    """Folded characters (the transcripts carry U+202B bidi marks), NFC, single spaces."""
    return clean(fold_characters(text))


def agree_key(text: str) -> str:
    return "".join(char for char in clean_text(text) if char.isalnum())


def parse_uuid(uuid: str) -> tuple[str, str, str]:
    """(show, episode, index) of `<show>/<episode title>/<index>`; ValueError otherwise."""
    episode, _, index = uuid.rpartition("/")
    if not episode or not index.isdigit():
        raise ValueError(f"unexpected uuid {uuid!r}")
    return uuid.split("/", 1)[0], episode, index


def choose_clips(records: list[dict], settings: dict) -> tuple[list[dict], Counter]:
    """One record per distinct clip, or the reason the clip is dropped.

    A clip can appear several times (one row per transcriber). The kept record carries the
    cleaned `text`, and the `show`, `episode` and `index` parsed from its uuid."""
    groups: dict[str, list[dict]] = {}
    for record in records:
        groups.setdefault(record["uuid"], []).append(record)
    kept: list[dict] = []
    dropped: Counter = Counter()
    for uuid, group in groups.items():
        best = min(group, key=lambda r: (not r["retranscribed"], r["shard"], r["row"]))
        duration = best["duration"]
        if any(r["flagged"] for r in group):
            dropped["flagged"] += 1
        elif duration < settings["min_duration"]:
            dropped["too_short"] += 1
        elif duration > settings["max_duration"]:
            dropped["too_long"] += 1
        elif settings["duplicates"] == "agree" and len({agree_key(r["text"]) for r in group}) > 1:
            dropped["transcribers_disagree"] += 1
        else:
            text = clean_text(best["text"])
            if not _HEBREW.search(text):
                dropped["no_hebrew"] += 1
            elif len(text.split()) < 2:
                dropped["few_words"] += 1
            else:
                try:
                    show, episode, index = parse_uuid(uuid)
                except ValueError:
                    dropped["bad_uuid"] += 1
                    continue
                kept.append(
                    {**best, "text": text, "show": show, "episode": episode, "index": index}
                )
    return kept, dropped


def glue_bare_punctuation(tokens: list[str]) -> list[str]:
    """Attach a token with nothing to say (a lone dash or quote) to the word before it."""
    out: list[str] = []
    for token in tokens:
        if _HAS_SPEECH.search(token):
            out.append(token)
        elif out:
            out[-1] += token
    return out


def viter_spellable(word: str) -> bool:
    """Whether the Viter backend can place `word`: only Hebrew letters (and geresh) remain."""
    letters = [char for char in _VITER_PUNCT.sub("", word).strip() if char != "'"]
    return bool(letters) and all("א" <= char <= "ת" for char in letters)


def unspellable_reason(word: str) -> str:
    cleaned = _VITER_PUNCT.sub("", word)
    if any(char.isdigit() for char in cleaned):
        return "digits"
    if any(char.isascii() and char.isalpha() for char in cleaned):
        return "latin"
    return "other_script" if any(char.isalpha() for char in cleaned) else "no_letters"


def prepare_text(text: str, normalize) -> tuple[list[str], dict | None]:
    """The words of a row and its `alt_words`, or ValueError(reason) when Viter cannot align it."""
    tokens = glue_bare_punctuation(text.split())
    alt_words = None
    if normalize is not None:
        placeholder = [{"word": token, "start": 0.0, "end": 0.0} for token in tokens]
        words, alt_words = normalize_words_with_alternates(placeholder, normalize)
        tokens = [word["word"] for word in words]
        if alt_words:
            validate_alt_words(alt_words, len(tokens))
    if len(tokens) < 2:
        raise ValueError("few_words")
    for token in tokens:
        if not viter_spellable(token):
            raise ValueError(unspellable_reason(token))
    return tokens, alt_words


def estimated_words(tokens: list[str], duration: float) -> list[dict]:
    """Placeholder times: the clip split between the words in proportion to their length.

    Not an alignment. They exist so the row has the shape the aligner reads, which replaces
    them; a row it cannot align is dropped (--strict), so none of these reach training."""
    weights = [len(token) + 1 for token in tokens]
    scale = duration / sum(weights)
    words, cursor = [], 0.0
    for token, weight in zip(tokens, weights):
        words.append({"word": token, "start": cursor, "end": cursor + weight * scale})
        cursor += weight * scale
    return words


def episode_order(clips: list[dict]) -> list[str]:
    return sorted({clip["episode"] for clip in clips}, key=stable_bucket)


def take_train(clips: list[dict], hours: float | None) -> list[dict]:
    """Whole episodes, in a fixed order, until `hours` is reached (all of them when None)."""
    if hours is None:
        return clips
    seconds: Counter = Counter()
    for clip in clips:
        seconds[clip["episode"]] += clip["duration"]
    chosen, total = set(), 0.0
    for episode in episode_order(clips):
        if total >= hours * 3600:
            break
        chosen.add(episode)
        total += seconds[episode]
    return [clip for clip in clips if clip["episode"] in chosen]


def take_valid(clips: list[dict], hours: float) -> list[dict]:
    """Clips of the held-out shows, in a fixed pseudo-random order, until `hours` is reached."""
    chosen, total = [], 0.0
    for clip in sorted(clips, key=lambda c: stable_bucket(c["uuid"])):
        if total >= hours * 3600:
            break
        chosen.append(clip)
        total += clip["duration"]
    return chosen


# ------------------------------------------------------------------- audio


def clip_path(audio_dir: Path, clip: dict, audio_format: str) -> Path:
    group = hashlib.sha1(clip["episode"].encode("utf-8")).hexdigest()[:12]
    return audio_dir / group / f"{clip['index']}{audio_suffix(audio_format)}"


def decode_clip(blob: bytes):
    """Mono float32 at SAMPLE_RATE from the MP3 bytes stored in the dataset."""
    import soundfile
    import sphn

    data, rate = soundfile.read(io.BytesIO(blob), dtype="float32", always_2d=True)
    mono = data.mean(axis=1)
    if rate != SAMPLE_RATE:
        mono = sphn.resample(mono, src_sample_rate=rate, dst_sample_rate=SAMPLE_RATE)
    return mono


def fetch_shard(shard: str, shard_dir: Path) -> Path:
    if Path(shard).is_absolute():
        return Path(shard)
    from huggingface_hub import hf_hub_download

    return Path(hf_hub_download(REPO, shard, repo_type="dataset", local_dir=str(shard_dir)))


def process_shard(job: dict) -> dict:
    """Write the audio of the wanted rows of one shard; existing files are kept."""
    import pyarrow.parquet as pq
    import soundfile

    wanted: dict[int, str] = {int(row): target for row, target in job["wanted"].items()}
    durations: dict[int, float] = {}
    pending = {row: Path(target) for row, target in wanted.items() if not Path(target).is_file()}
    for row, target in wanted.items():
        if row not in pending:
            info = soundfile.info(target)
            durations[row] = info.frames / info.samplerate
    downloaded = None
    try:
        if pending:
            path = fetch_shard(job["shard"], Path(job["shard_dir"]))
            downloaded = None if Path(job["shard"]).is_absolute() else path
            parquet = pq.ParquetFile(path)
            first = 0
            for group in range(parquet.num_row_groups):
                size = parquet.metadata.row_group(group).num_rows
                hits = [row for row in range(first, first + size) if row in pending]
                if hits:
                    audio = parquet.read_row_group(group, columns=["audio"]).column("audio")
                    for row in hits:
                        samples = decode_clip(audio[row - first].as_py()["bytes"])
                        write_audio(samples, pending[row], job["audio_format"])
                        durations[row] = len(samples) / SAMPLE_RATE
                first += size
    except Exception as error:  # noqa: BLE001 -- one bad shard should not end the build
        return {
            "shard": job["shard"],
            "status": "failed",
            "error": f"{type(error).__name__}: {error}",
        }
    finally:
        if downloaded is not None and not job["keep_shard"]:
            downloaded.unlink(missing_ok=True)
    return {"shard": job["shard"], "status": "ok", "durations": durations}


def run_jobs(jobs: list[dict], workers: int) -> list[dict]:
    results: list[dict] = []
    started = last = time.monotonic()
    for done, result in enumerate(parallel(process_shard, jobs, workers), 1):
        results.append(result)
        if result["status"] == "failed":
            print(f"  shard {result['shard']} failed: {result['error']}", file=sys.stderr)
        now = time.monotonic()
        if done == len(jobs) or now - last >= PROGRESS_SECONDS:
            last = now
            elapsed = now - started
            print(
                f"audio: {done}/{len(jobs)} shards, elapsed {format_seconds(elapsed)}, "
                f"ETA {format_seconds(elapsed * (len(jobs) - done) / done)}",
                flush=True,
            )
    return results


# -------------------------------------------------------------------- main


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    prepare_output(
        args.out_dir,
        {
            "builder": "crowd-transcribe",
            "layout": LAYOUT_VERSION,
            "audio_format": args.audio_format,
            "sample_rate": SAMPLE_RATE,
        },
        assume_yes=args.yes,
    )
    audio_dir, shard_dir = args.out_dir / "audio", args.out_dir / "meta" / "shards"
    shards = list_shards(args)
    records = load_records(shards, args.out_dir / "meta", args.workers)
    print(
        f"{len(records)} rows, {len({r['uuid'] for r in records})} distinct clips "
        f"({sum(r['duration'] for r in {r['uuid']: r for r in records}.values()) / 3600:.0f} h)"
    )

    settings = {
        "min_duration": args.min_duration,
        "max_duration": args.max_duration,
        "duplicates": args.duplicates,
    }
    clips, dropped = choose_clips(records, settings)
    normalize = load_normalizer(args.normalizer_dir) if args.normalize_text else None
    usable: list[dict] = []
    for clip in clips:
        try:
            clip["tokens"], clip["alt_words"] = prepare_text(clip["text"], normalize)
        except ValueError as error:
            dropped[f"text:{error}"] += 1
        else:
            usable.append(clip)
    print(f"dropped before audio: {dict(sorted(dropped.items()))}")

    by_split = {
        split: [clip for clip in usable if clip["split"] == split] for split in ("train", "test")
    }
    train_clips = take_train(by_split["train"], args.train_hours)
    valid_clips = take_valid(by_split["test"], args.valid_hours)
    if not train_clips or not valid_clips:
        raise SystemExit("one side of the split is empty; check --limit-shards and the filters")
    train_shows = {clip["show"] for clip in train_clips}
    valid_shows = {clip["show"] for clip in valid_clips}
    if train_shows & valid_shows:
        raise SystemExit(f"shows in both splits: {sorted(train_shows & valid_shows)}")

    jobs: dict[str, dict] = {}
    for clip in [*train_clips, *valid_clips]:
        job = jobs.setdefault(
            clip["shard"],
            {
                "shard": clip["shard"],
                "shard_dir": str(shard_dir),
                "audio_format": args.audio_format,
                "keep_shard": args.keep_shards,
                "wanted": {},
            },
        )
        job["wanted"][clip["row"]] = str(clip_path(audio_dir, clip, args.audio_format))
    results = run_jobs(list(jobs.values()), args.workers)
    shutil.rmtree(shard_dir, ignore_errors=True)
    lengths = {
        (result["shard"], row): seconds
        for result in results
        if result["status"] == "ok"
        for row, seconds in result["durations"].items()
    }

    stats: Counter = Counter()

    def build(clips: list[dict]) -> list[dict]:
        rows = []
        for clip in clips:
            length = lengths.get((clip["shard"], clip["row"]))
            if length is None:
                stats["no_audio"] += 1
                continue
            if length < args.min_duration:
                stats["short_audio"] += 1
                continue
            extra = {"alt_words": clip["alt_words"]} if clip["alt_words"] else {}
            try:
                rows.append(
                    make_row(
                        clip_path(audio_dir, clip, args.audio_format).resolve(),
                        length,
                        estimated_words(clip["tokens"], length),
                        clip["episode"],
                        dataset=DATASET,
                        clip=clip["uuid"],
                        show=clip["show"],
                        timing="estimated",
                        **extra,
                    )
                )
            except ValueError:
                stats["bad_row"] += 1
        return rows

    train, valid = build(train_clips), build(valid_clips)
    if not train or not valid:
        raise SystemExit("no usable rows were produced")
    train_path, valid_path = write_manifests(args.out_dir, train, valid)
    print(f"dropped at build: {dict(stats)}")
    print(f"shows: {len(train_shows)} in train, {len(valid_shows)} in valid, none in both")
    describe_split("train", train)
    describe_split("valid", valid)
    print_duration_report(train)
    print(f"wrote {train_path} and {valid_path}")
    print(
        "TIMES ARE PLACEHOLDERS. Next: python hebrew_training/align_official_manifest.py "
        f"{args.out_dir} --strict ... and train on train_aligned.jsonl (see ALIGNMENT.md)."
    )


if __name__ == "__main__":
    main()
