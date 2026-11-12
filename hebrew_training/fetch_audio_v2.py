"""Survey, filter and download a subset of ivrit.ai audio-v2 (weakly labeled, >20k hours).

`ivrit-ai/audio-v2` holds 1.2 TB of podcast/lecture audio and `ivrit-ai/audio-v2-transcripts`
holds Whisper transcripts for all of it. Nothing in them is human-checked and speakers are
unknown, so what to keep has to be decided from Whisper's own confidence (`avg_logprob`,
`compression_ratio`, `no_speech_prob`, word probabilities) BEFORE any audio is fetched.

Stages (all resumable, all under --work):

    list      listing.json      the file listing of both repos, joined on (source, episode)
    survey    survey.jsonl      per-episode histograms of speech seconds over each metric,
                                from the transcripts only (~9 GB for all of them)
    report    survey_report.txt how many hours survive each threshold, per metric, jointly
                                for logprob x compression, and per source
    select    selection.json    episodes to download for a threshold set and an hour budget,
                                with the exact audio bytes that implies
    fetch     <out>/<id>/...    audio + transcript.aligned.json + metadata.json per episode

    python -m hebrew_training.fetch_audio_v2 survey --limit-episodes 100     # ~25 MB test
    python -m hebrew_training.fetch_audio_v2 survey
    python -m hebrew_training.fetch_audio_v2 report --min-avg-logprob -0.5 --max-compression 2.4
    python -m hebrew_training.fetch_audio_v2 select --min-avg-logprob -0.5 \\
        --max-compression 2.4 --hours 3000
    python -m hebrew_training.fetch_audio_v2 fetch --out data/audio_v2 --max-gb 50

Layout written by `fetch`, which is what build_official_manifest.py reads:

    <out>/av2_<hash>/audio.m4a|audio.mp3
                    /transcript.aligned.json     {"segments": [...]} with Whisper's fields
                    /metadata.json               source, episode, user_id, ...

then `build_official_manifest --corpus <out> --audio-glob audio.m4a,audio.mp3 \\
    --min-avg-logprob ... --max-compression-ratio ...` applies the same thresholds per
segment. Episode directories are ASCII ids: the episode names are Hebrew with dates, emoji
and apostrophes, and the manifests hold absolute paths that later tools must handle.

Both repos are gated (one checkbox, auto-approved). Accept at
https://huggingface.co/datasets/ivrit-ai/audio-v2 first, then `hf auth login`.

Histogram resolution: thresholds are snapped to the bin edges (0.05 logprob, 0.1 compression,
0.05 probability), so hours quoted here are at that resolution; the builder uses exact values.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import statistics
import sys
import time
from collections import defaultdict
from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import NamedTuple

AUDIO_REPO = "ivrit-ai/audio-v2"
TRANSCRIPT_REPO = "ivrit-ai/audio-v2-transcripts"
TRANSCRIPT_FILE = "full_transcript.json.gz"
AUDIO_EXTS = (".m4a", ".mp3")

# What the builder reads from a Whisper segment; the rest (tokens, seek, ori_dict) is bulk.
SEGMENT_KEYS = ("start", "end", "text", "avg_logprob", "compression_ratio", "no_speech_prob")
WORD_KEYS = ("word", "start", "end", "probability")


class Axis(NamedTuple):
    """Uniform bins: 0 is `< lo`, 1..n cover [lo + (i-1) step, lo + i step), n+1 is the rest."""

    lo: float
    step: float
    n: int


LOGPROB = Axis(-1.5, 0.05, 30)  # to 0.0
COMPRESSION = Axis(0.0, 0.1, 40)  # to 4.0
NO_SPEECH = Axis(0.0, 0.05, 20)
WORD_PROB = Axis(0.0, 0.05, 20)  # median word probability of the segment, like --min-quality
RATE = Axis(0.0, 0.5, 12)  # words per second of the segment

LOGPROB_THRESHOLDS = (-1.0, -0.8, -0.6, -0.5, -0.4, -0.3, -0.25, -0.2, -0.15)
COMPRESSION_THRESHOLDS = (1.6, 1.8, 2.0, 2.2, 2.4, 2.6, 3.0)
NO_SPEECH_THRESHOLDS = (0.05, 0.1, 0.2, 0.4, 0.6, 0.8)
WORD_PROB_THRESHOLDS = (0.5, 0.6, 0.7, 0.8, 0.9)


def bin_of(axis: Axis, value: float) -> int:
    if value < axis.lo:
        return 0
    k = int((value - axis.lo) / axis.step + 1e-9)
    return axis.n + 1 if k >= axis.n else k + 1


def edge(axis: Axis, threshold: float) -> int:
    """Index k of the bin edge nearest `threshold`: the edge sits at lo + k * step."""
    return round((threshold - axis.lo) / axis.step)


def bin_mid(axis: Axis, index: int) -> float:
    if index == 0:
        return axis.lo
    if index == axis.n + 1:
        return axis.lo + axis.n * axis.step
    return axis.lo + (index - 0.5) * axis.step


# ---------------------------------------------------------------------------------------
# Identity and listing


def recording_id(source: str, episode: str) -> str:
    return "av2_" + hashlib.sha1(f"{source}/{episode}".encode()).hexdigest()[:12]


def stable_bucket(key: str) -> float:
    """Deterministic and nested: the episodes chosen for 500 h are a subset of 1500 h."""
    return int(hashlib.sha1(key.encode()).hexdigest()[:8], 16) / 2**32


def load_listing(work: Path, refresh: bool = False) -> list[dict]:
    """Episodes that have both a transcript and audio, in stable random order."""
    path = work / "listing.json"
    if path.exists() and not refresh:
        return json.loads(path.read_text(encoding="utf-8"))

    from huggingface_hub import HfApi

    api = HfApi()
    transcripts: dict[tuple[str, str], int] = {}
    for sibling in api.dataset_info(TRANSCRIPT_REPO, files_metadata=True).siblings:
        parts = sibling.rfilename.split("/")
        if len(parts) == 3 and parts[2] == TRANSCRIPT_FILE:
            transcripts[(parts[0], parts[1])] = sibling.size or 0

    audio: dict[tuple[str, str], tuple[str, int]] = {}
    for sibling in api.dataset_info(AUDIO_REPO, files_metadata=True).siblings:
        parts = sibling.rfilename.split("/")
        if len(parts) != 2:
            continue
        stem, dot, extension = parts[1].rpartition(".")
        if dot and f".{extension}" in AUDIO_EXTS:
            audio[(parts[0], stem)] = (f".{extension}", sibling.size or 0)

    episodes = []
    for (source, episode), transcript_bytes in transcripts.items():
        if (source, episode) not in audio:
            continue
        extension, audio_bytes = audio[(source, episode)]
        episodes.append(
            {
                "id": recording_id(source, episode),
                "source": source,
                "episode": episode,
                "ext": extension,
                "audio_bytes": audio_bytes,
                "transcript_bytes": transcript_bytes,
            }
        )
    episodes.sort(key=lambda e: stable_bucket(e["id"]))
    work.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(episodes, ensure_ascii=False), encoding="utf-8")
    print(
        f"listing: {len(transcripts)} transcripts, {len(audio)} audio, "
        f"{len(episodes)} joined, {len({e['source'] for e in episodes})} sources"
    )
    return episodes


# ---------------------------------------------------------------------------------------
# Per-episode statistics (pure; tested without network)


def episode_stats(transcript: dict) -> dict:
    """Speech seconds binned over each Whisper confidence metric, sparse and JSON-ready.

    `joint` is logprob x compression ("i,j" -> seconds) because those are the two thresholds
    that get tuned together; every other metric gets its own marginal. Segments without words
    are counted apart (`no_words_s`), as the builder drops them first.
    """
    joint: dict[str, float] = defaultdict(float)
    no_speech: dict[int, float] = defaultdict(float)
    word_prob: dict[int, float] = defaultdict(float)
    rate: dict[int, float] = defaultdict(float)
    speech = no_words = wall = 0.0
    segments = 0
    for segment in transcript.get("segments") or []:
        start, end = segment.get("start"), segment.get("end")
        if start is None or end is None:
            continue
        wall = max(wall, end)
        duration = max(0.0, end - start)
        words = segment.get("words") or []
        if not words or duration <= 0:
            no_words += duration
            continue
        probabilities = [w["probability"] for w in words if w.get("probability") is not None]
        logprob = segment.get("avg_logprob")
        compression = segment.get("compression_ratio")
        silent = segment.get("no_speech_prob")
        i = bin_of(LOGPROB, logprob if logprob is not None else 0.0)
        j = bin_of(COMPRESSION, compression if compression is not None else 0.0)
        joint[f"{i},{j}"] += duration
        no_speech[bin_of(NO_SPEECH, silent if silent is not None else 0.0)] += duration
        word_prob[
            bin_of(WORD_PROB, statistics.median(probabilities) if probabilities else 0.0)
        ] += duration
        rate[bin_of(RATE, len(words) / duration)] += duration
        speech += duration
        segments += 1

    def rounded(histogram: dict) -> dict[str, float]:
        return {str(k): round(v, 3) for k, v in histogram.items()}

    return {
        "segments": segments,
        "speech_s": round(speech, 2),
        "no_words_s": round(no_words, 2),
        "wall_s": round(wall, 2),
        "joint": rounded(joint),
        "nsp": rounded(no_speech),
        "wp": rounded(word_prob),
        "rate": rounded(rate),
    }


def kept_seconds(record: dict, min_logprob: float | None, max_compression: float | None) -> float:
    """Speech seconds with avg_logprob >= min_logprob and compression_ratio <= max_compression."""
    first_lp = 0 if min_logprob is None else max(0, edge(LOGPROB, min_logprob) + 1)
    last_cr = COMPRESSION.n + 1 if max_compression is None else edge(COMPRESSION, max_compression)
    total = 0.0
    for key, seconds in record["joint"].items():
        i, j = (int(x) for x in key.split(","))
        if i >= first_lp and j <= last_cr:
            total += seconds
    return total


def marginal(records: Iterable[dict], axis_key: str) -> dict[int, float]:
    """Seconds per bin summed over records; logprob/compression come out of `joint`."""
    out: dict[int, float] = defaultdict(float)
    for record in records:
        if axis_key in ("lp", "cr"):
            position = 0 if axis_key == "lp" else 1
            for key, seconds in record["joint"].items():
                out[int(key.split(",")[position])] += seconds
        else:
            for key, seconds in record[axis_key].items():
                out[int(key)] += seconds
    return out


def quantile(histogram: dict[int, float], axis: Axis, q: float) -> float:
    total = sum(histogram.values())
    if total <= 0:
        return float("nan")
    running = 0.0
    for index in sorted(histogram):
        running += histogram[index]
        if running >= q * total:
            return bin_mid(axis, index)
    return bin_mid(axis, max(histogram))


# ---------------------------------------------------------------------------------------
# Transport


def _fetch_bytes(repo: str, path: str, retries: int = 4) -> bytes:
    from huggingface_hub import hf_hub_url
    from huggingface_hub.utils import build_hf_headers, get_session

    url = hf_hub_url(repo, path, repo_type="dataset")
    for attempt in range(retries):
        try:
            response = get_session().get(
                url, headers=build_hf_headers(), follow_redirects=True, timeout=120
            )
            response.raise_for_status()
            return response.content
        except Exception:
            if attempt == retries - 1:
                raise
            time.sleep(2**attempt)
    raise AssertionError("unreachable")


def fetch_transcript(episode: dict) -> dict:
    path = f"{episode['source']}/{episode['episode']}/{TRANSCRIPT_FILE}"
    return json.loads(gzip.decompress(_fetch_bytes(TRANSCRIPT_REPO, path)))


def _download_audio(episode: dict, target: Path) -> None:
    from huggingface_hub import hf_hub_url
    from huggingface_hub.utils import build_hf_headers, get_session

    path = f"{episode['source']}/{episode['episode']}{episode['ext']}"
    url = hf_hub_url(AUDIO_REPO, path, repo_type="dataset")
    part = target.with_name(target.name + ".part")
    with get_session().stream(
        "GET", url, headers=build_hf_headers(), follow_redirects=True, timeout=300
    ) as response:
        response.raise_for_status()
        with open(part, "wb") as handle:
            handle.writelines(response.iter_bytes(1 << 20))
    if episode["audio_bytes"] and part.stat().st_size != episode["audio_bytes"]:
        raise OSError(
            f"{path}: got {part.stat().st_size} bytes, listing says {episode['audio_bytes']}"
        )
    part.replace(target)


# ---------------------------------------------------------------------------------------
# Commands


def read_survey(work: Path) -> dict[str, dict]:
    path = work / "survey.jsonl"
    records: dict[str, dict] = {}
    if path.exists():
        with open(path, encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    record = json.loads(line)
                    records[record["id"]] = record
    return records


def pick_sources(episodes: list[dict], include: list[str] | None, exclude: list[str] | None):
    if include:
        episodes = [e for e in episodes if e["source"] in include]
    if exclude:
        episodes = [e for e in episodes if e["source"] not in exclude]
    return episodes


def command_list(args: argparse.Namespace) -> None:
    episodes = load_listing(args.work, refresh=True)
    gib = sum(e["audio_bytes"] for e in episodes) / 2**30
    transcripts = sum(e["transcript_bytes"] for e in episodes) / 2**30
    print(f"{len(episodes)} episodes, {gib:.0f} GiB audio, {transcripts:.1f} GiB transcripts")


def command_survey(args: argparse.Namespace) -> None:
    episodes = pick_sources(load_listing(args.work), args.sources, args.exclude_sources)
    if args.limit_episodes:
        episodes = episodes[: args.limit_episodes]
    done = read_survey(args.work)
    todo = [e for e in episodes if e["id"] not in done]
    if args.max_mb:
        budget, kept = args.max_mb * 1e6, []
        for episode in todo:
            if budget < episode["transcript_bytes"]:
                break
            budget -= episode["transcript_bytes"]
            kept.append(episode)
        todo = kept
    size = sum(e["transcript_bytes"] for e in todo) / 1e6
    print(f"survey: {len(todo)} to fetch ({size:.0f} MB), {len(episodes) - len(todo)} already done")
    if args.dry_run or not todo:
        return

    args.work.mkdir(parents=True, exist_ok=True)
    failed = 0
    started = time.monotonic()

    def one(episode: dict) -> dict:
        stats = episode_stats(fetch_transcript(episode))
        return {
            "id": episode["id"],
            "source": episode["source"],
            "episode": episode["episode"],
            **stats,
        }

    with (
        open(args.work / "survey.jsonl", "a", encoding="utf-8") as out,
        ThreadPoolExecutor(max_workers=args.workers) as pool,
    ):
        futures = {pool.submit(one, e): e for e in todo}
        for count, future in enumerate(as_completed(futures), start=1):
            try:
                out.write(json.dumps(future.result(), ensure_ascii=False) + "\n")
            except Exception as exc:  # noqa: BLE001 -- one bad transcript must not stop 35k
                failed += 1
                print(f"  failed {futures[future]['id']}: {exc}", file=sys.stderr)
            if count % 100 == 0:
                out.flush()
                rate = count / (time.monotonic() - started)
                print(f"  {count}/{len(todo)}  ({failed} failed, {rate:.1f}/s)")
    print(f"surveyed {len(todo) - failed}, failed {failed}; next: report")


def _hours(seconds: float) -> float:
    return seconds / 3600


def build_report(records: list[dict], total_episodes: int, args: argparse.Namespace) -> str:
    lines: list[str] = []
    out = lines.append
    speech = sum(r["speech_s"] for r in records)
    wall = sum(r["wall_s"] for r in records)
    sources = {r["source"] for r in records}
    out(
        f"episodes surveyed : {len(records)} of {total_episodes} listed "
        f"({100 * len(records) / max(1, total_episodes):.1f}%)"
    )
    out(f"sources           : {len(sources)}")
    out(f"audio (wall)      : {_hours(wall):,.0f} h")
    out(
        f"speech in segments: {_hours(speech):,.0f} h  ({100 * speech / max(1, wall):.0f}% of wall)"
    )
    out(f"segments w/o words: {_hours(sum(r['no_words_s'] for r in records)):,.0f} h (dropped)")
    if len(records) < total_episodes:
        out("NOTE: partial survey (a stable random sample); hours here are not corpus totals.")

    def cumulative(title, axis_key, axis, thresholds, direction):
        histogram = marginal(records, axis_key)
        total = sum(histogram.values())
        out("")
        out(
            f"{title}   p5/p25/p50/p75/p95 = "
            + " / ".join(
                f"{quantile(histogram, axis, q):.2f}" for q in (0.05, 0.25, 0.5, 0.75, 0.95)
            )
        )
        out(f"  {'threshold':>10} {'hours':>10} {'% of speech':>12}")
        for t in thresholds:
            k = edge(axis, t)
            if direction == ">=":
                passing = sum(v for b, v in histogram.items() if b >= k + 1)
            else:
                passing = sum(v for b, v in histogram.items() if b <= k)
            out(
                f"  {direction}{t:>9} {_hours(passing):>10,.0f} {100 * passing / max(1e-9, total):>11.1f}%"
            )

    cumulative("avg_logprob (keep >=)", "lp", LOGPROB, LOGPROB_THRESHOLDS, ">=")
    cumulative("compression_ratio (keep <=)", "cr", COMPRESSION, COMPRESSION_THRESHOLDS, "<=")
    cumulative("no_speech_prob (keep <=)", "nsp", NO_SPEECH, NO_SPEECH_THRESHOLDS, "<=")
    cumulative("median word probability (keep >=)", "wp", WORD_PROB, WORD_PROB_THRESHOLDS, ">=")

    rate = marginal(records, "rate")
    out("")
    out(
        "words/second   p5/p25/p50/p75/p95 = "
        + " / ".join(f"{quantile(rate, RATE, q):.1f}" for q in (0.05, 0.25, 0.5, 0.75, 0.95))
    )

    out("")
    out("hours kept, avg_logprob >= row AND compression_ratio <= column")
    out("  " + " " * 8 + "".join(f"{t:>9}" for t in COMPRESSION_THRESHOLDS))
    for lp in LOGPROB_THRESHOLDS:
        cells = "".join(
            f"{_hours(sum(kept_seconds(r, lp, cr) for r in records)):>9,.0f}"
            for cr in COMPRESSION_THRESHOLDS
        )
        out(f"  {lp:>8}{cells}")

    lp, cr = args.min_avg_logprob, args.max_compression
    out("")
    out(f"candidate filter: avg_logprob >= {lp}, compression_ratio <= {cr}")
    kept_total = sum(kept_seconds(r, lp, cr) for r in records)
    out(
        f"  kept {_hours(kept_total):,.0f} h of {_hours(speech):,.0f} h "
        f"({100 * kept_total / max(1e-9, speech):.1f}%)"
    )
    out("  episode-level: share of an episode's speech that survives")
    out(f"  {'kept share':>12} {'episodes':>9} {'kept hours':>11}")
    shares = [
        (kept_seconds(r, lp, cr) / r["speech_s"], kept_seconds(r, lp, cr))
        for r in records
        if r["speech_s"] > 0
    ]
    floors = (0.95, 0.9, 0.75, 0.5, 0.25, 0.0)
    for floor, ceiling in zip(floors, (1.01,) + floors):
        group = [kept for share, kept in shares if floor <= share < ceiling]
        label = ">= 0.95" if floor == 0.95 else f"{floor:.2f}-{ceiling:.2f}"
        out(f"  {label:>12} {len(group):>9} {_hours(sum(group)):>11,.0f}")

    by_source: dict[str, list[dict]] = defaultdict(list)
    for record in records:
        by_source[record["source"]].append(record)
    rows = []
    for source, group in by_source.items():
        s = sum(r["speech_s"] for r in group)
        k = sum(kept_seconds(r, lp, cr) for r in group)
        rows.append(
            (
                source,
                len(group),
                _hours(s),
                _hours(k),
                quantile(marginal(group, "lp"), LOGPROB, 0.5),
                quantile(marginal(group, "cr"), COMPRESSION, 0.5),
            )
        )
    rows.sort(key=lambda row: -row[2])
    out("")
    out(f"per source (top {args.top_sources} by speech hours; full list in survey_by_source.csv)")
    out(
        f"  {'source':<34} {'eps':>5} {'speech h':>9} {'kept h':>9} {'kept%':>6} {'p50 lp':>7} {'p50 cr':>7}"
    )
    for source, count, s, k, med_lp, med_cr in rows[: args.top_sources]:
        out(
            f"  {source[:34]:<34} {count:>5} {s:>9,.0f} {k:>9,.0f} {100 * k / max(1e-9, s):>5.0f}% "
            f"{med_lp:>7.2f} {med_cr:>7.2f}"
        )
    csv = ["source,episodes,speech_h,kept_h,p50_logprob,p50_compression"]
    csv += [f'"{s}",{c},{sh:.2f},{kh:.2f},{lp_:.3f},{cr_:.3f}' for s, c, sh, kh, lp_, cr_ in rows]
    (args.work / "survey_by_source.csv").write_text("\n".join(csv) + "\n", encoding="utf-8")
    return "\n".join(lines)


def command_report(args: argparse.Namespace) -> None:
    records = list(read_survey(args.work).values())
    if not records:
        raise SystemExit("no survey.jsonl -- run `survey` first")
    total = len(load_listing(args.work))
    text = build_report(records, total, args)
    (args.work / "survey_report.txt").write_text(text + "\n", encoding="utf-8")
    print(text)


def command_select(args: argparse.Namespace) -> None:
    listing = {e["id"]: e for e in load_listing(args.work)}
    survey = read_survey(args.work)
    candidates = []
    for episode_id, record in survey.items():
        episode = listing.get(episode_id)
        if episode is None:
            continue
        if args.sources and episode["source"] not in args.sources:
            continue
        if args.exclude_sources and episode["source"] in args.exclude_sources:
            continue
        kept = kept_seconds(record, args.min_avg_logprob, args.max_compression)
        if record["speech_s"] < args.min_speech_minutes * 60:
            continue
        if kept / max(1e-9, record["speech_s"]) < args.min_kept_fraction:
            continue
        candidates.append((stable_bucket(episode_id), episode, record, kept))
    candidates.sort(key=lambda item: item[0])

    chosen, kept_total, speech_total, audio_bytes = [], 0.0, 0.0, 0
    for _, episode, record, kept in candidates:
        if args.hours and _hours(kept_total) >= args.hours:
            break
        chosen.append({**episode, "speech_s": record["speech_s"], "kept_s": round(kept, 2)})
        kept_total += kept
        speech_total += record["speech_s"]
        audio_bytes += episode["audio_bytes"]

    print(f"surveyed {len(survey)}; {len(candidates)} episodes pass the episode-level gates")
    print(f"selected {len(chosen)} episodes from {len({c['source'] for c in chosen})} sources")
    print(
        f"  {_hours(kept_total):,.0f} h kept speech inside {_hours(speech_total):,.0f} h of speech"
    )
    print(f"  {audio_bytes / 2**30:,.1f} GiB of audio to download")
    if not chosen:
        raise SystemExit("nothing selected -- loosen the thresholds")
    params = {k: v for k, v in vars(args).items() if k not in ("func", "command", "work")}
    selection = {"params": params, "episodes": chosen}
    (args.work / "selection.json").write_text(
        json.dumps(selection, ensure_ascii=False), encoding="utf-8"
    )
    print(f"wrote {args.work / 'selection.json'}; next: fetch --out DIR")


def _slim_segment(segment: dict) -> dict:
    slim = {k: segment[k] for k in SEGMENT_KEYS if k in segment}
    slim["words"] = [{k: w[k] for k in WORD_KEYS if k in w} for w in segment.get("words") or []]
    return slim


def fetch_one(episode: dict, out: Path, speaker_key: str) -> str:
    target = out / episode["id"]
    metadata = target / "metadata.json"
    if metadata.exists():  # written last, so its presence means the other two are complete
        return "skipped"
    target.mkdir(parents=True, exist_ok=True)

    transcript = fetch_transcript(episode)
    segments = [_slim_segment(s) for s in transcript.get("segments") or []]
    (target / "transcript.aligned.json").write_text(
        json.dumps({"segments": segments}, ensure_ascii=False), encoding="utf-8"
    )
    audio = target / f"audio{episode['ext']}"
    if not audio.exists():
        _download_audio(episode, audio)

    # Speakers are unknown. `source` keeps one show's hosts out of both splits; `recording`
    # is the honest-but-leaky alternative. A later speaker-classification pass can overwrite
    # user_id without touching anything else.
    user = f"av2-source-{hashlib.sha1(episode['source'].encode()).hexdigest()[:10]}"
    if speaker_key == "recording":
        user = episode["id"]
    metadata.write_text(
        json.dumps(
            {
                "recording_id": episode["id"],
                "source": episode["source"],
                "episode": episode["episode"],
                "user_id": user,
                "speaker_known": False,
                "speech_s": episode.get("speech_s"),
                "kept_s": episode.get("kept_s"),
                "dataset": "ivrit-ai/audio-v2",
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    return "done"


def command_fetch(args: argparse.Namespace) -> None:
    path = args.work / "selection.json"
    if not path.exists():
        raise SystemExit("no selection.json -- run `select` first")
    episodes = json.loads(path.read_text(encoding="utf-8"))["episodes"]
    if args.limit:
        episodes = episodes[: args.limit]
    gib = sum(e["audio_bytes"] for e in episodes) / 2**30
    print(f"fetch: {len(episodes)} episodes, {gib:.2f} GiB of audio")
    if gib > args.max_gb:
        raise SystemExit(f"{gib:.1f} GiB exceeds --max-gb {args.max_gb}; raise it or use --limit")
    if args.dry_run:
        return

    args.out.mkdir(parents=True, exist_ok=True)
    counts = {"done": 0, "skipped": 0, "failed": 0}
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(fetch_one, e, args.out, args.speaker_key): e for e in episodes}
        for count, future in enumerate(as_completed(futures), start=1):
            try:
                counts[future.result()] += 1
            except Exception as exc:  # noqa: BLE001 -- one dead episode must not stop the run
                counts["failed"] += 1
                print(f"  failed {futures[future]['id']}: {exc}", file=sys.stderr)
            if count % 25 == 0:
                print(f"  {count}/{len(episodes)}  {counts}")
    print(f"{counts}")
    print(
        f"next: python -m hebrew_training.build_official_manifest --corpus {args.out} "
        "--audio-glob audio.m4a,audio.mp3 --min-avg-logprob ... --max-compression-ratio ..."
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--work", type=Path, default=Path("data/audio_v2_work"))
    sub = parser.add_subparsers(dest="command", required=True)

    def with_sources(p: argparse.ArgumentParser) -> None:
        p.add_argument("--sources", nargs="*", help="Only these source folders (exact names).")
        p.add_argument("--exclude-sources", nargs="*")

    def with_filter(p: argparse.ArgumentParser, lp: float | None, cr: float | None) -> None:
        p.add_argument(
            "--min-avg-logprob",
            type=float,
            default=lp,
            help="Keep speech with Whisper avg_logprob >= this.",
        )
        p.add_argument(
            "--max-compression",
            type=float,
            default=cr,
            help="Keep speech with compression_ratio <= this.",
        )

    p = sub.add_parser("list", help="Refresh listing.json from the Hub.")
    p.set_defaults(func=command_list)

    p = sub.add_parser("survey", help="Per-episode metric histograms from the transcripts.")
    with_sources(p)
    p.add_argument(
        "--limit-episodes", type=int, help="First N episodes of the stable random order."
    )
    p.add_argument("--max-mb", type=float, help="Stop planning once transcripts reach this size.")
    p.add_argument("--workers", type=int, default=16)
    p.add_argument("--dry-run", action="store_true")
    p.set_defaults(func=command_survey)

    p = sub.add_parser("report", help="Hours surviving each threshold, from survey.jsonl.")
    with_filter(p, -0.5, 2.4)
    p.add_argument("--top-sources", type=int, default=40)
    p.set_defaults(func=command_report)

    p = sub.add_parser("select", help="Choose episodes for a threshold set and hour budget.")
    with_sources(p)
    with_filter(p, None, None)
    p.add_argument("--hours", type=float, default=0.0, help="Budget in KEPT speech hours; 0 = all.")
    p.add_argument(
        "--min-kept-fraction",
        type=float,
        default=0.5,
        help="Skip episodes where less than this share of speech survives the filter.",
    )
    p.add_argument("--min-speech-minutes", type=float, default=2.0)
    p.set_defaults(func=command_select)

    p = sub.add_parser("fetch", help="Download the selected audio and write episode folders.")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--limit", type=int, help="Only the first N selected episodes (testing).")
    p.add_argument("--max-gb", type=float, default=25.0, help="Refuse a larger download.")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument(
        "--speaker-key",
        choices=("source", "recording"),
        default="source",
        help="What metadata.json `user_id` holds; the manifest split is disjoint on it.",
    )
    p.add_argument("--dry-run", action="store_true")
    p.set_defaults(func=command_fetch)
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
