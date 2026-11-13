"""Rate ivrit.ai audio-v2 rows in the browser and tune the data cut from the ratings.

    python -m hebrew_training.audio_v2_tagger.server --work W --port 8765
    ssh -L 8765:localhost:8765 technion-compute-113      # then open http://localhost:8765

Needs the per-row table from `hebrew_training.audio_v2_rows` and `pip install xgboost pandas`
(extra `tagger`). Rows are drawn with probability proportional to their duration, so ratings are
a uniform sample of speech hours. A clip is prepared by seeking INSIDE the remote file (a signed
URL and an ffmpeg range request) and encoding ~12 s of Opus; only mp3 episodes, whose seek is
inexact, are downloaded whole, cut, and deleted. `--ahead` clips are always ready, and clips
you have rated are deleted from the cache.

Ratings append to `W/tagger/labels.jsonl` (resume and undo safe). Features are not stored with
a rating, only the row's identity; they are looked up in the table at fit time.
"""

from __future__ import annotations

import argparse
import collections
import contextlib
import json
import math
import shutil
import subprocess
import threading
import time
from pathlib import Path

import numpy as np
from pydantic import BaseModel

from hebrew_training import audio_v2_table as table_mod
from hebrew_training import fetch_audio_v2 as av2
from hebrew_training.audio_v2_tagger import model as tagmodel

PAD = 0.25  # seconds of audio around the row so the first/last word is not clipped
POPULATION_SAMPLE = 150_000
KEEP_SERVED = 40
INDEX = Path(__file__).with_name("index.html")


def disk_fraction(path: Path) -> float:
    usage = shutil.disk_usage(path)
    return usage.used / usage.total


def _run(command: list[str], timeout: int = 240) -> subprocess.CompletedProcess:
    return subprocess.run(command, capture_output=True, text=True, timeout=timeout, check=False)


def clip_duration(path: Path) -> float:
    done = _run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(path)]
    )
    try:
        return float(done.stdout.strip())
    except ValueError:
        return 0.0


def signed_url(repo: str, path: str) -> str | None:
    """The Hub answers a `resolve` URL with a redirect to a signed CDN URL; ffmpeg can range-read it."""
    from huggingface_hub import hf_hub_url
    from huggingface_hub.utils import build_hf_headers, get_session

    url = hf_hub_url(repo, path, repo_type="dataset")
    with get_session().stream(
        "GET", url, headers=build_hf_headers(), follow_redirects=False, timeout=60
    ) as response:
        if response.status_code in (301, 302, 303, 307, 308):
            return response.headers.get("location")
    return None


ENCODE = ["-vn", "-ac", "1", "-ar", "24000", "-c:a", "libopus", "-b:a", "24k", "-f", "ogg"]


def make_clip(episode: dict, start: float, duration: float, out: Path, scratch: Path) -> None:
    """Encode [start, start + duration) of the episode to `out` (Opus), exactly."""
    base = ["ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error"]
    expected = duration
    if episode["ext"] == ".m4a":
        url = signed_url(av2.AUDIO_REPO, f"{episode['source']}/{episode['episode']}.m4a")
        if url:
            done = _run(
                [
                    *base,
                    "-ss",
                    f"{start:.3f}",
                    "-i",
                    url,
                    "-t",
                    f"{duration:.3f}",
                    *ENCODE,
                    "-y",
                    str(out),
                ]
            )
            if done.returncode == 0 and abs(clip_duration(out) - expected) < 0.5:
                return
    # Whole-file route: mp3 (inexact seek) and any failed range read.
    source = scratch / f"{episode['id']}{episode['ext']}"
    try:
        av2._download_audio(episode, source)
        done = _run(
            [
                *base,
                "-i",
                str(source),
                "-ss",
                f"{start:.3f}",
                "-t",
                f"{duration:.3f}",
                *ENCODE,
                "-y",
                str(out),
            ]
        )
    finally:
        source.unlink(missing_ok=True)
        source.with_name(source.name + ".part").unlink(missing_ok=True)
    if done.returncode != 0 or abs(clip_duration(out) - expected) > 0.5:
        out.unlink(missing_ok=True)
        raise RuntimeError(f"clip failed for {episode['id']}: {done.stderr[-200:]}")


class Tagger:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.table = table_mod.load_table(args.work)
        listing = {e["id"]: e for e in av2.load_listing(args.work)}
        self.episodes = [listing[i] for i in self.table.episode_ids]
        self.names = list(table_mod.FEATURES)
        self.total_hours = float(self.table.duration.sum() / 3600)
        self.cum = np.cumsum(self.table.duration)
        self.rng = np.random.default_rng(args.seed)

        self.clips = args.cache / "clips"
        self.scratch = args.cache / "tmp"
        for folder in (self.clips, self.scratch):
            shutil.rmtree(folder, ignore_errors=True)
            folder.mkdir(parents=True, exist_ok=True)
        self.labels_path = args.work / "tagger" / "labels.jsonl"
        self.labels_path.parent.mkdir(parents=True, exist_ok=True)

        self.lock = threading.RLock()
        self.ready: collections.deque[dict] = collections.deque()
        self.inflight = 0
        self.failed = 0
        self.used: set[int] = set()
        self.served: collections.OrderedDict[str, dict] = collections.OrderedDict()
        self.ratings: dict[str, dict] = {}  # key -> {row, rating}; rating 0 = skipped
        self.history: list[str] = []
        self.transcripts: collections.OrderedDict[str, list] = collections.OrderedDict()
        self.settings = {"use_source": False, "cutoff": 0.5}
        self.fit_result: dict | None = None
        self.fit_n = -1
        self.fitting = False
        self.refit_pending = False
        self.stop = threading.Event()
        self._load_labels()
        self._population_cache: dict[bool, object] = {}

    # -- labels ---------------------------------------------------------------------------
    def key_of(self, row: int) -> str:
        return f"{self.table.episode_ids[self.table.episode[row]]}_{round(self.table.start[row] * 1000)}"

    def _load_labels(self) -> None:
        if not self.labels_path.exists():
            return
        for line in self.labels_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            record = json.loads(line)
            if "undo" in record:
                self.ratings.pop(record["undo"], None)
                if record["undo"] in self.history:
                    self.history.remove(record["undo"])
                continue
            row = record["row"]
            if row >= len(self.table) or self.key_of(row) != record["key"]:
                print(f"label {record['key']} does not match the table; ignored")
                continue
            self.ratings[record["key"]] = {"row": row, "rating": record["rating"]}
            self.history.append(record["key"])
            self.used.add(row)
        print(f"loaded {len(self.ratings)} ratings")

    def _append(self, record: dict) -> None:
        with open(self.labels_path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    # -- sampling and preparation ---------------------------------------------------------
    def draw(self) -> int:
        with self.lock:
            while True:
                row = int(np.searchsorted(self.cum, self.rng.random() * self.cum[-1], side="right"))
                row = min(row, len(self.cum) - 1)
                if row not in self.used:
                    self.used.add(row)
                    return row

    def _words(self, episode: dict, start: float, end: float) -> list[dict]:
        with self.lock:
            segments = self.transcripts.get(episode["id"])
        if segments is None:
            segments = av2.fetch_transcript(episode).get("segments") or []
            with self.lock:
                self.transcripts[episode["id"]] = segments
                while len(self.transcripts) > 32:
                    self.transcripts.popitem(last=False)
        clip_start = max(0.0, start - PAD)
        words = []
        for segment in table_mod.row_segments(segments, start, end):
            for word in segment["words"]:
                text = word["word"].strip()
                if text:
                    words.append(
                        {
                            "w": text,
                            "start": round(word["start"] - clip_start, 3),
                            "end": round(word["end"] - clip_start, 3),
                        }
                    )
        return words

    def prepare(self, row: int) -> dict:
        table = self.table
        episode = self.episodes[table.episode[row]]
        start, end = float(table.start[row]), float(table.end[row])
        key = self.key_of(row)
        clip_start = max(0.0, start - PAD)
        out = self.clips / f"{key}.ogg"
        words = self._words(episode, start, end)
        make_clip(episode, clip_start, end + PAD - clip_start, out, self.scratch)
        return {
            "id": key,
            "row": row,
            "audio_url": f"/clips/{key}.ogg",
            "words": words,
            "source": episode["source"],
            "episode": episode["episode"],
            "start": start,
            "duration": end - start,
            "metrics": {n: _num(table.x[row, i]) for i, n in enumerate(self.names)},
        }

    def worker(self) -> None:
        while not self.stop.is_set():
            # The disk check comes first: a slot claimed and then abandoned would never be freed.
            paused = disk_fraction(self.args.cache) > self.args.max_disk_fraction
            with self.lock:
                full = len(self.ready) + self.inflight >= self.args.ahead
                if not full and not paused:
                    self.inflight += 1
            if full or paused:
                time.sleep(0.4)
                continue
            row = self.draw()
            try:
                item = self.prepare(row)
                with self.lock:
                    self.ready.append(item)
            except Exception as exc:  # noqa: BLE001 -- a dead episode must not stop the queue
                with self.lock:
                    self.failed += 1
                print(f"prepare failed for row {row}: {exc}")
            finally:
                with self.lock:
                    self.inflight -= 1

    # -- api actions ----------------------------------------------------------------------
    def next_item(self) -> dict | None:
        with self.lock:
            if not self.ready:
                return None
            item = self.ready.popleft()
            self.served[item["id"]] = item
            while len(self.served) > KEEP_SERVED:
                _, old = self.served.popitem(last=False)
                (self.clips / f"{old['id']}.ogg").unlink(missing_ok=True)
            return item

    def label(self, item_id: str, rating: int) -> None:
        with self.lock:
            item = self.served.get(item_id)
            if item is None or item_id in self.ratings:
                raise KeyError(item_id)
            self.ratings[item_id] = {"row": item["row"], "rating": rating}
            self.history.append(item_id)
            self._append({"t": time.time(), "key": item_id, "row": item["row"], "rating": rating})
        self.maybe_refit()

    def undo(self, current_id: str | None) -> dict | None:
        with self.lock:
            if not self.history:
                return None
            key = self.history.pop()
            self.ratings.pop(key, None)
            self._append({"t": time.time(), "undo": key})
            if current_id and current_id in self.served and current_id not in self.ratings:
                self.ready.appendleft(self.served.pop(current_id))
            item = self.served.get(key)
        self.maybe_refit(force=True)
        return item

    # -- model ----------------------------------------------------------------------------
    def _population(self, use_source: bool):
        if use_source not in self._population_cache:
            rows = np.minimum(
                np.searchsorted(
                    self.cum,
                    np.random.default_rng(7).random(POPULATION_SAMPLE) * self.cum[-1],
                    side="right",
                ),
                len(self.cum) - 1,
            )
            self._population_cache[use_source] = tagmodel.frame(
                self.table.x[rows],
                self.table.source[rows],
                len(self.table.sources),
                self.names,
                use_source,
            )
        return self._population_cache[use_source]

    def maybe_refit(self, force: bool = False) -> None:
        with self.lock:
            rated = sum(1 for r in self.ratings.values() if r["rating"] > 0)
            due = force or rated - self.fit_n >= self.args.refit_every or self.fit_result is None
            if not due:
                return
            if self.fitting:
                # A change that arrives mid-fit (new settings, an undo) must not be lost.
                self.refit_pending = self.refit_pending or force
                return
            self.fitting = True
        threading.Thread(target=self._fit, daemon=True).start()

    def _fit(self) -> None:
        try:
            with self.lock:
                chosen = [r for r in self.ratings.values() if r["rating"] > 0]
                settings = dict(self.settings)
            rows = np.array([r["row"] for r in chosen], dtype=int)
            y = np.array([r["rating"] for r in chosen], dtype=int)
            result = tagmodel.analyze(
                tagmodel.frame(
                    self.table.x[rows] if len(rows) else np.zeros((0, len(self.names))),
                    self.table.source[rows] if len(rows) else np.zeros(0, dtype=int),
                    len(self.table.sources),
                    self.names,
                    settings["use_source"],
                ),
                y,
                self._population(settings["use_source"]),
                self.total_hours,
                cutoff=settings["cutoff"],
                seed=len(rows),
            )
            result["settings"] = settings
            with self.lock:
                self.fit_result = result
                self.fit_n = len(rows)
        except Exception as exc:  # noqa: BLE001 -- show the error in the UI rather than die
            print(f"fit failed: {exc}")
            with self.lock:
                self.fit_result = {"error": str(exc), "n": 0}
        finally:
            with self.lock:
                self.fitting = False
                again, self.refit_pending = self.refit_pending, False
            if again:
                self.maybe_refit(force=True)

    def status(self) -> dict:
        with self.lock:
            counts = collections.Counter(r["rating"] for r in self.ratings.values())
            return {
                "ratings": {str(k): counts.get(k, 0) for k in (0, 1, 2, 3)},
                "queue": {
                    "ready": len(self.ready),
                    "inflight": self.inflight,
                    "failed": self.failed,
                },
                "fit": self.fit_result,
                "fitting": self.fitting,
                "settings": self.settings,
                "population": {
                    "rows": len(self.table),
                    "hours": self.total_hours,
                    "sources": len(self.table.sources),
                },
                "disk_fraction": disk_fraction(self.args.cache),
                "disk_limit": self.args.max_disk_fraction,
            }


class Label(BaseModel):
    id: str
    rating: int


class Undo(BaseModel):
    current_id: str | None = None


class Settings(BaseModel):
    use_source: bool
    cutoff: float


def _num(value) -> float | None:
    value = float(value)
    return None if math.isnan(value) else round(value, 4)


def create_app(args: argparse.Namespace):
    from fastapi import FastAPI, HTTPException
    from fastapi.responses import HTMLResponse
    from fastapi.staticfiles import StaticFiles

    tagger = Tagger(args)

    @contextlib.asynccontextmanager
    async def lifespan(_app):
        for _ in range(args.workers):
            threading.Thread(target=tagger.worker, daemon=True).start()
        tagger.maybe_refit(force=True)
        yield
        tagger.stop.set()

    app = FastAPI(title="audio-v2 tagger", lifespan=lifespan)
    app.state.tagger = tagger

    @app.get("/", response_class=HTMLResponse)
    def index() -> str:
        return INDEX.read_text(encoding="utf-8")

    @app.get("/api/next")
    def next_item() -> dict:
        item = tagger.next_item()
        return item if item is not None else {"wait": True, **tagger.status()["queue"]}

    @app.post("/api/label")
    def label(body: Label) -> dict:
        if body.rating not in (0, 1, 2, 3):
            raise HTTPException(400, "rating must be 0 (skip), 1, 2 or 3")
        try:
            tagger.label(body.id, body.rating)
        except KeyError:
            raise HTTPException(404, "unknown or already rated clip") from None
        return tagger.status()

    @app.post("/api/undo")
    def undo(body: Undo) -> dict:
        item = tagger.undo(body.current_id)
        return {"item": item, **tagger.status()}

    @app.post("/api/settings")
    def settings(body: Settings) -> dict:
        with tagger.lock:
            tagger.settings = {"use_source": body.use_source, "cutoff": body.cutoff}
        tagger.maybe_refit(force=True)
        return tagger.status()

    @app.post("/api/refit")
    def refit() -> dict:
        tagger.maybe_refit(force=True)
        return tagger.status()

    @app.get("/api/status")
    def status() -> dict:
        return tagger.status()

    app.mount("/clips", StaticFiles(directory=tagger.clips), name="clips")
    return app


def main() -> None:
    import uvicorn

    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument(
        "--cache",
        type=Path,
        default=Path("/tmp/av2_tagger_cache"),
        help="Local scratch for clips; keep it on fast local disk.",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--ahead", type=int, default=12, help="Clips kept ready ahead of you.")
    parser.add_argument("--workers", type=int, default=3, help="Parallel clip preparations.")
    parser.add_argument("--refit-every", type=int, default=5)
    parser.add_argument(
        "--max-disk-fraction",
        type=float,
        default=0.8,
        help="Pause preloading when the cache disk is fuller than this.",
    )
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    args.cache.mkdir(parents=True, exist_ok=True)
    uvicorn.run(create_app(args), host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
