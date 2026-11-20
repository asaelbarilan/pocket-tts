"""Local web UI for the voice-prompt pool: extract, preview, compare.

    uv run python -m hebrew_training.voice_prompts.ui [--host 127.0.0.1] [--port 8765]

FastAPI serves `index.html` (Alpine.js from a CDN, no build step) and a small JSON API. Paths are
paths on the machine running this server, so it binds to localhost by default. One extraction
runs at a time; it is `extract.run` in a thread, so the CLI and the UI do the same work, and
every run *adds* to the target folder (`--append`) unless "start fresh" is ticked.
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import secrets
import threading
import time
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, HTMLResponse
from pydantic import BaseModel, Field

from hebrew_training.manifest_files import SPLITS
from hebrew_training.voice_prompts import dedupe, extract

logger = logging.getLogger("voice_prompts_ui")
INDEX_HTML = Path(__file__).with_name("index.html")
REFERENCE_SAMPLE = 200  # prompts embedded to calibrate similarity when no embeddings.npz exists
REFERENCE_MAX = 2000  # vectors taken from embeddings.npz

app = FastAPI(title="Voice prompt pool")


# ---------------------------------------------------------------------------------------------
# Target folder: manifest index (byte offsets, so a random sample never loads the whole file)
# ---------------------------------------------------------------------------------------------


class ManifestIndex:
    def __init__(self, path: Path):
        self.path = path
        self.stamp = (path.stat().st_mtime_ns, path.stat().st_size)
        self.ids: list[int] = []
        self.offsets: list[int] = []
        self.position: dict[int, int] = {}
        speakers, datasets = set(), {}
        seconds = 0.0
        rates = set()
        with path.open("rb") as handle:
            while True:
                offset = handle.tell()
                line = handle.readline()
                if not line:
                    break
                if not line.strip():
                    continue
                row = json.loads(line)
                self.position[row["id"]] = len(self.ids)
                self.ids.append(row["id"])
                self.offsets.append(offset)
                speakers.add((row["dataset"], extract.speaker_key(row)))
                datasets[row["dataset"]] = datasets.get(row["dataset"], 0) + 1
                seconds += row["duration"]
                rates.add(row["sample_rate"])
        count = len(self.ids)
        self.summary = {
            "exists": True,
            "count": count,
            "speakers": len(speakers),
            "hours": round(seconds / 3600, 3),
            "mean_duration": round(seconds / count, 3) if count else None,
            "datasets": datasets,
            "sample_rates": sorted(rates),
        }

    def row(self, index: int) -> dict:
        with self.path.open("rb") as handle:
            handle.seek(self.offsets[index])
            return json.loads(handle.readline())


_indexes: dict[str, ManifestIndex] = {}
_index_lock = threading.Lock()


def target_dir(output: str) -> Path:
    if not output.strip():
        raise HTTPException(400, "target folder is empty")
    return Path(output).expanduser()


def manifest_index(output: str) -> ManifestIndex | None:
    path = target_dir(output) / extract.MANIFEST
    if not path.is_file():
        return None
    stamp = (path.stat().st_mtime_ns, path.stat().st_size)
    with _index_lock:
        index = _indexes.get(str(path))
        if index is None or index.stamp != stamp:
            index = _indexes[str(path)] = ManifestIndex(path)
        return index


def with_urls(output: str, row: dict) -> dict:
    return {**row, "audio_url": f"/api/audio?output={_quote(output)}&file={_quote(row['audio'])}"}


def _quote(text: str) -> str:
    from urllib.parse import quote

    return quote(text, safe="")


# ---------------------------------------------------------------------------------------------
# Datasets
# ---------------------------------------------------------------------------------------------


class DatasetCheck(BaseModel):
    specs: list[str]


@app.post("/api/datasets/check")
def check_datasets(body: DatasetCheck) -> list[dict]:
    """For every `[NAME=]DIR`: does it exist, and which official manifests does it have."""
    results = []
    for spec in body.specs:
        name, separator, location = spec.partition("=")
        folder = Path(location if separator else spec).expanduser()
        name = name if separator else folder.resolve().name
        entry: dict[str, Any] = {"spec": spec, "name": name, "path": str(folder), "files": []}
        if not folder.is_dir():
            entry["error"] = "not a directory"
        else:
            for split in SPLITS:
                found = extract.resolve_manifest(folder, split, "auto")
                if found:
                    path, stage = found
                    entry["files"].append(
                        {
                            "split": split,
                            "stage": stage,
                            "file": path.name,
                            "mb": round(path.stat().st_size / 1e6, 1),
                        }
                    )
            if not entry["files"]:
                entry["error"] = "no train/valid manifest (aligned, diac or built)"
        results.append(entry)
    return results


# ---------------------------------------------------------------------------------------------
# Extraction job
# ---------------------------------------------------------------------------------------------


class ExtractRequest(BaseModel):
    datasets: list[str]
    output: str
    fresh: bool = False
    count: int = Field(1000, ge=1, description="new prompts to add in this run")
    splits: list[str] = list(SPLITS)
    stage: str = "auto"
    per_speaker: int = Field(10, ge=0)
    per_row: int = Field(1, ge=1)
    min_duration: float = 3.0
    max_duration: float = 5.0
    min_gap_ms: float = 300.0
    edge_cuts: bool = True
    min_words: int = 3
    max_inner_gap_ms: float = 1000.0
    min_speech_ratio: float = 0.6
    sample_rate: int = 24_000
    workers: int | None = None
    seed: int | None = None


class _Parser(argparse.ArgumentParser):
    def error(self, message: str):  # raise instead of exiting the server
        raise ValueError(message)


class Job:
    """The single extraction that may run; `snapshot()` is what the page polls."""

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.stop = threading.Event()
        self.thread: threading.Thread | None = None
        self.data: dict[str, Any] = {"state": "idle"}

    def running(self) -> bool:
        return self.thread is not None and self.thread.is_alive()

    def update(self, **fields: Any) -> None:
        with self.lock:
            self.data.update(fields)

    def snapshot(self) -> dict:
        with self.lock:
            return {**self.data, "elapsed": self._elapsed()}

    def _elapsed(self) -> float | None:
        if "started" not in self.data:
            return None
        return round(self.data.get("finished", time.time()) - self.data["started"], 1)


JOB = Job()


def build_args(request: ExtractRequest, seed: int) -> argparse.Namespace:
    parser = _Parser(add_help=False)
    extract.add_arguments(parser)
    argv = [
        *request.datasets,
        "--output", request.output,
        "--splits", *request.splits,
        "--stage", request.stage,
        "--max-prompts", str(request.count),
        "--per-speaker", str(request.per_speaker),
        "--per-row", str(request.per_row),
        "--min-duration", str(request.min_duration),
        "--max-duration", str(request.max_duration),
        "--min-gap-ms", str(request.min_gap_ms),
        "--edge-cuts" if request.edge_cuts else "--no-edge-cuts",
        "--min-words", str(request.min_words),
        "--max-inner-gap-ms", str(request.max_inner_gap_ms),
        "--min-speech-ratio", str(request.min_speech_ratio),
        "--sample-rate", str(request.sample_rate),
        "--seed", str(seed),
        "--overwrite" if request.fresh else "--append",
    ]  # fmt: skip
    if request.workers:
        argv += ["--workers", str(request.workers)]
    args = parser.parse_args(argv)
    extract.check_arguments(parser, args)
    return args


@app.post("/api/extract")
def start_extract(request: ExtractRequest) -> dict:
    if JOB.running():
        raise HTTPException(409, "an extraction is already running")
    if not request.datasets:
        raise HTTPException(400, "add at least one dataset")
    seed = request.seed if request.seed is not None else secrets.randbelow(2**31)
    try:
        args = build_args(request, seed)
        extract.parse_datasets(args.datasets)  # fail now, not in the thread
    except (ValueError, SystemExit) as error:
        raise HTTPException(400, str(error.args[0] if error.args else error)) from error

    JOB.stop.clear()

    def progress(phase: str, done: int, total: int | None) -> None:
        JOB.update(phase=phase, done=done, total=total)

    def work() -> None:
        try:
            stats = extract.run(args, progress, JOB.stop.is_set)
            JOB.update(state="done", stats=dict(stats), finished=time.time())
        except extract.Cancelled:
            JOB.update(state="cancelled", finished=time.time())
        except SystemExit as error:
            JOB.update(state="error", error=str(error.code), finished=time.time())
        except Exception as error:  # reported to the page, with the traceback in the log
            logger.exception("extraction failed")
            JOB.update(state="error", error=repr(error), finished=time.time())

    with JOB.lock:
        JOB.data = {
            "state": "running",
            "phase": "starting",
            "done": 0,
            "total": None,
            "seed": seed,
            "output": request.output,
            "requested": request.count,
            "started": time.time(),
        }
    JOB.thread = threading.Thread(target=work, daemon=True)
    JOB.thread.start()
    return JOB.snapshot()


@app.post("/api/cancel")
def cancel() -> dict:
    JOB.stop.set()
    return JOB.snapshot()


@app.get("/api/job")
def job() -> dict:
    return JOB.snapshot()


# ---------------------------------------------------------------------------------------------
# Target folder: summary, random samples, audio
# ---------------------------------------------------------------------------------------------


@app.get("/api/target")
def target(output: str) -> dict:
    index = manifest_index(output)
    return index.summary if index else {"exists": False, "count": 0}


@app.get("/api/samples")
def samples(output: str, n: int = Query(4, ge=1, le=50)) -> list[dict]:
    index = manifest_index(output)
    if index is None or not index.ids:
        raise HTTPException(404, "no prompts in that folder yet")
    picks = random.sample(range(len(index.ids)), min(n, len(index.ids)))
    return [with_urls(output, index.row(i)) for i in picks]


@app.get("/api/audio")
def audio(output: str, file: str) -> FileResponse:
    prompts = (target_dir(output) / extract.PROMPTS_DIR).resolve()
    path = (target_dir(output) / file).resolve()
    if path.parent != prompts or path.suffix != ".wav" or not path.is_file():
        raise HTTPException(404, "no such prompt")
    return FileResponse(path, media_type="audio/wav")


# ---------------------------------------------------------------------------------------------
# Similarity debugging, in the space `dedupe` uses
# ---------------------------------------------------------------------------------------------


class SimilarityRequest(BaseModel):
    output: str
    a: int | None = None
    b: int | None = None
    space: str = "latent"
    language: str = "english"
    config: str = ""
    checkpoint: str = ""


MODEL_LOCK = threading.Lock()
_models: dict[tuple, Any] = {}
_vectors: dict[tuple, Any] = {}
_references: dict[tuple, dict] = {}


def embedding_settings(request: SimilarityRequest) -> argparse.Namespace:
    if request.space not in dedupe.SPACES:
        raise HTTPException(400, f"space must be one of {dedupe.SPACES}")
    return argparse.Namespace(
        space=request.space,
        language=request.language or "english",
        config=request.config.strip() or None,
        checkpoint=request.checkpoint.strip() or None,
    )


def pooled(settings: argparse.Namespace, output: str, row: dict):
    """The pooled vector of one prompt; the model loads on first use."""
    label = dedupe.source_label(settings)
    key = (str(target_dir(output)), label, row["id"])
    if key not in _vectors:
        model_key = (settings.language, settings.config, settings.checkpoint)
        if model_key not in _models:
            logger.info("loading the model for %s", label)
            _models[model_key] = dedupe.load_prompt_model(settings)
        path = target_dir(output) / row["audio"]
        _vectors[key] = dedupe.pooled_embedding(_models[model_key], path, settings.space)
    return _vectors[key]


def reference(settings: argparse.Namespace, output: str, index: ManifestIndex) -> dict:
    """Standardization stats and a standardized sample of the pool, to calibrate a pair against.

    Taken from the folder's `embeddings.npz` (a dedupe run) when it matches, else from
    `REFERENCE_SAMPLE` random prompts embedded now."""
    import numpy as np

    label = dedupe.source_label(settings)
    key = (str(target_dir(output)), label, index.stamp)
    if key not in _references:
        cached = dedupe.load_cache(target_dir(output) / dedupe.EMBEDDINGS, label)
        if len(cached) >= 20:
            wanted = random.Random(0).sample(sorted(cached), min(REFERENCE_MAX, len(cached)))
            pooled_set = np.stack([cached[i] for i in wanted])
            origin = f"embeddings.npz ({len(wanted)} prompts)"
        else:
            picks = random.Random(0).sample(
                range(len(index.ids)), min(REFERENCE_SAMPLE, len(index.ids))
            )
            pooled_set = np.stack([pooled(settings, output, index.row(i)) for i in picks])
            origin = f"{len(picks)} random prompts"
        fitted = dedupe.fit_standardizer(pooled_set)
        matrix = dedupe.standardize(pooled_set, fitted)
        pairs = matrix @ matrix.T
        _references[key] = {
            "fitted": fitted,
            "origin": origin,
            "pair_similarities": np.sort(pairs[np.triu_indices(len(matrix), 1)]),
            "nearest": dedupe.nearest_neighbour_quantiles(matrix),
        }
    return _references[key]


@app.post("/api/similarity")
def similarity(request: SimilarityRequest) -> dict:
    import numpy as np

    index = manifest_index(request.output)
    if index is None or len(index.ids) < 2:
        raise HTTPException(404, "need at least two prompts in the target folder")
    for wanted in (request.a, request.b):
        if wanted is not None and wanted not in index.position:
            raise HTTPException(404, f"no prompt with id {wanted}")
    ids = [request.a, request.b]
    if None in ids:
        pool = [i for i in index.ids if i not in ids]
        fill = random.sample(pool, ids.count(None))
        ids = [fill.pop() if value is None else value for value in ids]
    if ids[0] == ids[1]:
        raise HTTPException(400, "pick two different prompts")
    rows = [index.row(index.position[i]) for i in ids]
    settings = embedding_settings(request)
    try:
        with MODEL_LOCK:
            vectors = [pooled(settings, request.output, row) for row in rows]
            stats = reference(settings, request.output, index)
    except Exception as error:
        logger.exception("similarity failed")
        raise HTTPException(500, f"{type(error).__name__}: {error}") from error
    pair = dedupe.standardize(np.stack(vectors), stats["fitted"])
    value = float(pair[0] @ pair[1])
    raw = float(vectors[0] @ vectors[1] / (np.linalg.norm(vectors[0]) * np.linalg.norm(vectors[1])))
    sorted_pairs = stats["pair_similarities"]
    nearest = None
    if stats["nearest"] is not None:
        nearest = {
            name: round(float(q), 4)
            for name, q in zip(["p5", "p25", "p50", "p75", "p95"], stats["nearest"], strict=True)
        }
    return {
        "a": with_urls(request.output, rows[0]),
        "b": with_urls(request.output, rows[1]),
        "similarity": round(value, 4),
        "raw_cosine": round(raw, 4),
        "percentile": round(
            100 * float(np.searchsorted(sorted_pairs, value) / len(sorted_pairs)), 1
        ),
        "reference": stats["origin"],
        "nearest_neighbour": nearest,
        "same_speaker": extract.speaker_key(rows[0]) == extract.speaker_key(rows[1]),
        "space": request.space,
    }


# ---------------------------------------------------------------------------------------------


@app.get("/", response_class=HTMLResponse)
def page() -> str:
    return INDEX_HTML.read_text(encoding="utf-8")


def main(argv: list[str] | None = None) -> None:
    import uvicorn

    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
