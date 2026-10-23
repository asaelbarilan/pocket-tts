"""Web preview of what the training dataloader produces.

    uv run python -m hebrew_training.data_preview.server training/configs/finetune_hebrew_curriculum.yaml

Reads a training config, builds the same loaders `training/train.py` builds (the same
`training.dataloader.DataLoader`, the same curriculum mixing, the same text/cut/digit/niqqud
augmentation) and serves a page to look at the batches: the text the model is trained on, and
the voice prompt and target audio of every sample.

What a "phase" and a "split" mean here:

* Phases are the config's `data.curriculum` phases (0-based). A config without a curriculum
  (plain `train_jsonl` / `valid_jsonl`) has one phase 0 holding that dataset.
* Train batches follow the phase's mix, drawn by the training run's own `MixSchedule`.
  Validation is not phase-dependent in training (it is a fixed benchmark over all datasets);
  here, "eval" draws from the validation sets of the datasets in the chosen phase's mix, with
  the mix as weights. Pick a single dataset to look at one on its own.
* "eval" defaults match training's validation loader: seed 0, no shuffle, `fixed_draws` on.

Every edit to the loader parameters (or phase / split / dataset) rebuilds the loaders on the
next request; the stream then starts again from its beginning. Paths in the config are
relative to the repo root, so the server changes to it, as training is run from there.

The loaders are not thread-safe, so one request is served at a time.

Big manifests: the DataLoader reads its whole manifest into memory when built, and is used
here unchanged. For manifests of millions of rows, set the `rows` parameter (or `--max-rows`):
each loader then gets a random subset of that many rows, cached on disk (manifest_sample.py).
The loader's behaviour on those rows is exactly training's; what changes is that "an epoch"
and the shuffle cover the subset, not the full manifest.
"""

from __future__ import annotations

import argparse
import io
import logging
import os
import threading
import time
import wave
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import numpy as np
import sentencepiece
import uvicorn
from fastapi import FastAPI, Header, HTTPException
from fastapi.responses import FileResponse, Response
from pydantic import BaseModel, Field

from hebrew_training.data_preview.manifest_sample import SampledManifest, sample_manifest
from training.args import TrainArgs, load_args
from training.curriculum import Curriculum, CurriculumLoader, DatasetSpec, MixSchedule, Phase
from training.dataloader import DataLoader
from training.modules.builders import load_model_config

logger = logging.getLogger("data_preview")

REPO_ROOT = Path(__file__).resolve().parents[2]
INDEX_HTML = Path(__file__).resolve().parent / "index.html"
MAX_SKIP = 10_000
PREVIEW_IO_WORKERS = 8
SINGLE_DATASET = "default"  # name of the one dataset of a config without a curriculum
DEFAULT_CACHE_DIR = Path.home() / ".cache" / "pocket_tts" / "data_preview"
LARGE_MANIFEST_BYTES = 256 << 20  # the page suggests sampling rows above this


class PreviewError(Exception):
    """A request the loaders cannot satisfy; the message goes to the page."""


class Params(BaseModel):
    batch_size: int = Field(ge=1, le=64)
    seed: int
    shuffle: bool
    fixed_draws: bool
    max_duration_sec: float = Field(gt=0)
    max_voice_prompt_sec: float  # <= 0 removes the window, as in training
    alt_words_sample_prob: float = Field(ge=0, le=1)
    alt_word_swap_prob: float = Field(ge=0, le=1)
    nikud_word_prob: float = Field(ge=0, le=1)
    # Preview only: the loader is given a random subset of this many rows of each manifest
    # (0: the whole manifest, which the loader reads into memory). See manifest_sample.py.
    max_rows: int = Field(ge=0)
    sample_seed: int


class Spec(BaseModel):
    """Everything that decides which batches come out; a change means a new loader."""

    phase: int = Field(ge=0)
    split: Literal["train", "valid"]
    dataset: str | None = None  # None: the phase's whole mix
    params: Params


class NextRequest(BaseModel):
    spec: Spec
    skip: int = Field(0, ge=0, le=MAX_SKIP)  # batches to draw and drop before the one returned
    reload: bool = False  # rebuild the loaders even if `spec` is unchanged (start over)


class PreviewDataLoader(DataLoader):
    """The training loader, plus what the preview shows: the exact text given to the tokenizer
    and where each sample came from. Nothing about the samples themselves changes: the extra
    information rides on the returned token tensor as `.preview`.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._seen = threading.local()
        self._plain_tokenize = self.tokenize
        self.tokenize = self._recording_tokenize

    def _recording_tokenize(self, text: str):
        self._seen.text = text  # `_sample` runs on one pool thread, tokenize included
        return self._plain_tokenize(text)

    def _sample(self, entry):
        self._seen.text = None
        wav, tokens, prompt, length = super()._sample(entry)
        tokens.preview = {
            "text": self._seen.text,
            "transcript": entry.transcript,
            "path": entry.path,
            "start": entry.start,
            "row_duration": entry.duration,
            "has_words": bool(entry.words),
            "has_alt_words": bool(entry.alt_words),
            "has_nikud": bool(entry.nikud),
            "augmented": entry.text_seed is not None,
            "prompt_samples": int(length),
            "target_samples": len(wav),
        }
        return wav, tokens, prompt, length


@dataclass
class LoadedBatch:
    id: int
    sample_rate: int
    prompts: list[np.ndarray]
    targets: list[np.ndarray]
    wav_cache: dict[tuple[int, str], bytes] = field(default_factory=dict)

    def wav(self, index: int, kind: str) -> bytes:
        key = (index, kind)
        if key not in self.wav_cache:
            audio = (self.prompts if kind == "prompt" else self.targets)[index]
            pcm = (np.clip(audio, -1.0, 1.0) * 32767).astype("<i2")
            buf = io.BytesIO()
            with wave.open(buf, "wb") as w:
                w.setnchannels(1)
                w.setsampwidth(2)
                w.setframerate(self.sample_rate)
                w.writeframes(pcm.tobytes())
            self.wav_cache[key] = buf.getvalue()
        return self.wav_cache[key]


def single_dataset_curriculum(args: TrainArgs) -> Curriculum:
    if not args.data.train_jsonl:
        raise PreviewError("the config has neither data.datasets nor data.train_jsonl")
    spec = DatasetSpec(
        SINGLE_DATASET,
        args.data.train_jsonl,
        args.data.valid_jsonl,
        1.0 if args.data.valid_jsonl else 0.0,
    )
    return Curriculum({SINGLE_DATASET: spec}, (Phase(1, {SINGLE_DATASET: 1.0}),))


class Session:
    """The config, the tokenizer, and the one live loader with the batch last returned."""

    def __init__(
        self,
        config_path: Path,
        tokenizer_path: str | None,
        max_rows: int = 0,
        cache_dir: Path = DEFAULT_CACHE_DIR,
    ):
        self.config_path = config_path
        self.default_max_rows = max_rows
        self.cache_dir = cache_dir
        self.args = load_args(config_path)
        self.curriculum = self.args.data.resolve_curriculum()
        self.is_curriculum = self.curriculum is not None
        if self.curriculum is None:
            self.curriculum = single_dataset_curriculum(self.args)

        overrides = dict(self.args.model_overrides)
        if tokenizer_path:
            overrides["flow_lm.lookup_table.tokenizer_path"] = tokenizer_path
        model_config = load_model_config(self.args.model_config, overrides)
        self.tokenizer_path = str(model_config.flow_lm.lookup_table.tokenizer_path)
        self.sample_rate = int(model_config.mimi.sample_rate)
        self.frame_rate = float(model_config.mimi.frame_rate)
        from pocket_tts.utils.utils import download_if_necessary

        self.sp = sentencepiece.SentencePieceProcessor(
            model_file=str(download_if_necessary(self.tokenizer_path))
        )

        self.lock = threading.Lock()
        self.spec: Spec | None = None
        self.mixed: CurriculumLoader | None = None
        self.loaders: dict[str, PreviewDataLoader] = {}
        self.sampled: dict[str, SampledManifest] = {}
        self.drawn = 0
        self.batch: LoadedBatch | None = None
        self._next_id = 0

    @staticmethod
    def _file_bytes(path: str) -> int | None:
        try:
            return Path(path).stat().st_size if path else None
        except OSError:
            return None

    def info(self) -> dict:
        data = self.args.data
        datasets = {
            name: {
                "train_jsonl": spec.train_jsonl,
                "valid_jsonl": spec.valid_jsonl,
                "valid_weight": spec.valid_weight,
                "train_bytes": self._file_bytes(spec.train_jsonl),
                "valid_bytes": self._file_bytes(spec.valid_jsonl),
            }
            for name, spec in self.curriculum.datasets.items()
        }
        phases = [
            {"index": i, "steps": phase.steps if self.is_curriculum else None, "mix": phase.mix}
            for i, phase in enumerate(self.curriculum.phases)
        ]
        base = {
            "batch_size": self.args.batch_size,
            "max_duration_sec": data.max_duration_sec,
            "max_voice_prompt_sec": data.max_voice_prompt_sec,
            "alt_words_sample_prob": data.alt_words_sample_prob,
            "alt_word_swap_prob": data.alt_word_swap_prob,
            "nikud_word_prob": data.nikud_word_prob,
            "max_rows": self.default_max_rows,
            "sample_seed": 0,
        }
        return {
            "large_manifest_bytes": LARGE_MANIFEST_BYTES,
            "config": str(self.config_path),
            "is_curriculum": self.is_curriculum,
            "datasets": datasets,
            "phases": phases,
            "sample_rate": self.sample_rate,
            "frame_rate": self.frame_rate,
            "tokenizer": self.tokenizer_path,
            "grad_accum_steps": self.args.grad_accum_steps,
            # What training uses: `args.seed + rank` and the configured shuffle for train,
            # seed 0, no shuffle and per-row fixed draws for validation (see _valid_pass).
            "defaults": {
                "train": {
                    **base,
                    "seed": self.args.seed,
                    "shuffle": data.shuffle,
                    "fixed_draws": False,
                },
                "valid": {**base, "seed": 0, "shuffle": False, "fixed_draws": True},
            },
        }

    def close(self) -> None:
        if self.mixed is not None:
            self.mixed.close()
        self.mixed = None
        self.loaders = {}
        self.sampled = {}
        self.spec = None
        self.drawn = 0

    def _build(self, spec: Spec) -> None:
        self.close()
        phases = self.curriculum.phases
        if spec.phase >= len(phases):
            raise PreviewError(f"phase {spec.phase} does not exist: the config has {len(phases)}")
        datasets = self.curriculum.datasets
        attr = "train_jsonl" if spec.split == "train" else "valid_jsonl"
        mix = dict(phases[spec.phase].mix)
        if spec.dataset is not None:
            if spec.dataset not in mix:
                raise PreviewError(f"dataset {spec.dataset!r} is not in phase {spec.phase}'s mix")
            mix = {spec.dataset: 1.0}
        mix = {name: w for name, w in mix.items() if getattr(datasets[name], attr)}
        if not mix:
            raise PreviewError(f"no {spec.split} manifest for the datasets of phase {spec.phase}")
        total = sum(mix.values())
        mix = {name: w / total for name, w in mix.items()}
        sub = Curriculum({name: datasets[name] for name in mix}, (Phase(1, mix),))

        p = spec.params
        sample_rate, frame_rate = self.sample_rate, self.frame_rate

        def open_stream(name: str):
            manifest = getattr(datasets[name], attr)
            if p.max_rows:
                self.sampled[name] = sample_manifest(
                    manifest, p.max_rows, p.sample_seed, self.cache_dir
                )
                manifest = str(self.sampled[name].path)
            loader = PreviewDataLoader(
                manifest,
                self.sp.encode,
                p.batch_size,
                sample_rate,
                frame_rate,
                p.max_duration_sec,
                p.max_voice_prompt_sec,
                0,
                1,
                seed=p.seed,
                shuffle=p.shuffle,
                io_workers=PREVIEW_IO_WORKERS,
                alt_words_sample_prob=p.alt_words_sample_prob,
                alt_word_swap_prob=p.alt_word_swap_prob,
                nikud_word_prob=p.nikud_word_prob,
                fixed_draws=p.fixed_draws,
            )
            self.loaders[name] = loader
            return iter(loader), len(loader.entries)

        accum = self.args.grad_accum_steps
        self.mixed = CurriculumLoader(MixSchedule(sub, accum), open_stream, 0, 1, p.batch_size)
        self.spec = spec

    def _draw(self):
        """The next batch of the stream and the dataset it came from."""
        before = dict(self.mixed.batches_drawn)
        accum = self.args.grad_accum_steps
        batch = self.mixed.next_batch(self.drawn // accum, self.drawn % accum)
        self.drawn += 1
        name = next(n for n, c in self.mixed.batches_drawn.items() if c != before.get(n, 0))
        return batch, name

    def next(self, request: NextRequest) -> dict:
        started = time.perf_counter()
        reloaded = request.reload or self.mixed is None or request.spec != self.spec
        try:
            if reloaded:
                self._build(request.spec)
            for _ in range(request.skip):
                self._draw()
            batch, dataset = self._draw()
        except PreviewError:
            self.close()
            raise
        except Exception as exc:
            logger.exception("loading a batch failed")
            self.close()
            raise PreviewError(f"{type(exc).__name__}: {exc}") from exc

        self._next_id += 1
        prompts, targets, samples = [], [], []
        for b, tokens in enumerate(batch.text_tokens):
            meta = tokens.preview
            ids = tokens.tolist()
            prompts.append(batch.voice_audio[b, 0, : meta["prompt_samples"]].numpy())
            targets.append(batch.audio[b, 0, : meta["target_samples"]].numpy())
            samples.append(
                {
                    **meta,
                    "index": b,
                    "token_ids": ids,
                    "pieces": [self.sp.id_to_piece(i) for i in ids],
                    "prompt_sec": meta["prompt_samples"] / self.sample_rate,
                    "target_sec": meta["target_samples"] / self.sample_rate,
                    "prompt_url": f"/api/audio/{self._next_id}/{b}/prompt",
                    "target_url": f"/api/audio/{self._next_id}/{b}/target",
                }
            )
        self.batch = LoadedBatch(self._next_id, self.sample_rate, prompts, targets)
        return {
            "id": self._next_id,
            "number": self.drawn,
            "dataset": dataset,
            "samples": samples,
            "skipped": request.skip,
            "reloaded": reloaded,
            "elapsed_ms": round((time.perf_counter() - started) * 1000),
            "status": self.status(),
        }

    def status(self) -> dict:
        assert self.mixed is not None
        return {
            "batches_drawn": dict(self.mixed.batches_drawn),
            "epochs": self.mixed.epochs(),
            "rows": {name: len(loader.entries) for name, loader in self.loaders.items()},
            "sampled_from_bytes": {n: s.source_bytes for n, s in self.sampled.items()},
            "unreadable_skipped": {name: l._failures for name, l in self.loaders.items()},
        }


def audio_response(data: bytes, range_header: str | None) -> Response:
    """The wav, with byte ranges: Safari will not play an <audio> source without them."""
    size = len(data)
    headers = {"Accept-Ranges": "bytes", "Cache-Control": "no-store"}
    if range_header and range_header.startswith("bytes="):
        first, _, last = range_header.removeprefix("bytes=").partition("-")
        try:
            if first:
                start, end = int(first), int(last) if last else size - 1
            else:
                start, end = max(0, size - int(last)), size - 1
        except ValueError:
            raise HTTPException(416, "bad range") from None
        end = min(end, size - 1)
        if start > end:
            raise HTTPException(416, "range not satisfiable")
        headers["Content-Range"] = f"bytes {start}-{end}/{size}"
        return Response(data[start : end + 1], 206, headers, media_type="audio/wav")
    return Response(data, headers=headers, media_type="audio/wav")


def create_app(session: Session) -> FastAPI:
    app = FastAPI(title="Dataloader preview")

    @app.get("/")
    def index():
        return FileResponse(INDEX_HTML, headers={"Cache-Control": "no-store"})

    @app.get("/api/config")
    def config():
        return session.info()

    # Plain `def` endpoints run in FastAPI's thread pool, so a slow batch does not block the
    # page's other requests; the lock keeps the loaders to one user at a time.
    @app.post("/api/next")
    def next_batch(request: NextRequest):
        with session.lock:
            try:
                return session.next(request)
            except PreviewError as exc:
                raise HTTPException(400, str(exc)) from exc

    @app.get("/api/audio/{batch_id}/{index}/{kind}")
    def audio(
        batch_id: int,
        index: int,
        kind: Literal["prompt", "target"],
        range: str | None = Header(None),
    ):
        batch = session.batch
        if batch is None or batch.id != batch_id or not 0 <= index < len(batch.targets):
            raise HTTPException(404, "that batch has been replaced")
        return audio_response(batch.wav(index, kind), range)

    return app


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("config", type=Path, help="training config yaml")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument(
        "--tokenizer",
        help="sentencepiece model to use instead of the config's "
        "model_overrides / model_config tokenizer_path (e.g. when it is not built on this machine)",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=0,
        help="default for the page's `rows` parameter: give each loader a random subset of this "
        "many rows per manifest instead of the whole file (0: whole). The DataLoader reads its "
        "entire manifest into memory, so set this for manifests of millions of rows",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=DEFAULT_CACHE_DIR,
        help="where the sampled manifests are kept (default: %(default)s)",
    )
    cli = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    config_path = cli.config.resolve()
    tokenizer = str(Path(cli.tokenizer).resolve()) if cli.tokenizer else None
    os.chdir(REPO_ROOT)  # the config's relative paths are relative to the repo root
    session = Session(config_path, tokenizer, cli.max_rows, cli.cache_dir.expanduser().resolve())
    logger.info(
        "config %s: %s, tokenizer %s, %d Hz",
        config_path,
        f"curriculum of {len(session.curriculum.phases)} phases"
        if session.is_curriculum
        else "single dataset (phase 0)",
        session.tokenizer_path,
        session.sample_rate,
    )
    uvicorn.run(create_app(session), host=cli.host, port=cli.port, log_level="info")


if __name__ == "__main__":
    main()
