"""Standalone HTTP API for testing the Hebrew 24-layer model: text + voice prompt -> WAV.

Run from the repo root:

    uv run python -m deploy.hebrew_tts_server                      # cuda > mps > cpu
    uv run python -m deploy.hebrew_tts_server --device cpu --port 8002

Why this exists instead of `pocket-tts serve`: the model was trained with CFG dropout and is
evaluated with cfg=2.0, and `pocket_tts` has no CFG. This server samples with the training
code (`TrainableTTS.generate`), so it matches the training evals: same CFG, EOS threshold,
temperature and decoding. The weights are the exported artifacts/model.safetensors, loaded
through the config in artifacts/hebrew_24l.yaml. Or, with env MODEL_STORE (a directory of
published models, see deploy/model_store.py) and MODEL_SHA256 (which one), the model is copied
from the store and verified at startup. A model whose config has a semantic_conditioner section
also gets its frozen encoder (DictaBERT or RenikudPlus) from the store, found by the identity the
config names (deploy/encoder_store.py); a model without the section needs none, so one image
serves all three kinds.

The text goes to the tokenizer with only its characters folded (curly quotes, Unicode dashes,
invisible marks -- what training saw), with no chunking and no other normalization, and is
generated in a single pass of at most `max_seconds`.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import io
import json
import logging
import os
import re
import tempfile
import threading
import time
from collections import OrderedDict
from pathlib import Path

import soundfile
import torch
import uvicorn
import yaml
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import (
    FileResponse,
    HTMLResponse,
    JSONResponse,
    RedirectResponse,
    Response,
    StreamingResponse,
)

from deploy.downloads import CODE_ZIP, GUIDE, Downloads
from deploy.encoder_store import EncoderStore, EncoderStoreSet, describe, semantic_configs_of
from deploy.milim_bench import DEFAULT_DIR as DEFAULT_MILIM_DIR
from deploy.milim_bench import MAX_BATCH, MilimBench
from deploy.model_store import ModelStore
from deploy.validation_set import ValidationBusy, ValidationSet
from pocket_tts import TTSModel
from pocket_tts.data.audio import audio_read
from pocket_tts.data.audio_utils import convert_audio
from pocket_tts.modules.stateful_module import init_states
from pocket_tts.utils.text_fold import fold_characters
from training.args import TrainArgs
from training.modules.model import TrainableTTS
from training.modules.samplers import LSD

logger = logging.getLogger("hebrew_tts_server")

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = REPO_ROOT / "artifacts" / "hebrew_24l.yaml"
DEFAULT_VOICES_DIR = Path(__file__).resolve().parent / "voices"
DEFAULT_SENTENCES = Path(__file__).resolve().parent / "eval_sentences_he.jsonl"
DEFAULT_CACHE_DIR = Path(__file__).resolve().parent / "validation_cache"
AUDIO_SUFFIXES = {".wav", ".flac", ".mp3", ".ogg", ".m4a"}
MIN_FRAMES = 8  # shorter generations are an immediate EOS, not speech (as in training eval)
MAX_TEXT_CHARS = 1500
VOICE_CACHE_SIZE = 32

# Training and training evals use these; every one can be overridden per request.
DEFAULTS = {
    "temp": 0.3,
    "cfg": 2.0,
    "n_steps": 1,
    "eos_threshold": -1.0,
    "eos_countdown": 3,
    "max_seconds": 30.0,
    "prompt_seconds": 5.0,
}


def open_encoder_store(
    config: Path, model_store_dir: str | None
) -> EncoderStore | EncoderStoreSet | None:
    """The store of the frozen semantic encoder the model's config names (an EncoderStoreSet,
    one per encoder, for a combined model), or None when it needs none or no store is
    configured (then the encoders come from the Hugging Face cache).

    Env SEMANTIC_ENCODER_STORE says where encoders are copied from, default
    `<MODEL_STORE>/encoders`; SEMANTIC_ENCODER_SHA256 pins one by hash (it must still be the
    encoder the config names; comma-separated, one per encoder, for a combined model). A
    missing or mismatching encoder is a startup error, like a missing model.
    """
    encoders_dir = os.environ.get("SEMANTIC_ENCODER_STORE") or (
        str(Path(model_store_dir) / "encoders") if model_store_dir else None
    )
    pin = os.environ.get("SEMANTIC_ENCODER_SHA256") or None
    try:
        semantic = semantic_configs_of(config)
    except (OSError, ValueError) as exc:
        raise SystemExit(f"{config}: {exc}") from exc
    if not semantic:
        if pin:
            logger.warning("SEMANTIC_ENCODER_SHA256 is set but the model has no semantic encoder")
        logger.info("semantic conditioning: off (the config has no semantic_conditioner)")
        return None
    described = " + ".join(describe(c) for c in semantic)
    if encoders_dir is None:
        logger.info(
            "semantic conditioning: %s, from the Hugging Face cache (no encoder store set)",
            described,
        )
        return None
    try:
        if len(semantic) == 1:
            encoder_store = EncoderStore(Path(encoders_dir), semantic[0], REPO_ROOT, pin)
        else:
            pins = [p.strip() or None for p in pin.split(",")] if pin else None
            encoder_store = EncoderStoreSet(Path(encoders_dir), semantic, REPO_ROOT, pins)
    except (OSError, ValueError, KeyError) as exc:
        raise SystemExit(f"encoder store: {exc}") from exc
    logger.info(
        "semantic conditioning: %s, encoder store %s, encoder %s",
        described,
        encoders_dir,
        encoder_store.sha256,
    )
    return encoder_store


def pick_device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def list_voice_files(voices_dir: Path) -> dict[str, Path]:
    if not voices_dir.is_dir():
        return {}
    return {
        p.name: p
        for p in sorted(voices_dir.iterdir())
        if p.is_file() and p.suffix.lower() in AUDIO_SUFFIXES and not p.name.startswith(".")
    }


class Engine:
    """The loaded model plus a voice-latent cache. One request generates at a time."""

    def __init__(
        self,
        config: Path,
        voices_dir: Path,
        device: torch.device,
        semantic_encoder_dir: Path | list[Path] | None = None,
    ):
        self.config = config
        self.voices_dir = voices_dir
        self.device = device
        self.lock = threading.Lock()
        self._voice_cache: OrderedDict[tuple, torch.Tensor] = OrderedDict()

        started = time.perf_counter()
        # Builds FlowLM + Mimi from the config and loads artifacts/model.safetensors (strict).
        # With a semantic_conditioner section the frozen encoder is loaded here too: from
        # semantic_encoder_dir (copied from the encoder store) when there is one, else from the
        # Hugging Face cache. A config without the section ignores the directory.
        tts = TTSModel.load_model(config=config, semantic_encoder_dir=semantic_encoder_dir)
        self.sample_rate: int = tts.sample_rate
        self.mimi = tts.mimi.to(device).eval()
        flow_lm = tts.flow_lm.to(device).eval()
        if flow_lm.semantic_fusion is not None:
            for enc in getattr(flow_lm.semantic_fusion, "_encoders", ()):
                if getattr(enc, "model", None) is not None:
                    enc.model.to(device)
        # The LSD sampler has no weights of its own at decode time; the wrapper only provides
        # the training-time `generate` (CFG, right-aligned batching, EOS countdown).
        self.model = TrainableTTS(flow_lm, LSD(normalize=False), TrainArgs()).eval()
        sp_encode = flow_lm.conditioner.tokenizer.sp.encode
        # Curly quotes, Unicode dashes and the like never reached the model in training.
        self.tokenize = lambda text: sp_encode(fold_characters(text))
        self.n_layers = len(flow_lm.transformer.layers)
        logger.info(
            "loaded %d-layer model on %s in %.1fs",
            self.n_layers,
            device,
            time.perf_counter() - started,
        )

    # -- voices -------------------------------------------------------------------------

    def list_voices(self) -> dict[str, Path]:
        return list_voice_files(self.voices_dir)

    def _encode_voice(self, path: Path, prompt_seconds: float) -> torch.Tensor:
        audio, sr = audio_read(path)  # [channels, samples]
        if prompt_seconds > 0:
            audio = audio[..., : int(prompt_seconds * sr)]
        if audio.shape[-1] < sr // 2:
            raise HTTPException(status_code=400, detail="voice prompt is shorter than 0.5 s")
        mono = convert_audio(audio, sr, self.sample_rate, 1)  # [1, T]
        with torch.no_grad():
            latents = self.mimi.encode_to_latent(mono[None].to(self.device))  # [1, Tv, C]
        return latents[0].float()

    def voice_latents(self, key: tuple, path: Path, prompt_seconds: float) -> torch.Tensor:
        key = (*key, prompt_seconds)
        if key in self._voice_cache:
            self._voice_cache.move_to_end(key)
            return self._voice_cache[key]
        latents = self._encode_voice(path, prompt_seconds)
        self._voice_cache[key] = latents
        while len(self._voice_cache) > VOICE_CACHE_SIZE:
            self._voice_cache.popitem(last=False)
        return latents

    # -- generation ---------------------------------------------------------------------

    def _decode(self, latents: torch.Tensor) -> torch.Tensor:
        ratio = round(self.mimi.encoder_frame_rate / self.mimi.frame_rate)
        state = init_states(self.mimi, 1, (latents.shape[0] + MIN_FRAMES) * ratio)
        with torch.no_grad():
            return self.mimi.decode_from_latent(latents[None].to(self.device), state)[0, 0]

    def synthesize(
        self,
        text: str,
        voice: torch.Tensor,
        *,
        temp: float,
        cfg: float,
        n_steps: int,
        eos_threshold: float,
        eos_countdown: int = DEFAULTS["eos_countdown"],
        max_seconds: float,
        seed: int | None,
    ) -> tuple[torch.Tensor | None, bool]:
        """(waveform or None when the model emitted EOS at once, hit_max_length)."""
        tokens = torch.tensor(self.tokenize(text), dtype=torch.long)
        max_frames = int(max_seconds * self.mimi.frame_rate)
        if seed is not None:
            torch.manual_seed(seed)
        latents = self.model.generate(
            [tokens],
            [voice],
            texts=[fold_characters(text)],  # what was tokenized; semantic conditioning only
            max_frames=max_frames,
            temp=temp,
            n_steps=n_steps,
            cfg_coef=cfg,
            eos_threshold=eos_threshold,
            eos_countdown=eos_countdown,
        )[0]
        if latents.shape[0] < MIN_FRAMES:
            return None, False
        return self._decode(latents).float().cpu(), latents.shape[0] >= max_frames

    def render_voice_file(
        self, text: str, voice: str, seed: int, settings: dict | None = None
    ) -> tuple[bytes | None, float, bool]:
        """WAV bytes (None on immediate EOS), audio seconds and hit-max for a named voice file,
        with the given sampling settings (or defaults). Takes the generation lock per call."""
        cfg_vals = {**DEFAULTS, **(settings or {})}
        path = self.list_voices()[voice]
        key = ("file", voice, path.stat().st_mtime_ns)
        with self.lock:
            latents = self.voice_latents(key, path, float(cfg_vals["prompt_seconds"]))
            waveform, hit_max = self.synthesize(
                text,
                latents,
                temp=float(cfg_vals["temp"]),
                cfg=float(cfg_vals["cfg"]),
                n_steps=int(cfg_vals["n_steps"]),
                eos_threshold=float(cfg_vals["eos_threshold"]),
                eos_countdown=int(cfg_vals["eos_countdown"]),
                max_seconds=float(cfg_vals["max_seconds"]),
                seed=seed,
            )
        if waveform is None:
            return None, 0.0, False
        return wav_bytes(waveform, self.sample_rate), waveform.shape[0] / self.sample_rate, hit_max


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1 << 22):
            digest.update(chunk)
    return digest.hexdigest()


_RUN_NAME_RE = re.compile(r"\b(finetune_[A-Za-z0-9_.-]+|hebrew[-_][A-Za-z0-9_.-]+)\b")
_CFG_RUN_RE = re.compile(r"^\s*#\s*run:\s*(\S+)", re.MULTILINE)
_CFG_PAREN_RUN_RE = re.compile(r"\(run\s+([A-Za-z0-9_.-]+)")
_CFG_RUN_CONFIG_RE = re.compile(r"^\s*#\s*run_config:\s*(\S+)", re.MULTILINE)
_CFG_PATH_RE = re.compile(r"\b(training/configs/[A-Za-z0-9_./-]+\.yaml)\b")


def describe_stored_model(
    store_root: Path | None, sha_prefix: str, fallback: dict | None = None
) -> dict[str, str | None]:
    """Run name, note and publish date of the published model whose weights hash starts with
    `sha_prefix` (the validation cache folders only carry 16 characters), read from its store
    manifest. The run is the manifest's `run`, else a run name found in its note. `fallback` is
    used when the model is not in the store (a local run without one)."""
    manifest: dict = {}
    if store_root is not None and len(sha_prefix) >= 8:
        try:
            for folder in store_root.iterdir():
                if (
                    folder.name.startswith(sha_prefix.lower())
                    and (folder / "manifest.json").is_file()
                ):
                    manifest = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
                    break
        except (OSError, ValueError):
            manifest = {}
    if not manifest:
        return dict(fallback or {})
    note = (manifest.get("note") or "").strip() or None
    run = (manifest.get("run") or "").strip() or None
    if not run and note:
        m = _RUN_NAME_RE.search(note)
        run = m.group(1) if m else None
    return {"run": run, "note": note, "created": (manifest.get("created") or "").strip() or None}


def extract_model_metadata(
    config: Path | None, store: ModelStore | None, model_sha256: str | None = None
) -> dict[str, str | None]:
    """Identify the currently served model: its weights sha256, output run folder name,
    training/inference config name, and store note/timestamp when available."""
    manifest = store.manifest if store is not None else {}
    sha = model_sha256 or (store.sha256 if store is not None else None)
    note = (manifest.get("note") or "").strip() or None
    created = (manifest.get("created") or "").strip() or None
    run = (manifest.get("run") or "").strip() or None
    run_config = (manifest.get("run_config") or "").strip() or None

    cfg_text = ""
    if config is not None and config.is_file():
        try:
            cfg_text = config.read_text(encoding="utf-8")
        except OSError:
            cfg_text = ""

    # 1. Explicit `# run:` / `# run_config:` headers in the config YAML, or run name in store note.
    if not run and note:
        m = _RUN_NAME_RE.search(note)
        if m:
            run = m.group(1)
    if not run and cfg_text:
        m = _CFG_RUN_RE.search(cfg_text) or _CFG_PAREN_RUN_RE.search(cfg_text)
        if m:
            run = m.group(1).rstrip(",;)")
    if not run_config and cfg_text:
        m = _CFG_RUN_CONFIG_RE.search(cfg_text) or _CFG_PATH_RE.search(cfg_text)
        if m:
            run_config = m.group(1).rstrip(",;)")

    # 2. If weights_path points inside a run directory with args.yaml, read run_dir from it.
    if (not run or not run_config) and cfg_text and config is not None:
        try:
            parsed = yaml.safe_load(cfg_text) or {}
            wp = parsed.get("weights_path")
            if wp:
                wpath = Path(wp) if Path(wp).is_absolute() else config.parent.parent / wp
                args_yaml = wpath.parent / "args.yaml"
                if args_yaml.is_file():
                    args_data = yaml.safe_load(args_yaml.read_text(encoding="utf-8")) or {}
                    run_dir = args_data.get("run_dir")
                    if not run and run_dir:
                        run = Path(str(run_dir)).name
                if not run and wpath.parent.name not in ("", "artifacts", "."):
                    run = wpath.parent.name
        except (OSError, yaml.YAMLError, TypeError, AttributeError) as exc:
            logger.debug("could not inspect weights_path metadata: %s", exc)

    if not run:
        run = note or (config.stem if config is not None else None)
    if not run_config:
        if run and run.startswith("finetune_"):
            run_config = f"training/configs/{run}.yaml"
        elif config is not None:
            run_config = config.name

    return {
        "model_sha256": sha,
        "model_run": run,
        "model_config": run_config,
        "model_note": note,
        "model_created": created,
    }


class EngineLoader:
    """Loads the Engine in a background thread so the HTTP port can open immediately.

    Until `engine` is set, `/tts` answers 503 "warming up..." and `/health` reports it.
    `model_sha256` is the weights hash. With a model store the weights are copied from it
    (verified against the manifest, and the store is pinned to this hash); without one the
    weights file the config points at is hashed.
    """

    def __init__(
        self,
        config: Path,
        voices_dir: Path,
        device: torch.device,
        store: ModelStore | None = None,
        encoder_store: EncoderStore | EncoderStoreSet | None = None,
    ):
        self.config = config
        self.voices_dir = voices_dir
        self.device = device
        self.store = store
        self.encoder_store = encoder_store
        self.engine: Engine | None = None
        self.model_sha256: str | None = None  # set once the model is loaded
        self.stage = "starting"
        self.error: str | None = None

    def start(self) -> None:
        threading.Thread(target=self._load, name="engine-loader", daemon=True).start()

    def semantic_encoder_info(self) -> dict | None:
        """The frozen encoder this model runs with, None for a model without semantic
        conditioning. With an encoder store it includes the hash of the copied file."""
        if self.encoder_store is not None:
            return self.encoder_store.info()
        if self.engine is None:
            return None
        fusion = self.engine.model.flow_lm.semantic_fusion
        if fusion is None:
            return None
        configs = getattr(fusion, "configs", None) or [fusion.config]
        infos = [
            {
                "encoder": c.encoder,
                "model_name": c.model_name,
                "revision": c.revision,
                "sha256": None,
            }
            for c in configs
        ]
        return infos[0] if len(infos) == 1 else {"encoders": infos}

    def cache_key(self) -> str | None:
        """Names the validation cache: model weights + device kind (CPU and GPU audio differ)."""
        if self.engine is None or self.model_sha256 is None:
            return None
        return f"{self.model_sha256[:16]}-{self.device.type}"

    def model_metadata(self) -> dict[str, str | None]:
        return extract_model_metadata(self.config, self.store, self.model_sha256)

    def _load(self) -> None:
        try:
            started = time.perf_counter()
            if self.store is not None:
                self.stage = "copying and verifying the weights"
                self.store.fetch_weights()
                sha = self.store.sha256
                logger.info(
                    "weights %s copied and verified in %.1fs", sha, time.perf_counter() - started
                )
            else:
                self.stage = "hashing the weights"
                sha = file_sha256(Path(yaml.safe_load(self.config.read_text())["weights_path"]))
            encoder_dir = None
            if self.encoder_store is not None:
                self.stage = "copying and verifying the semantic encoder"
                started = time.perf_counter()
                encoder_dir = self.encoder_store.fetch()
                logger.info(
                    "semantic encoder %s copied and verified in %.1fs",
                    self.encoder_store.sha256,
                    time.perf_counter() - started,
                )
            self.stage = "loading the model"
            engine = Engine(self.config, self.voices_dir, self.device, encoder_dir)
            if self.encoder_store is not None:
                self.encoder_store.discard()  # the encoder is in memory now
            self.model_sha256 = sha.lower()
            self.engine = engine
            self.stage = "ready"
            logger.info("model sha256 %s", self.model_sha256)
        except Exception as exc:
            self.stage = "failed"
            logger.exception("model failed to load")
            self.error = f"{type(exc).__name__}: {exc}"


def wav_bytes(samples: torch.Tensor, sample_rate: int) -> bytes:
    buffer = io.BytesIO()
    audio = samples.clamp(-1.0, 1.0).numpy()
    soundfile.write(buffer, audio, sample_rate, format="WAV", subtype="PCM_16")
    return buffer.getvalue()


def make_validation_set(loader: EngineLoader, sentences: Path, cache_dir: Path) -> ValidationSet:
    def describe(sha_prefix: str) -> dict:
        store = getattr(loader, "store", None)
        fallback = None
        if (loader.model_sha256 or "").startswith(sha_prefix):
            meta = loader.model_metadata()
            fallback = {"run": meta["model_run"], "note": meta["model_note"]}
        return describe_stored_model(
            store.dir.parent if store is not None else None, sha_prefix, fallback
        )

    def generate(
        text: str, voice: str, seed: int, settings: dict | None = None
    ) -> tuple[bytes | None, float, bool]:
        engine = loader.engine
        if engine is None:
            raise RuntimeError("model not loaded")
        return engine.render_voice_file(text, voice, seed, settings)

    val_settings = {k: v for k, v in DEFAULTS.items() if k != "eos_countdown"}
    return ValidationSet(
        sentences,
        cache_dir,
        val_settings,
        voices=lambda: list(list_voice_files(loader.voices_dir)),
        generate=generate,
        cache_key=loader.cache_key,
        describe=describe,
    )


def require_engine(loader: EngineLoader) -> Engine:
    engine = loader.engine
    if engine is None:
        if loader.error:
            raise HTTPException(status_code=500, detail=f"model failed to load: {loader.error}")
        raise HTTPException(status_code=503, detail="warming up...", headers={"Retry-After": "10"})
    return engine


def resolve_voice(
    engine: Engine, voice: str | None, voice_wav: UploadFile | None
) -> tuple[tuple, Path, str | None]:
    """(cache key, audio path, temp file to delete when done or None) for a request's voice."""
    if voice_wav is not None and voice_wav.filename:
        data = voice_wav.file.read()
        suffix = Path(voice_wav.filename).suffix.lower() or ".wav"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as handle:
            handle.write(data)
        return ("upload", hashlib.sha256(data).hexdigest()), Path(handle.name), handle.name
    if voice:
        known = engine.list_voices()
        if voice not in known:
            raise HTTPException(status_code=400, detail=f"unknown voice {voice!r}; see GET /voices")
        path = known[voice]
        return ("file", voice, path.stat().st_mtime_ns), path, None
    raise HTTPException(status_code=400, detail="give `voice` or upload `voice_wav`")


def create_app(
    loader: EngineLoader, validation: ValidationSet, downloads: Downloads, milim: MilimBench
) -> FastAPI:
    app = FastAPI(title="Hebrew Pocket TTS (test API)")

    @app.get("/milim")
    def milim_samples():
        if not milim.available:
            raise HTTPException(status_code=404, detail="MILIM-Bench is not installed")
        return milim.payload()

    @app.post("/milim/generate")
    def milim_generate(
        ids: list[int] = Form(..., description="MILIM sample ids from GET /milim"),  # noqa: B008
        voice: str | None = Form(None, description="name of a file from GET /voices"),
        voice_wav: UploadFile | None = File(None, description="or upload a prompt (wins)"),  # noqa: B008
        temp: float = Form(DEFAULTS["temp"], ge=0.0, le=2.0),
        cfg: float = Form(DEFAULTS["cfg"], ge=0.0, le=6.0),
        n_steps: int = Form(DEFAULTS["n_steps"], ge=1, le=8),
        eos_threshold: float = Form(DEFAULTS["eos_threshold"], ge=-12.0, le=12.0),
        eos_countdown: int = Form(DEFAULTS["eos_countdown"], ge=1, le=25),
        max_seconds: float = Form(DEFAULTS["max_seconds"], gt=0.0, le=60.0),
        prompt_seconds: float = Form(DEFAULTS["prompt_seconds"], ge=0.0, le=30.0),
        seed: int | None = Form(None),
    ):
        """Generates up to 5 MILIM samples one after another and streams one JSON line per
        sample (NDJSON) as each finishes, with the WAV base64-encoded in `wav`."""
        if not milim.available:
            raise HTTPException(status_code=404, detail="MILIM-Bench is not installed")
        if not 1 <= len(ids) <= MAX_BATCH:
            raise HTTPException(status_code=400, detail=f"give 1 to {MAX_BATCH} sample ids")
        samples = [milim.get(i) for i in ids]
        if any(s is None for s in samples):
            raise HTTPException(status_code=400, detail="unknown MILIM sample id")
        engine = require_engine(loader)

        key, path, temp_path = resolve_voice(engine, voice, voice_wav)
        try:
            with engine.lock:
                latents = engine.voice_latents(key, path, prompt_seconds)
        finally:
            if temp_path is not None:
                os.unlink(temp_path)

        def stream():
            for sample in samples:
                line = {"id": sample["id"], "text": sample["text"]}
                started = time.perf_counter()
                try:
                    with engine.lock:
                        waveform, hit_max = engine.synthesize(
                            sample["text"],
                            latents,
                            temp=temp,
                            cfg=cfg,
                            n_steps=n_steps,
                            eos_threshold=eos_threshold,
                            eos_countdown=eos_countdown,
                            max_seconds=max_seconds,
                            seed=seed,
                        )
                    elapsed = time.perf_counter() - started
                    if waveform is None:
                        line.update(status="eos", generation_seconds=round(elapsed, 2))
                    else:
                        line.update(
                            status="ok",
                            audio_seconds=round(waveform.shape[0] / engine.sample_rate, 2),
                            generation_seconds=round(elapsed, 2),
                            hit_max_length=hit_max,
                            wav=base64.b64encode(wav_bytes(waveform, engine.sample_rate)).decode(),
                        )
                except Exception as exc:
                    logger.exception("MILIM sample %s failed", sample["id"])
                    line.update(status="error", error=f"{type(exc).__name__}: {exc}")
                yield json.dumps(line, ensure_ascii=False) + "\n"

        return StreamingResponse(stream(), media_type="application/x-ndjson")

    @app.get("/download/{name}")
    def download(name: str):
        # On Cloud Run `model.safetensors` redirects (302) to a 30-minute signed GCS URL so the
        # 1.4 GB weights download straight from the bucket; small files stream from disk, and
        # the guide and code zip are generated.
        if name in (GUIDE, CODE_ZIP):
            guide = downloads.guide()
            if guide is None:
                raise HTTPException(
                    status_code=503, detail="warming up...", headers={"Retry-After": "10"}
                )
            if name == GUIDE:
                return Response(
                    guide,
                    media_type="text/markdown; charset=utf-8",
                    headers={"Content-Disposition": f'attachment; filename="{GUIDE}"'},
                )
            return Response(
                downloads.code_zip(guide),
                media_type="application/zip",
                headers={"Content-Disposition": f'attachment; filename="{CODE_ZIP}"'},
            )
        signed = downloads.signed_url(name)
        if signed is not None:
            return RedirectResponse(signed, status_code=302)
        found = downloads.file(name)
        if found is None:
            raise HTTPException(status_code=404, detail="no such download")
        return FileResponse(found.path, filename=found.name)

    @app.get("/validation")
    def validation_status():
        return validation.status()

    @app.post("/validation/start")
    def validation_start(
        force: bool = Form(False, description="discard the cached set and render it again"),
        voice: str | None = Form(
            None, description="name of a file from GET /voices (used for initial or forced run)"
        ),
        temp: float | None = Form(None, ge=0.0, le=2.0),
        cfg: float | None = Form(None, ge=0.0, le=6.0),
        n_steps: int | None = Form(None, ge=1, le=8),
        eos_threshold: float | None = Form(None, ge=-12.0, le=12.0),
        eos_countdown: int | None = Form(None, ge=1, le=25),
        max_seconds: float | None = Form(None, gt=0.0, le=60.0),
        prompt_seconds: float | None = Form(None, ge=0.0, le=30.0),
        seed: int | None = Form(None),
    ):
        """Show the validation set of the loaded model, generating it with the selected `voice`
        and sampling settings if it is not cached. With `force=true` the cached set is deleted
        and rendered again with `voice` and the current sampling settings."""
        if loader.engine is None:
            raise HTTPException(
                status_code=503, detail="warming up...", headers={"Retry-After": "10"}
            )
        raw_settings = {
            "temp": temp,
            "cfg": cfg,
            "n_steps": n_steps,
            "eos_threshold": eos_threshold,
            "eos_countdown": eos_countdown,
            "max_seconds": max_seconds,
            "prompt_seconds": prompt_seconds,
        }
        overrides = {k: v for k, v in raw_settings.items() if v is not None}
        settings = {**DEFAULTS, **overrides} if overrides else None
        try:
            return validation.start(force=force, voice=voice, settings=settings, seed=seed)
        except ValidationBusy as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.get("/validation/audio/{item_id}.wav")
    def validation_audio(item_id: str):
        path = validation.audio_path(item_id)
        if path is None or not path.is_file():
            raise HTTPException(status_code=404, detail="no such validation audio")
        return FileResponse(path, media_type="audio/wav")

    @app.get("/validation/runs")
    def validation_runs():
        """Cached sets of other models for the same sentences (the page's compare list), plus
        the sentences, so the table can be laid out before any run is chosen."""
        return {"sentences": validation.sentence_rows(), "runs": validation.past_runs()}

    @app.get("/validation/runs/{run_id}")
    def validation_run(run_id: str):
        run = validation.run_detail(run_id)
        if run is None:
            raise HTTPException(status_code=404, detail="no such validation run")
        return run

    @app.get("/validation/runs/{run_id}/audio/{item_id}.wav")
    def validation_run_audio(run_id: str, item_id: str):
        path = validation.run_audio_path(run_id, item_id)
        if path is None or not path.is_file():
            raise HTTPException(status_code=404, detail="no such validation audio")
        return FileResponse(path, media_type="audio/wav")

    def _meta() -> dict[str, str | None]:
        if hasattr(loader, "model_metadata"):
            return loader.model_metadata()
        return extract_model_metadata(
            getattr(loader, "config", None),
            getattr(loader, "store", None),
            getattr(loader, "model_sha256", None),
        )

    @app.get("/ready")
    def ready():
        """200 only once the model is loaded: the Cloud Run startup probe, so a revision whose
        model cannot be loaded never takes traffic."""
        meta = _meta()
        if loader.engine is None:
            return JSONResponse(
                status_code=503,
                content={"ready": False, "stage": loader.stage, "error": loader.error, **meta},
            )
        return {"ready": True, **meta, "semantic_encoder": loader.semantic_encoder_info()}

    @app.get("/health")
    def health():
        engine = loader.engine
        voices = list_voice_files(loader.voices_dir)
        meta = _meta()
        if engine is None:
            if loader.error:
                return JSONResponse(
                    status_code=500,
                    content={"status": "load failed", "error": loader.error, **meta},
                )
            return {
                "status": "warming up...",
                "ready": False,
                "stage": loader.stage,
                "voices": len(voices),
                **meta,
            }
        return {
            "status": "ok",
            "ready": True,
            **meta,
            "model_store": str(loader.store.dir) if loader.store else None,
            "semantic_encoder": loader.semantic_encoder_info(),
            "device": str(engine.device),
            "layers": engine.n_layers,
            "sample_rate": engine.sample_rate,
            "config": str(engine.config),
            "voices": len(voices),
            "defaults": DEFAULTS,
        }

    @app.get("/voices")
    def voices():
        return {"voices": list(list_voice_files(loader.voices_dir))}

    @app.get("/voices/{name}")
    def voice_audio(name: str):
        known = list_voice_files(loader.voices_dir)
        if name not in known:
            raise HTTPException(status_code=404, detail="no such voice")
        return FileResponse(known[name])

    @app.get("/", response_class=HTMLResponse)
    def index():
        meta = _meta()
        return (
            INDEX_HTML.replace("__DEFAULTS__", repr(DEFAULTS).replace("'", '"'))
            .replace("__MODEL_META__", json.dumps(meta, ensure_ascii=False))
            .replace("__DOWNLOADS__", downloads.html())
        )

    @app.post("/tts")
    def tts(
        text: str = Form(...),
        voice: str | None = Form(None, description="name of a file from GET /voices"),
        voice_wav: UploadFile | None = File(None, description="or upload a prompt (wins)"),  # noqa: B008
        temp: float = Form(DEFAULTS["temp"], ge=0.0, le=2.0),
        cfg: float = Form(DEFAULTS["cfg"], ge=0.0, le=6.0),
        n_steps: int = Form(DEFAULTS["n_steps"], ge=1, le=8),
        eos_threshold: float = Form(DEFAULTS["eos_threshold"], ge=-12.0, le=12.0),
        eos_countdown: int = Form(DEFAULTS["eos_countdown"], ge=1, le=25),
        max_seconds: float = Form(DEFAULTS["max_seconds"], gt=0.0, le=60.0),
        prompt_seconds: float = Form(DEFAULTS["prompt_seconds"], ge=0.0, le=30.0),
        seed: int | None = Form(None),
    ):
        engine = require_engine(loader)
        text = text.strip()
        if not text:
            raise HTTPException(status_code=400, detail="text is empty")
        if len(text) > MAX_TEXT_CHARS:
            raise HTTPException(status_code=400, detail=f"text is over {MAX_TEXT_CHARS} characters")

        started = time.perf_counter()
        temp_path: str | None = None
        try:
            key, path, temp_path = resolve_voice(engine, voice, voice_wav)
            with engine.lock:
                latents = engine.voice_latents(key, path, prompt_seconds)
                waveform, hit_max = engine.synthesize(
                    text,
                    latents,
                    temp=temp,
                    cfg=cfg,
                    n_steps=n_steps,
                    eos_threshold=eos_threshold,
                    eos_countdown=eos_countdown,
                    max_seconds=max_seconds,
                    seed=seed,
                )
        finally:
            if temp_path is not None:
                os.unlink(temp_path)

        if waveform is None:
            raise HTTPException(
                status_code=422,
                detail="the model ended speech immediately; raise eos_threshold or try another "
                "voice prompt",
            )
        elapsed = time.perf_counter() - started
        seconds = waveform.shape[0] / engine.sample_rate
        headers = {
            "X-Audio-Seconds": f"{seconds:.2f}",
            "X-Generation-Seconds": f"{elapsed:.2f}",
            "X-Realtime-Factor": f"{seconds / elapsed:.2f}",
            "X-Hit-Max-Length": str(hit_max).lower(),
            "X-Device": str(engine.device),
            "X-Model-Sha256": loader.model_sha256 or "",
            "Content-Disposition": 'inline; filename="speech.wav"',
        }
        logger.info(
            "%.1fs audio in %.1fs (%.2fx) cfg=%s temp=%s eos=%s countdown=%s hit_max=%s chars=%d",
            seconds,
            elapsed,
            seconds / elapsed,
            cfg,
            temp,
            eos_threshold,
            eos_countdown,
            hit_max,
            len(text),
        )
        return Response(
            wav_bytes(waveform, engine.sample_rate), media_type="audio/wav", headers=headers
        )

    return app


INDEX_HTML = """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><title>Hebrew TTS test</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>
 body{font:15px system-ui,sans-serif;max-width:760px;margin:2rem auto;padding:0 1rem;color:#222;background:#fff}
 textarea{width:100%;height:7rem;font-size:1.2rem;direction:rtl;box-sizing:border-box}
 label{display:block;margin:.6rem 0 .2rem;font-weight:600}
 .grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(170px,1fr));gap:.4rem .8rem}
 .grid label{font-weight:400;margin:0} .grid input{width:100%;box-sizing:border-box}
 button{margin:1rem 0;padding:.6rem 1.4rem;font-size:1rem} audio{width:100%}
 #status{color:#555;white-space:pre-wrap} #status.err{color:#b00020}
 .dl{background:#f4f6f8;border-radius:6px;padding:.6rem 1rem;margin-bottom:1rem}
 .dl ul{margin:.4rem 0 0;padding-left:1.2rem} .dl li{margin:.2rem 0} .meta{color:#777;font-size:.85rem}
 .mcard{border:1px solid #d0d7de;border-radius:6px;padding:.6rem .9rem;margin-bottom:.85rem;background:#f8fafc;font-size:.9rem;line-height:1.5}
 .mcard code{background:#e8eef5;padding:.08rem .35rem;border-radius:4px;font-size:.84rem;word-break:break-all}
 .vbox{border:1px solid #ddd;border-radius:6px;padding:.55rem .75rem;margin-top:.5rem;background:#fafbfc}
 .vhead{display:flex;gap:.6rem;align-items:center;flex-wrap:wrap;margin-bottom:.35rem}
 .vhead button{margin:0;padding:.25rem .7rem;font-size:.85rem}
 #vwave{width:100%;height:76px;display:block;border-radius:4px;background:#fff;border:1px solid #e3e6ea;cursor:pointer}
 #vplayer{height:32px;margin-top:.35rem}
 .shelp{margin-top:.6rem;padding:.55rem .75rem;background:#f4f6f8;border-radius:6px;font-size:.86rem;line-height:1.45;color:#333}
 .shelp code{background:#e6eaef;padding:.05rem .3rem;border-radius:3px;font-size:.82rem}
 #vstatus{color:#555} #vstatus.err{color:#b00020}
 .vcat{margin:1.2rem 0 .3rem;font-weight:700;text-transform:capitalize}
 .vrow{border:1px solid #ddd;border-radius:6px;padding:.5rem .7rem;margin:.4rem 0}
 .vtext{direction:rtl;font-size:1.2rem;margin-bottom:.3rem}
 .vmeta{color:#777;font-size:.85rem} .vrow.bad{border-color:#b00020}
 .vcmp{width:min(96vw,1500px);position:relative;left:50%;transform:translateX(-50%);margin:.6rem 0;box-sizing:border-box}
 .vcmp .pick{display:grid;grid-template-columns:repeat(auto-fill,minmax(260px,1fr));gap:.4rem .8rem;margin:.4rem 0}
 .vcmp .pick label{display:flex;gap:.45rem;align-items:flex-start;margin:0;font-weight:400;border:1px solid #ddd;border-radius:6px;padding:.35rem .55rem;background:#fff;cursor:pointer}
 .vcmp .pick label.on{border-color:#1a73e8;background:#eef4ff}
 .vcmp .pick label>div{min-width:0;overflow-wrap:anywhere}
 .vcmp .pick .nm{font-weight:600} .vcmp .pick .sub{color:#777;font-size:.8rem;line-height:1.35}
 .vcmp .scroll{overflow:auto;max-height:80vh;border:1px solid #ddd;border-radius:6px;background:#fff}
 .vcmp table{border-collapse:separate;border-spacing:0;min-width:100%}
 .vcmp th,.vcmp td{padding:.4rem .55rem;border-bottom:1px solid #eee;border-right:1px solid #eee;vertical-align:top;text-align:left;min-width:230px;max-width:300px;background:#fff}
 .vcmp thead th{position:sticky;top:0;z-index:3;background:#f4f6f8}
 .vcmp th.txt,.vcmp td.txt{position:sticky;left:0;z-index:2;min-width:220px;max-width:280px;background:#f8fafc}
 .vcmp thead th.txt{z-index:4}
 .vcmp td.txt .vtext{font-size:1.05rem;margin-bottom:.2rem}
 .vcmp tr.cat td{background:#eef1f4;font-weight:700;text-transform:capitalize;min-width:0;max-width:none}
 .vcmp td.bad{color:#b00020;font-size:.85rem}
 .vcmp .cur{color:#1a73e8;font-size:.78rem;font-weight:700}
 .vcmp .warn{color:#a15c00;font-size:.78rem}
 .row{display:flex;gap:.6rem;align-items:center;flex-wrap:wrap}
 #milimDlg{width:min(720px,94vw);max-height:88vh;border:1px solid #bbb;border-radius:8px;padding:0;color:#222;background:#fff}
 #milimDlg::backdrop{background:rgba(0,0,0,.4)}
 #milimDlg form{display:flex;flex-direction:column;max-height:88vh;margin:0}
 .dhead,.dfoot{padding:.8rem 1rem;display:flex;gap:.6rem;align-items:center;flex-wrap:wrap}
 .dhead{border-bottom:1px solid #ddd} .dfoot{border-top:1px solid #ddd}
 .dhead select{font-size:1rem;padding:.3rem} .dhead .grow,.dfoot .grow{flex:1}
 #mlist{overflow-y:auto;padding:.4rem 1rem;min-height:12rem;flex:1}
 .mrow{display:flex;gap:.7rem;align-items:flex-start;border:1px solid #ddd;border-radius:6px;padding:.45rem .7rem;margin:.35rem 0;cursor:pointer}
 .mrow.sel{border-color:#1a73e8;background:#eef4ff} .mrow.off{opacity:.45;cursor:not-allowed}
 .mtext{font-size:1.2rem;line-height:1.9} .mcat{color:#777;font-size:.8rem}
 mark.w{background:#ffe08a;color:inherit;border-radius:3px;padding:0 .15em;position:relative}
 mark.w::after{content:attr(data-ipa);position:absolute;top:100%;left:50%;transform:translateX(-50%);
   direction:ltr;white-space:nowrap;background:#222;color:#fff;font:.9rem system-ui,sans-serif;
   padding:.15rem .5rem;border-radius:4px;z-index:5;display:none;pointer-events:none}
 mark.w:hover::after{display:block}
 #mres .vmeta{direction:ltr;text-align:right}
</style></head><body>
<h2>Hebrew TTS test</h2>
<div class="mcard" id="modelCard">
 <div><b>Model run:</b> <code id="mRun">…</code> &nbsp; <b>Run config:</b> <code id="mCfg">…</code></div>
 <div><b>Weights SHA-256:</b> <code id="mSha">…</code></div>
 <div id="mExtra" class="meta"></div>
</div>
<div class="dl"><b>Download for local inference</b>__DOWNLOADS__</div>
<label for="text">Text</label>
<textarea id="text" dir="rtl">שלום, זהו מבחן קצר של המודל.</textarea>
<label for="voice">Voice prompt</label>
<select id="voice"></select>
<label for="file">…or upload one (wins over the list; first <span id="ps"></span>s are used)</label>
<input type="file" id="file" accept="audio/*">
<div class="vbox" id="vbox">
 <div class="vhead">
  <span id="vinfo" class="meta">loading voice prompt…</span>
  <span style="flex:1"></span>
  <button type="button" id="vplaySlice" title="Play only the active prompt slice (0 to prompt seconds)">▶ Play used slice</button>
 </div>
 <canvas id="vwave" height="76" title="Click to seek in voice prompt"></canvas>
 <audio id="vplayer" controls preload="metadata"></audio>
</div>
<details><summary style="margin-top:1rem;cursor:pointer">Sampling settings</summary>
 <div class="grid" id="params"></div>
 <div class="shelp">
  <b>How <code>eos threshold</code> and <code>eos countdown</code> control the ending:</b>
  Audio is generated in <b>80 ms frames</b> (12.5 Hz). At each frame <i>t</i>, the model outputs an EOS logit (amplified by <code>cfg</code>).
  EOS triggers on the first frame <i>t</i> where <code>eos_logit &gt; eos_threshold</code> (higher threshold = needs stronger confidence = stops later).
  Once triggered at frame <i>t</i>, <code>eos_countdown</code> ticks down by 1 each frame (including frame <i>t</i> itself) and stops when it hits 0, dropping the frame where it hits 0:
  <b><code>1</code></b> stops immediately before frame <i>t</i> (0 extra frames; can clip the final consonant if EOS fires early), while
  <b><code>2</code>–<code>3</code></b> keeps 1–2 extra frames (80–160 ms) from frame <i>t</i> onward so the final word releases naturally into silence.
 </div>
</details>
<div class="row">
<button id="go">Generate</button>
<button id="milimBtn" dir="rtl" hidden>בחר דוגמה מ MILIM</button>
</div>
<div id="status"></div>
<audio id="player" controls></audio>
<div id="mres" dir="rtl"></div>
<dialog id="milimDlg" dir="rtl"><form method="dialog">
 <div class="dhead">
  <b>MILIM-Bench</b>
  <select id="mcat"></select>
  <span class="grow"></span>
  <span id="mcount"></span>
 </div>
 <div id="mlist"></div>
 <div class="dfoot">
  <button type="button" id="mclear" style="margin:0">נקה בחירה</button>
  <span class="grow"></span>
  <button value="cancel" style="margin:0">סגור</button>
  <button type="button" id="mgo" style="margin:0" disabled>צור</button>
 </div>
</form></dialog>
<hr style="margin:1.5rem 0">
<button id="val" dir="rtl" disabled>הצג סט ולידציה</button>
<button id="valRedo" dir="rtl" disabled title="deletes the cached set of this model and renders it again with the voice selected above">צור מחדש עם הקול הנבחר</button>
<button id="valCmp" dir="rtl" title="put the validation audio of earlier models next to this one's, sample by sample">השווה לריצות קודמות</button>
<div id="vstatus"></div>
<div id="vcmp" class="vcmp vbox" hidden>
 <div class="vhead"><b>Compare with past runs</b><span class="meta" id="vcmpInfo"></span></div>
 <div class="pick" id="vcmpPick"></div>
 <div class="scroll" id="vcmpScroll" hidden><table id="vcmpTable"></table></div>
</div>
<div id="vlist"></div>
<script>
const D = __DEFAULTS__;
let M_META = __MODEL_META__;
const renderModelMeta = (m) => {
  if (!m) return;
  M_META = {...M_META, ...m};
  document.getElementById("mRun").textContent = M_META.model_run || "(unknown run)";
  document.getElementById("mCfg").textContent = M_META.model_config || "hebrew_24l.yaml";
  document.getElementById("mSha").textContent = M_META.model_sha256 || "computing…";
  const parts = [];
  if (M_META.layers) parts.push(`${M_META.layers} layers`);
  if (M_META.device) parts.push(`device: ${M_META.device}`);
  const se = M_META.semantic_encoder;
  if (se) {
    const encs = se.encoders ? se.encoders.map(e => e.encoder).join(" + ") : se.encoder;
    if (encs) parts.push(`semantic: ${encs}`);
  }
  if (M_META.model_note && M_META.model_note !== M_META.model_run) parts.push(`note: ${M_META.model_note}`);
  if (M_META.model_created) parts.push(`published: ${M_META.model_created}`);
  document.getElementById("mExtra").textContent = parts.join(" · ");
};
renderModelMeta(M_META);
const names = {temp:"temperature", cfg:"cfg (guidance)", n_steps:"decode steps",
  eos_threshold:"eos threshold", eos_countdown:"eos countdown", max_seconds:"max seconds",
  prompt_seconds:"prompt seconds", seed:"seed"};
const intKeys = new Set(["n_steps", "eos_countdown", "seed"]);
const box = document.getElementById("params");
for (const k of [...Object.keys(D), "seed"]) {
  const l = document.createElement("label");
  l.textContent = names[k] + " ";
  const i = document.createElement("input");
  i.id = "p_" + k; i.type = "number"; i.step = intKeys.has(k) ? "1" : "any"; i.value = D[k] ?? "";
  if (k === "n_steps" || k === "eos_countdown") i.min = "1";
  if (k === "seed") i.placeholder = "random";
  l.appendChild(i); box.appendChild(l);
}
document.getElementById("ps").textContent = D.prompt_seconds;

// Browser-side voice prompt waveform + cutoff marker + playback.
const vSel = document.getElementById("voice");
const fInp = document.getElementById("file");
const psInp = document.getElementById("p_prompt_seconds");
const vInfo = document.getElementById("vinfo");
const vWave = document.getElementById("vwave");
const vPlayer = document.getElementById("vplayer");
const vPlaySlice = document.getElementById("vplaySlice");
let vPeaks = null, vDur = 0, vName = "", vObjUrl = null, sliceTimer = null, audioCtx = null;

const getPromptSec = () => {
  const v = parseFloat(psInp.value);
  return Number.isFinite(v) && v >= 0 ? v : D.prompt_seconds;
};
const drawWave = () => {
  const dpr = window.devicePixelRatio || 1;
  const w = vWave.clientWidth || 680, h = vWave.clientHeight || 76;
  if (vWave.width !== Math.round(w * dpr) || vWave.height !== Math.round(h * dpr)) {
    vWave.width = Math.round(w * dpr); vWave.height = Math.round(h * dpr);
  }
  const ctx = vWave.getContext("2d");
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, w, h);
  if (!vPeaks || !vDur) return;
  const ps = getPromptSec();
  document.getElementById("ps").textContent = ps;
  const cutSec = ps > 0 ? Math.min(ps, vDur) : vDur;
  const cutX = Math.round((cutSec / vDur) * w);
  if (cutX < w) {
    ctx.fillStyle = "#f9ebed";
    ctx.fillRect(cutX, 0, w - cutX, h);
  }
  const mid = h / 2, n = vPeaks.length;
  for (let i = 0; i < n; i++) {
    const x = (i / n) * w;
    const amp = Math.max(1, vPeaks[i] * (mid - 6));
    ctx.fillStyle = x <= cutX ? "#1a73e8" : "#b8bec6";
    ctx.fillRect(x, mid - amp, Math.max(1, w / n - 0.5), amp * 2);
  }
  if (ps > 0 && ps < vDur) {
    ctx.strokeStyle = "#b00020";
    ctx.lineWidth = 1.5;
    ctx.setLineDash([4, 3]);
    ctx.beginPath(); ctx.moveTo(cutX + 0.5, 0); ctx.lineTo(cutX + 0.5, h); ctx.stroke();
    ctx.setLineDash([]);
    ctx.fillStyle = "#b00020";
    ctx.font = "600 11px system-ui, sans-serif";
    const lbl = `cut @ ${ps.toFixed(2)}s`;
    const tw = ctx.measureText(lbl).width;
    const lx = cutX + 6 + tw < w ? cutX + 5 : Math.max(4, cutX - tw - 5);
    ctx.fillText(lbl, lx, 13);
  }
  if (vPlayer.currentTime > 0 && vDur > 0) {
    const px = (vPlayer.currentTime / vDur) * w;
    ctx.strokeStyle = "#111";
    ctx.lineWidth = 1.5;
    ctx.beginPath(); ctx.moveTo(px, 0); ctx.lineTo(px, h); ctx.stroke();
  }
  const cutNote = ps <= 0 ? `all ${vDur.toFixed(2)}s used (prompt_seconds=0)` :
    (ps < vDur ? `using first ${ps.toFixed(2)}s of ${vDur.toFixed(2)}s (${(vDur - ps).toFixed(2)}s cut off)` :
    `using all ${vDur.toFixed(2)}s (shorter than ${ps.toFixed(2)}s cutoff)`);
  vInfo.textContent = `${vName} — ${cutNote}`;
};

const loadVoicePrompt = async () => {
  if (sliceTimer) { clearInterval(sliceTimer); sliceTimer = null; }
  const file = fInp.files[0];
  let url = "", label = "";
  if (vObjUrl) { URL.revokeObjectURL(vObjUrl); vObjUrl = null; }
  if (file) {
    vObjUrl = URL.createObjectURL(file);
    url = vObjUrl; label = file.name;
  } else if (vSel.value) {
    url = "/voices/" + encodeURIComponent(vSel.value); label = vSel.value;
  } else {
    vPeaks = null; vDur = 0; vInfo.textContent = "no voice prompt selected";
    vPlayer.removeAttribute("src"); drawWave(); return;
  }
  vName = label;
  vInfo.textContent = `loading ${label}…`;
  vPlayer.src = url;
  try {
    const buf = file ? await file.arrayBuffer() : await (await fetch(url)).arrayBuffer();
    audioCtx = audioCtx || new (window.AudioContext || window.webkitAudioContext)();
    const decoded = await audioCtx.decodeAudioData(buf.slice(0));
    vDur = decoded.duration;
    const ch = decoded.getChannelData(0);
    const bins = 240, step = Math.max(1, Math.floor(ch.length / bins));
    const peaks = new Float32Array(bins);
    let maxPeak = 0.01;
    for (let b = 0; b < bins; b++) {
      let m = 0, start = b * step, end = Math.min(ch.length, start + step);
      for (let j = start; j < end; j++) { const a = Math.abs(ch[j]); if (a > m) m = a; }
      peaks[b] = m; if (m > maxPeak) maxPeak = m;
    }
    for (let b = 0; b < bins; b++) peaks[b] /= maxPeak;
    vPeaks = peaks;
    drawWave();
  } catch (e) {
    vPeaks = null; vDur = 0;
    vInfo.textContent = `${label} — waveform unavailable`;
    drawWave();
  }
};

vWave.onclick = e => {
  if (!vDur) return;
  const rect = vWave.getBoundingClientRect();
  const frac = Math.max(0, Math.min(1, (e.clientX - rect.left) / rect.width));
  vPlayer.currentTime = frac * vDur;
  drawWave();
};
vPlayer.ontimeupdate = drawWave;
vPlayer.onended = () => { if (sliceTimer) { clearInterval(sliceTimer); sliceTimer = null; } drawWave(); };
vPlaySlice.onclick = () => {
  if (!vPlayer.src) return;
  if (sliceTimer) { clearInterval(sliceTimer); sliceTimer = null; }
  const ps = getPromptSec();
  const stopAt = ps > 0 && vDur > 0 ? Math.min(ps, vDur) : vDur;
  vPlayer.currentTime = 0;
  vPlayer.play();
  if (stopAt > 0) {
    sliceTimer = setInterval(() => {
      drawWave();
      if (vPlayer.paused || vPlayer.currentTime >= stopAt) {
        if (vPlayer.currentTime >= stopAt) vPlayer.pause();
        clearInterval(sliceTimer); sliceTimer = null; drawWave();
      }
    }, 30);
  }
};
vSel.onchange = () => { if (!fInp.files[0]) loadVoicePrompt(); };
fInp.onchange = loadVoicePrompt;
psInp.oninput = drawWave;
window.addEventListener("resize", drawWave);

fetch("/voices").then(r => r.json()).then(j => {
  j.voices.forEach(v => vSel.add(new Option(v, v)));
  if (!j.voices.length) vSel.add(new Option("(none - put files in deploy/voices or upload)", ""));
  loadVoicePrompt();
});
const status = document.getElementById("status");
const go = document.getElementById("go");
go.disabled = true;
const valBtn = document.getElementById("val");
const redoBtn = document.getElementById("valRedo");
const vstatus = document.getElementById("vstatus");
const vlist = document.getElementById("vlist");
const shown = new Set();
const el = (tag, cls, text) => {
  const e = document.createElement(tag);
  if (cls) e.className = cls;
  if (text !== undefined) e.textContent = text;
  return e;
};
const renderItems = items => {
  for (const it of items) {
    const sig = it.id + ":" + it.status;
    if (shown.has(sig)) continue;
    shown.add(sig);
    let cat = document.getElementById("vcat_" + it.category);
    if (!cat) {
      cat = el("div", "vcat", it.category.replace(/_/g, " ")); cat.id = "vcat_" + it.category;
      vlist.appendChild(cat);
    }
    let row = document.getElementById("vrow_" + it.id);
    if (!row) {
      row = el("div", "vrow"); row.id = "vrow_" + it.id;
      let after = cat;
      while (after.nextSibling && after.nextSibling.className === "vrow") after = after.nextSibling;
      after.after(row);
    }
    row.replaceChildren();
    row.classList.toggle("bad", it.status !== "ok");
    row.appendChild(el("div", "vtext", it.text));
    if (it.note) row.appendChild(el("div", "vmeta", it.note));
    if (it.status === "ok") {
      const a = el("audio"); a.controls = true; a.preload = "none"; a.style.width = "100%";
      a.src = "/validation/audio/" + it.id + ".wav?t=" + Date.now();  // a regenerated set must not play from the browser cache
      row.appendChild(a);
      row.appendChild(el("div", "vmeta", it.id + " - " + it.audio_seconds + "s audio in " +
        it.generation_seconds + "s" + (it.hit_max_length ? " - hit max length" : "")));
    } else {
      row.appendChild(el("div", "vmeta", it.id + " - " +
        (it.status === "eos" ? "the model ended speech immediately" : it.error)));
    }
  }
};
const appendSamplingParams = f => {
  for (const k of [...Object.keys(D), "seed"]) {
    const v = document.getElementById("p_" + k).value;
    if (v !== "") f.append(k, v);
  }
  return f;
};
const formatValSettings = st => {
  if (!st) return "";
  const parts = [];
  for (const k of ["cfg", "temp", "eos_threshold", "eos_countdown", "n_steps", "prompt_seconds"]) {
    if (st[k] !== undefined) parts.push(`${k}=${st[k]}`);
  }
  return parts.length ? " (" + parts.join(", ") + ")" : "";
};

// -- compare with past runs: one row per sample, one wav column per run -------------------
const cmpBtn = document.getElementById("valCmp");
const cmpBox = document.getElementById("vcmp");
const cmpPick = document.getElementById("vcmpPick");
const cmpInfo = document.getElementById("vcmpInfo");
const cmpScroll = document.getElementById("vcmpScroll");
const cmpTable = document.getElementById("vcmpTable");
let cmpData = null;            // {sentences, runs} from /validation/runs
const cmpPicked = new Set();   // ids of the past runs in the table
const cmpDetail = {};          // id -> run with items, fetched once
let cmpCurrent = null;         // the latest /validation status
let cmpDrawn = "";
const settingsText = st => st ? Object.keys(st).sort().map(k => k + "=" + st[k]).join(", ") : "";
const cmpCurrentCol = () => {
  const s = cmpCurrent || {};
  const run = M_META && M_META.model_run;
  return {
    id: "", current: true, label: run || (s.model_sha256 || "current model"),
    sub: [(s.model_sha256 || ""), s.voice, s.seed !== undefined && s.seed !== null ? "seed " + s.seed : null,
      s.done !== undefined ? s.done + "/" + s.total : null].filter(Boolean).join(" · "),
    voice: s.voice, settings: s.settings, items: s.items || [],
  };
};
const cmpPastCol = r => ({
  id: r.id, current: false, label: r.label,
  sub: [r.model_sha256, r.device, r.voice, r.seed !== undefined && r.seed !== null ? "seed " + r.seed : null,
    r.date, r.state === "partial" ? r.done + "/" + r.total : null].filter(Boolean).join(" · "),
  voice: r.voice, settings: r.settings, note: r.note, run: r, items: (cmpDetail[r.id] || {}).items || [],
});
const renderCmpPick = () => {
  cmpPick.replaceChildren();
  for (const r of cmpData.runs) {
    const lab = el("label", cmpPicked.has(r.id) ? "on" : "");
    const cb = el("input"); cb.type = "checkbox"; cb.checked = cmpPicked.has(r.id);
    cb.onchange = async () => {
      if (cb.checked) {
        cmpPicked.add(r.id);
        if (!cmpDetail[r.id]) {
          cmpInfo.textContent = "loading " + r.label + "…";
          try {
            const resp = await fetch("/validation/runs/" + encodeURIComponent(r.id));
            if (!resp.ok) throw new Error(resp.status + " " + await resp.text());
            cmpDetail[r.id] = await resp.json();
          } catch (e) { cmpPicked.delete(r.id); cb.checked = false; cmpInfo.textContent = "failed to load " + r.label + ": " + e.message; return; }
          cmpInfo.textContent = "";
        }
      } else cmpPicked.delete(r.id);
      lab.classList.toggle("on", cb.checked);
      renderCmpTable(true);
    };
    const box = el("div");
    box.appendChild(el("div", "nm", r.label));
    box.appendChild(el("div", "sub", cmpPastCol(r).sub));
    if (r.note && r.note !== r.label) box.appendChild(el("div", "sub", r.note));
    lab.append(cb, box);
    cmpPick.appendChild(lab);
  }
  if (!cmpData.runs.length) cmpInfo.textContent = "no other cached runs of this validation set";
};
const renderCmpTable = (force) => {
  const cols = [cmpCurrentCol(), ...cmpData.runs.filter(r => cmpPicked.has(r.id)).map(cmpPastCol)];
  const sig = JSON.stringify([cols.map(c => [c.id, c.items.map(i => i.id + i.status).join()]), cmpCurrent && cmpCurrent.voice]);
  if (!force && sig === cmpDrawn) return;   // do not rebuild (and stop playing audio) when nothing changed
  cmpDrawn = sig;
  cmpScroll.hidden = cols.length < 2;
  cmpTable.replaceChildren();
  if (cols.length < 2) return;
  const base = settingsText(cols[0].settings);
  const head = el("tr");
  head.appendChild(el("th", "txt", "sample"));
  for (const c of cols) {
    const th = el("th");
    if (c.current) th.appendChild(el("div", "cur", "CURRENT MODEL"));
    th.appendChild(el("div", "nm", c.label));
    th.appendChild(el("div", "vmeta", c.sub));
    if (c.note && c.note !== c.label) th.appendChild(el("div", "vmeta", c.note));
    if (!c.current && cols[0].voice && c.voice && c.voice !== cols[0].voice)
      th.appendChild(el("div", "warn", "different voice prompt from the current model's"));
    if (!c.current && base && c.settings && settingsText(c.settings) !== base)
      th.appendChild(el("div", "warn", "sampling settings differ: " + settingsText(c.settings)));
    head.appendChild(th);
  }
  const thead = el("thead"); thead.appendChild(head); cmpTable.appendChild(thead);
  const body = el("tbody");
  const byId = cols.map(c => Object.fromEntries(c.items.map(i => [i.id, i])));
  let lastCat = null;
  for (const sn of cmpData.sentences) {
    if (sn.category !== lastCat) {
      lastCat = sn.category;
      const tr = el("tr", "cat"), td = el("td", "", sn.category.replace(/_/g, " "));
      td.colSpan = cols.length + 1; tr.appendChild(td); body.appendChild(tr);
    }
    const tr = el("tr");
    const first = el("td", "txt");
    first.appendChild(el("div", "vtext", sn.text)); first.firstChild.dir = "rtl";
    first.appendChild(el("div", "vmeta", sn.id + (sn.note ? " - " + sn.note : "")));
    tr.appendChild(first);
    cols.forEach((c, idx) => {
      const it = byId[idx][sn.id];
      const td = el("td");
      if (!it) td.appendChild(el("span", "vmeta", "-"));
      else if (it.status === "ok") {
        const a = el("audio"); a.controls = true; a.preload = "none";
        a.src = c.current ? "/validation/audio/" + sn.id + ".wav?t=" + Date.now()
          : "/validation/runs/" + encodeURIComponent(c.id) + "/audio/" + sn.id + ".wav?v=" + c.run.mtime;
        td.appendChild(a);
        td.appendChild(el("div", "vmeta", it.audio_seconds + "s audio" + (it.hit_max_length ? " - hit max length" : "")));
      } else {
        td.className = "bad";
        td.textContent = it.status === "eos" ? "the model ended speech immediately" : (it.error || it.status);
      }
      tr.appendChild(td);
    });
    body.appendChild(tr);
  }
  cmpTable.appendChild(body);
};
const openCompare = async () => {
  cmpBox.hidden = !cmpBox.hidden;
  if (cmpBox.hidden) return;
  if (!cmpData) {
    cmpInfo.textContent = "looking for cached runs…";
    try {
      const resp = await fetch("/validation/runs");
      if (!resp.ok) throw new Error(resp.status + " " + await resp.text());
      cmpData = await resp.json();
      if (!cmpCurrent) cmpCurrent = await (await fetch("/validation")).json();
    } catch (e) { cmpInfo.textContent = "failed: " + e.message; return; }
    cmpInfo.textContent = "";
    renderCmpPick();
  }
  renderCmpTable(true);
};
cmpBtn.onclick = openCompare;
const pollValidation = async () => {
  let s;
  try { s = await (await fetch("/validation")).json(); }
  catch (e) { setTimeout(pollValidation, 5000); return; }
  cmpCurrent = s;
  if (cmpData && !cmpBox.hidden) renderCmpTable(false);
  renderItems(s.items);
  vstatus.className = "";
  const head = "voice " + (s.voice || "?") + formatValSettings(s.settings) + ", " + s.done + "/" + s.total;
  if (s.state === "running") {
    vstatus.textContent = "generating validation set… " + head;
    valBtn.disabled = true; redoBtn.disabled = true; setTimeout(pollValidation, 5000); return;
  }
  valBtn.disabled = false; redoBtn.disabled = false;
  const runLbl = (M_META && M_META.model_run) ? `${M_META.model_run} (${s.model_sha256})` : s.model_sha256;
  if (s.state === "done") vstatus.textContent = head + (s.failed ? " (" + s.failed + " failed)" : "") +
    " - cached for model " + runLbl;
  else if (s.state === "warming") { vstatus.textContent = "warming up…"; setTimeout(pollValidation, 5000); }
  else { vstatus.className = "err"; vstatus.textContent = (s.error || "stopped") + " - " + head; }
};
valBtn.onclick = async () => {
  valBtn.disabled = true;
  vstatus.className = ""; vstatus.textContent = "starting…";
  const f = new FormData();
  const voice = document.getElementById("voice").value;
  if (voice) f.append("voice", voice);
  appendSamplingParams(f);
  const r = await fetch("/validation/start", {method: "POST", body: f});
  if (!r.ok) { vstatus.className = "err"; vstatus.textContent = r.status + ": " + await r.text(); valBtn.disabled = false; return; }
  pollValidation();
};
redoBtn.onclick = async () => {
  const voice = document.getElementById("voice").value;
  if (document.getElementById("file").files[0]) {
    vstatus.className = "err";
    vstatus.textContent = "the validation set uses a voice from the list: clear the uploaded file first";
    return;
  }
  if (!voice) { vstatus.className = "err"; vstatus.textContent = "no voice selected"; return; }
  if (!confirm("Delete the cached validation set of this model and render it again with " + voice +
      " and current sampling settings? It takes a few minutes.")) return;
  valBtn.disabled = true; redoBtn.disabled = true;
  vstatus.className = ""; vstatus.textContent = "starting…";
  shown.clear(); vlist.replaceChildren();
  const f = new FormData(); f.append("force", "true"); f.append("voice", voice);
  appendSamplingParams(f);
  const r = await fetch("/validation/start", {method: "POST", body: f});
  if (!r.ok) {
    let d = await r.text(); try { d = JSON.stringify(JSON.parse(d).detail); } catch (e) {}
    vstatus.className = "err"; vstatus.textContent = r.status + ": " + d;
    valBtn.disabled = false; redoBtn.disabled = false; return;
  }
  pollValidation();
};
let ready = false;
const waitReady = async () => {
  status.className = ""; status.textContent = "warming up… (loading the model)";
  for (;;) {
    try {
      const h = await (await fetch("/health")).json();
      renderModelMeta(h);
      if (h.ready) { ready = true; status.textContent = "ready"; go.disabled = false; valBtn.disabled = false; redoBtn.disabled = false; syncMilim(); return; }
      if (h.error) { status.className = "err"; status.textContent = "model failed to load: " + h.error; return; }
    } catch (e) {}
    await new Promise(res => setTimeout(res, 3000));
  }
};
waitReady();
const voiceAndParams = f => {
  const file = document.getElementById("file").files[0];
  if (file) f.append("voice_wav", file); else f.append("voice", document.getElementById("voice").value);
  for (const k of [...Object.keys(D), "seed"]) {
    const v = document.getElementById("p_" + k).value;
    if (v !== "") f.append(k, v);
  }
  return f;
};
go.onclick = async () => {
  const f = new FormData();
  f.append("text", document.getElementById("text").value);
  voiceAndParams(f);
  status.className = ""; status.textContent = "generating…";
  const t0 = performance.now();
  const r = await fetch("/tts", {method: "POST", body: f});
  if (!r.ok) {
    let d = await r.text(); try { d = JSON.stringify(JSON.parse(d).detail); } catch (e) {}
    status.className = "err"; status.textContent = r.status + ": " + d; return;
  }
  const blob = await r.blob();
  const p = document.getElementById("player");
  p.src = URL.createObjectURL(blob); p.play();
  status.textContent = `${r.headers.get("X-Audio-Seconds")}s audio in ${r.headers.get("X-Generation-Seconds")}s ` +
    `(${r.headers.get("X-Realtime-Factor")}x realtime) on ${r.headers.get("X-Device")}` +
    (r.headers.get("X-Hit-Max-Length") === "true" ? "\\nhit max length: EOS never fired" : "");
};

// MILIM-Bench sample picker: choose up to max_batch sentences, generate them in one request.
const milimBtn = document.getElementById("milimBtn");
const dlg = document.getElementById("milimDlg");
const mcat = document.getElementById("mcat");
const mlist = document.getElementById("mlist");
const mcount = document.getElementById("mcount");
const mgo = document.getElementById("mgo");
const mres = document.getElementById("mres");
let M = null, busy = false;
const picked = new Map();  // sample id -> sample, in the order chosen
const syncMilim = () => {
  const n = picked.size;
  mcount.textContent = "נבחרו " + n + "/" + (M ? M.max_batch : 5);
  mgo.textContent = n ? "צור (" + n + ")" : "צור";
  mgo.disabled = !n || !ready || busy;
  mgo.title = ready ? "" : "המודל נטען…";
  milimBtn.disabled = busy;
  const full = M && n >= M.max_batch;
  for (const row of mlist.children) {
    const id = Number(row.dataset.id), on = picked.has(id);
    row.classList.toggle("sel", on);
    row.classList.toggle("off", full && !on);
    row.querySelector("input").disabled = full && !on;
  }
};
const sentenceEl = s => {
  const box = el("div", "mtext");
  const ipa = new Map(s.targets.map(t => [t.index, t.ipa]));
  s.text.split(/\\s+/).forEach((w, i) => {
    if (i) box.append(" ");
    if (ipa.has(i)) { const m = el("mark", "w", w); m.dataset.ipa = ipa.get(i); box.append(m); }
    else box.append(w);
  });
  return box;
};
const renderMilimList = () => {
  mlist.replaceChildren();
  mlist.scrollTop = 0;
  for (const s of M.samples) {
    if (mcat.value !== "" && s.category !== mcat.value) continue;
    const row = el("label", "mrow");
    row.dataset.id = s.id;
    const cb = el("input"); cb.type = "checkbox"; cb.checked = picked.has(s.id);
    cb.onchange = () => {
      if (cb.checked) picked.set(s.id, s); else picked.delete(s.id);
      syncMilim();
    };
    const body = el("div");
    body.append(sentenceEl(s));
    if (mcat.value === "") body.append(el("div", "mcat", s.category));
    row.append(cb, body);
    mlist.append(row);
  }
  syncMilim();
};
fetch("/milim").then(r => r.ok ? r.json() : null).then(j => {
  if (!j) return;
  M = j;
  mcat.add(new Option("כל הקטגוריות (" + M.samples.length + ")", ""));
  M.categories.forEach(c => mcat.add(new Option(c.name + " (" + c.count + ")", c.name)));
  milimBtn.hidden = false;
  renderMilimList();
}).catch(() => {});
mcat.onchange = renderMilimList;
milimBtn.onclick = () => dlg.showModal();
document.getElementById("mclear").onclick = () => { picked.clear(); renderMilimList(); };
mgo.onclick = () => { dlg.close(); runMilim(); };
const runMilim = async () => {
  const items = [...picked.values()];
  busy = true; syncMilim();
  const rows = new Map(), pending = [];
  mres.replaceChildren(el("div", "vcat", "תוצאות MILIM"));
  for (const s of items) {
    const row = el("div", "vrow");
    row.append(el("div", "vtext", s.text), el("div", "vmeta", s.category + " - ממתין…"));
    mres.append(row); rows.set(s.id, row); pending.push(s);
  }
  const markNext = () => {
    if (pending.length) rows.get(pending[0].id).lastChild.textContent = pending[0].category + " - generating…";
  };
  const finish = (s, r) => {
    pending.splice(pending.indexOf(s), 1);
    const row = rows.get(s.id);
    row.replaceChildren(el("div", "vtext", s.text));
    row.classList.toggle("bad", r.status !== "ok");
    if (r.status === "ok") {
      const a = el("audio"); a.controls = true; a.src = "data:audio/wav;base64," + r.wav;
      row.append(a, el("div", "vmeta", s.category + " - " + r.audio_seconds + "s audio in " +
        r.generation_seconds + "s" + (r.hit_max_length ? " - hit max length" : "")));
    } else {
      row.append(el("div", "vmeta", s.category + " - " +
        (r.status === "eos" ? "the model ended speech immediately" : r.error)));
    }
  };
  const fail = msg => { for (const s of [...pending]) finish(s, {status: "error", error: msg}); };
  try {
    const f = new FormData();
    items.forEach(s => f.append("ids", s.id));
    voiceAndParams(f);
    markNext();
    const r = await fetch("/milim/generate", {method: "POST", body: f});
    if (!r.ok) {
      let d = await r.text(); try { d = JSON.stringify(JSON.parse(d).detail); } catch (e) {}
      fail(r.status + ": " + d); return;
    }
    const reader = r.body.getReader(), dec = new TextDecoder();
    let buf = "";
    for (;;) {
      const {done, value} = await reader.read();
      if (done) break;
      buf += dec.decode(value, {stream: true});
      let i;
      while ((i = buf.indexOf("\\n")) >= 0) {
        const line = buf.slice(0, i); buf = buf.slice(i + 1);
        if (!line.trim()) continue;
        const res = JSON.parse(line);
        const s = pending.find(p => p.id === res.id);
        if (s) { finish(s, res); markNext(); }
      }
    }
    fail("no response");
  } catch (e) {
    fail(String(e));
  } finally {
    busy = false; syncMilim();
  }
};
syncMilim();
</script></body></html>
"""


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--voices-dir", type=Path, default=DEFAULT_VOICES_DIR)
    parser.add_argument(
        "--device", default="auto", help="auto = cuda, then mps, then cpu (default); or force one"
    )
    parser.add_argument("--sentences", type=Path, default=DEFAULT_SENTENCES)
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path(os.environ.get("VALIDATION_CACHE_DIR") or DEFAULT_CACHE_DIR),
        help="where the validation set audio is cached (persistent storage on Cloud Run)",
    )
    parser.add_argument(
        "--milim-dir",
        type=Path,
        default=Path(os.environ.get("MILIM_DIR") or DEFAULT_MILIM_DIR),
        help="MILIM-Bench snapshot (downloaded at image build time; locally: "
        "python -m deploy.milim_bench)",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8002)
    parser.add_argument(
        "--threads", type=int, default=min(4, os.cpu_count() or 1), help="torch CPU threads"
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    torch.set_num_threads(args.threads)  # pocket_tts pins this to 1 at import
    config = args.config if args.config.is_absolute() else Path.cwd() / args.config
    store = None
    store_dir = os.environ.get("MODEL_STORE")
    if store_dir:
        # The model comes from the store, pinned to one weights hash; config and tokenizer are
        # copied now (the config is read below), the weights in the background.
        pinned = os.environ.get("MODEL_SHA256")
        if not pinned:
            raise SystemExit("MODEL_STORE is set: also set MODEL_SHA256 to the model to serve")
        if config != DEFAULT_CONFIG:
            raise SystemExit("--config cannot be combined with MODEL_STORE (the store has it)")
        try:
            store = ModelStore(Path(store_dir), pinned, REPO_ROOT)
            store.fetch_small()
        except (OSError, ValueError, KeyError) as exc:
            raise SystemExit(f"model store: {exc}") from exc
        logger.info("model store %s, model %s", store_dir, store.sha256)
    if not config.exists():
        raise SystemExit(f"config not found: {config}")
    encoder_store = open_encoder_store(config, store_dir)
    voices_dir = args.voices_dir.resolve()
    sentences = args.sentences.resolve()
    cache_dir = args.cache_dir.resolve()
    milim_dir = args.milim_dir.resolve()
    if not sentences.exists():
        raise SystemExit(f"sentences file not found: {sentences}")
    # The config's weights/tokenizer paths are relative to the repo root.
    os.chdir(REPO_ROOT)

    device = pick_device(args.device)
    logger.info("device: %s", device)
    if not list_voice_files(voices_dir):
        logger.warning("no voice prompts in %s; upload one per request or add files", voices_dir)
    loader = EngineLoader(config, voices_dir, device, store, encoder_store)
    loader.start()
    validation = make_validation_set(loader, sentences, cache_dir)
    downloads = Downloads(
        REPO_ROOT,
        config,
        sentences,
        DEFAULTS,
        lambda: loader.model_sha256 or (store.sha256 if store is not None else None),
        encoder_info=loader.semantic_encoder_info,
        weights_size=store.files["model.safetensors"]["size"] if store is not None else None,
    )
    logger.info("validation cache: %s", cache_dir)
    milim = MilimBench(milim_dir)
    uvicorn.run(
        create_app(loader, validation, downloads, milim),
        host=args.host,
        port=args.port,
        log_level="info",
    )


if __name__ == "__main__":
    main()
