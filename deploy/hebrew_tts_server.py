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
from the store and verified at startup.

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
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, Response, StreamingResponse

from deploy.downloads import CODE_ZIP, GUIDE, Downloads
from deploy.milim_bench import DEFAULT_DIR as DEFAULT_MILIM_DIR
from deploy.milim_bench import MAX_BATCH, MilimBench
from deploy.model_store import ModelStore
from deploy.validation_set import ValidationSet
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
    "max_seconds": 30.0,
    "prompt_seconds": 5.0,
}


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

    def __init__(self, config: Path, voices_dir: Path, device: torch.device):
        self.config = config
        self.voices_dir = voices_dir
        self.device = device
        self.lock = threading.Lock()
        self._voice_cache: OrderedDict[tuple, torch.Tensor] = OrderedDict()

        started = time.perf_counter()
        # Builds FlowLM + Mimi from the config and loads artifacts/model.safetensors (strict).
        tts = TTSModel.load_model(config=config)
        self.sample_rate: int = tts.sample_rate
        self.mimi = tts.mimi.to(device).eval()
        flow_lm = tts.flow_lm.to(device).eval()
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
        )[0]
        if latents.shape[0] < MIN_FRAMES:
            return None, False
        return self._decode(latents).float().cpu(), latents.shape[0] >= max_frames

    def render_voice_file(
        self, text: str, voice: str, seed: int
    ) -> tuple[bytes | None, float, bool]:
        """WAV bytes (None on immediate EOS), audio seconds and hit-max for a named voice file,
        with the default sampling settings. Takes the generation lock per call."""
        path = self.list_voices()[voice]
        key = ("file", voice, path.stat().st_mtime_ns)
        with self.lock:
            latents = self.voice_latents(key, path, DEFAULTS["prompt_seconds"])
            waveform, hit_max = self.synthesize(
                text,
                latents,
                temp=DEFAULTS["temp"],
                cfg=DEFAULTS["cfg"],
                n_steps=DEFAULTS["n_steps"],
                eos_threshold=DEFAULTS["eos_threshold"],
                max_seconds=DEFAULTS["max_seconds"],
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
    ):
        self.config = config
        self.voices_dir = voices_dir
        self.device = device
        self.store = store
        self.engine: Engine | None = None
        self.model_sha256: str | None = None  # set once the model is loaded
        self.stage = "starting"
        self.error: str | None = None

    def start(self) -> None:
        threading.Thread(target=self._load, name="engine-loader", daemon=True).start()

    def cache_key(self) -> str | None:
        """Names the validation cache: model weights + device kind (CPU and GPU audio differ)."""
        if self.engine is None or self.model_sha256 is None:
            return None
        return f"{self.model_sha256[:16]}-{self.device.type}"

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
            self.stage = "loading the model"
            engine = Engine(self.config, self.voices_dir, self.device)
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
    def generate(text: str, voice: str, seed: int) -> tuple[bytes | None, float, bool]:
        engine = loader.engine
        if engine is None:
            raise RuntimeError("model not loaded")
        return engine.render_voice_file(text, voice, seed)

    return ValidationSet(
        sentences,
        cache_dir,
        DEFAULTS,
        voices=lambda: list(list_voice_files(loader.voices_dir)),
        generate=generate,
        cache_key=loader.cache_key,
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
        # Weights, tokenizer, config and sentences stream from disk (Range requests work, so a
        # dropped 1.4 GB download can resume); the guide and code zip are generated.
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
        found = downloads.file(name)
        if found is None:
            raise HTTPException(status_code=404, detail="no such download")
        return FileResponse(found.path, filename=found.name)

    @app.get("/validation")
    def validation_status():
        return validation.status()

    @app.post("/validation/start")
    def validation_start():
        if loader.engine is None:
            raise HTTPException(
                status_code=503, detail="warming up...", headers={"Retry-After": "10"}
            )
        return validation.start()

    @app.get("/validation/audio/{item_id}.wav")
    def validation_audio(item_id: str):
        path = validation.audio_path(item_id)
        if path is None or not path.is_file():
            raise HTTPException(status_code=404, detail="no such validation audio")
        return FileResponse(path, media_type="audio/wav")

    @app.get("/ready")
    def ready():
        """200 only once the model is loaded: the Cloud Run startup probe, so a revision whose
        model cannot be loaded never takes traffic."""
        if loader.engine is None:
            return JSONResponse(
                status_code=503, content={"ready": False, "stage": loader.stage, "error": loader.error}
            )
        return {"ready": True, "model_sha256": loader.model_sha256}

    @app.get("/health")
    def health():
        engine = loader.engine
        voices = list_voice_files(loader.voices_dir)
        if engine is None:
            if loader.error:
                return JSONResponse(
                    status_code=500, content={"status": "load failed", "error": loader.error}
                )
            return {
                "status": "warming up...",
                "ready": False,
                "stage": loader.stage,
                "voices": len(voices),
            }
        return {
            "status": "ok",
            "ready": True,
            "model_sha256": loader.model_sha256,
            "model_store": str(loader.store.dir) if loader.store else None,
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

    @app.get("/", response_class=HTMLResponse)
    def index():
        return INDEX_HTML.replace("__DEFAULTS__", repr(DEFAULTS).replace("'", '"')).replace(
            "__DOWNLOADS__", downloads.html()
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
            "%.1fs audio in %.1fs (%.2fx) cfg=%s temp=%s eos=%s hit_max=%s chars=%d",
            seconds,
            elapsed,
            seconds / elapsed,
            cfg,
            temp,
            eos_threshold,
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
 #vstatus{color:#555} #vstatus.err{color:#b00020}
 .vcat{margin:1.2rem 0 .3rem;font-weight:700;text-transform:capitalize}
 .vrow{border:1px solid #ddd;border-radius:6px;padding:.5rem .7rem;margin:.4rem 0}
 .vtext{direction:rtl;font-size:1.2rem;margin-bottom:.3rem}
 .vmeta{color:#777;font-size:.85rem} .vrow.bad{border-color:#b00020}
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
<div class="dl"><b>Download for local inference</b>__DOWNLOADS__</div>
<label for="text">Text</label>
<textarea id="text" dir="rtl">שלום, זהו מבחן קצר של המודל.</textarea>
<label for="voice">Voice prompt</label>
<select id="voice"></select>
<label for="file">…or upload one (wins over the list; first <span id="ps"></span>s are used)</label>
<input type="file" id="file" accept="audio/*">
<details><summary style="margin-top:1rem;cursor:pointer">Sampling settings</summary><div class="grid" id="params"></div></details>
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
<div id="vstatus"></div>
<div id="vlist"></div>
<script>
const D = __DEFAULTS__;
const names = {temp:"temperature", cfg:"cfg (guidance)", n_steps:"decode steps",
  eos_threshold:"eos threshold", max_seconds:"max seconds", prompt_seconds:"prompt seconds", seed:"seed"};
const box = document.getElementById("params");
for (const k of [...Object.keys(D), "seed"]) {
  const l = document.createElement("label");
  l.textContent = names[k] + " ";
  const i = document.createElement("input");
  i.id = "p_" + k; i.type = "number"; i.step = "any"; i.value = D[k] ?? "";
  if (k === "seed") i.placeholder = "random";
  l.appendChild(i); box.appendChild(l);
}
document.getElementById("ps").textContent = D.prompt_seconds;
fetch("/voices").then(r => r.json()).then(j => {
  const s = document.getElementById("voice");
  j.voices.forEach(v => s.add(new Option(v, v)));
  if (!j.voices.length) s.add(new Option("(none - put files in deploy/voices or upload)", ""));
});
const status = document.getElementById("status");
const go = document.getElementById("go");
go.disabled = true;
const valBtn = document.getElementById("val");
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
      a.src = "/validation/audio/" + it.id + ".wav";
      row.appendChild(a);
      row.appendChild(el("div", "vmeta", it.id + " - " + it.audio_seconds + "s audio in " +
        it.generation_seconds + "s" + (it.hit_max_length ? " - hit max length" : "")));
    } else {
      row.appendChild(el("div", "vmeta", it.id + " - " +
        (it.status === "eos" ? "the model ended speech immediately" : it.error)));
    }
  }
};
const pollValidation = async () => {
  let s;
  try { s = await (await fetch("/validation")).json(); }
  catch (e) { setTimeout(pollValidation, 5000); return; }
  renderItems(s.items);
  vstatus.className = "";
  const head = "voice " + (s.voice || "?") + ", " + s.done + "/" + s.total;
  if (s.state === "running") {
    vstatus.textContent = "generating validation set… " + head;
    valBtn.disabled = true; setTimeout(pollValidation, 5000); return;
  }
  valBtn.disabled = false;
  if (s.state === "done") vstatus.textContent = head + (s.failed ? " (" + s.failed + " failed)" : "") +
    " - cached for model " + s.model_sha256;
  else if (s.state === "warming") { vstatus.textContent = "warming up…"; setTimeout(pollValidation, 5000); }
  else { vstatus.className = "err"; vstatus.textContent = (s.error || "stopped") + " - " + head; }
};
valBtn.onclick = async () => {
  valBtn.disabled = true;
  vstatus.className = ""; vstatus.textContent = "starting…";
  const r = await fetch("/validation/start", {method: "POST"});
  if (!r.ok) { vstatus.className = "err"; vstatus.textContent = r.status + ": " + await r.text(); valBtn.disabled = false; return; }
  pollValidation();
};
let ready = false;
const waitReady = async () => {
  status.className = ""; status.textContent = "warming up… (loading the model)";
  for (;;) {
    try {
      const h = await (await fetch("/health")).json();
      if (h.ready) { ready = true; status.textContent = "ready"; go.disabled = false; valBtn.disabled = false; syncMilim(); return; }
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
    loader = EngineLoader(config, voices_dir, device, store)
    loader.start()
    validation = make_validation_set(loader, sentences, cache_dir)
    downloads = Downloads(REPO_ROOT, config, sentences, DEFAULTS, lambda: loader.model_sha256)
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
