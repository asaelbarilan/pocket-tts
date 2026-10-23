"""Standalone HTTP API for testing the Hebrew 24-layer model: text + voice prompt -> WAV.

Run from the repo root:

    uv run python -m deploy.hebrew_tts_server                      # cuda > mps > cpu
    uv run python -m deploy.hebrew_tts_server --device cpu --port 8002

Why this exists instead of `pocket-tts serve`: the model was trained with CFG dropout and is
evaluated with cfg=2.0, and `pocket_tts` has no CFG. This server samples with the training
code (`TrainableTTS.generate`), so it matches the training evals: same CFG, EOS threshold,
temperature and decoding. The weights are the exported artifacts/model.safetensors, loaded
through the config in artifacts/hebrew_24l.yaml.

The text goes to the tokenizer as is (no chunking, no normalization), like the training evals
do, and is generated in a single pass of at most `max_seconds`.
"""

from __future__ import annotations

import argparse
import hashlib
import io
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
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import HTMLResponse, Response

from pocket_tts import TTSModel
from pocket_tts.data.audio import audio_read
from pocket_tts.data.audio_utils import convert_audio
from pocket_tts.modules.stateful_module import init_states
from training.args import TrainArgs
from training.modules.model import TrainableTTS
from training.modules.samplers import LSD

logger = logging.getLogger("hebrew_tts_server")

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = REPO_ROOT / "artifacts" / "hebrew_24l.yaml"
DEFAULT_VOICES_DIR = Path(__file__).resolve().parent / "voices"
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
        self.tokenize = flow_lm.conditioner.tokenizer.sp.encode
        self.n_layers = len(flow_lm.transformer.layers)
        logger.info(
            "loaded %d-layer model on %s in %.1fs",
            self.n_layers,
            device,
            time.perf_counter() - started,
        )

    # -- voices -------------------------------------------------------------------------

    def list_voices(self) -> dict[str, Path]:
        if not self.voices_dir.is_dir():
            return {}
        return {
            p.name: p
            for p in sorted(self.voices_dir.iterdir())
            if p.is_file() and p.suffix.lower() in AUDIO_SUFFIXES and not p.name.startswith(".")
        }

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
            max_frames=max_frames,
            temp=temp,
            n_steps=n_steps,
            cfg_coef=cfg,
            eos_threshold=eos_threshold,
        )[0]
        if latents.shape[0] < MIN_FRAMES:
            return None, False
        return self._decode(latents).float().cpu(), latents.shape[0] >= max_frames


def wav_bytes(samples: torch.Tensor, sample_rate: int) -> bytes:
    buffer = io.BytesIO()
    audio = samples.clamp(-1.0, 1.0).numpy()
    soundfile.write(buffer, audio, sample_rate, format="WAV", subtype="PCM_16")
    return buffer.getvalue()


def create_app(engine: Engine) -> FastAPI:
    app = FastAPI(title="Hebrew Pocket TTS (test API)")

    @app.get("/health")
    def health():
        return {
            "status": "ok",
            "device": str(engine.device),
            "layers": engine.n_layers,
            "sample_rate": engine.sample_rate,
            "config": str(engine.config),
            "voices": len(engine.list_voices()),
            "defaults": DEFAULTS,
        }

    @app.get("/voices")
    def voices():
        return {"voices": list(engine.list_voices())}

    @app.get("/", response_class=HTMLResponse)
    def index():
        return INDEX_HTML.replace("__DEFAULTS__", repr(DEFAULTS).replace("'", '"'))

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
        text = text.strip()
        if not text:
            raise HTTPException(status_code=400, detail="text is empty")
        if len(text) > MAX_TEXT_CHARS:
            raise HTTPException(status_code=400, detail=f"text is over {MAX_TEXT_CHARS} characters")

        started = time.perf_counter()
        temp_path: str | None = None
        try:
            if voice_wav is not None and voice_wav.filename:
                data = voice_wav.file.read()
                suffix = Path(voice_wav.filename).suffix.lower() or ".wav"
                with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as handle:
                    handle.write(data)
                    temp_path = handle.name
                key, path = ("upload", hashlib.sha256(data).hexdigest()), Path(temp_path)
            elif voice:
                known = engine.list_voices()
                if voice not in known:
                    raise HTTPException(
                        status_code=400, detail=f"unknown voice {voice!r}; see GET /voices"
                    )
                path = known[voice]
                key = ("file", voice, path.stat().st_mtime_ns)
            else:
                raise HTTPException(status_code=400, detail="give `voice` or upload `voice_wav`")

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
 body{font:15px system-ui,sans-serif;max-width:760px;margin:2rem auto;padding:0 1rem;color:#222}
 textarea{width:100%;height:7rem;font-size:1.2rem;direction:rtl;box-sizing:border-box}
 label{display:block;margin:.6rem 0 .2rem;font-weight:600}
 .grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(170px,1fr));gap:.4rem .8rem}
 .grid label{font-weight:400;margin:0} .grid input{width:100%;box-sizing:border-box}
 button{margin:1rem 0;padding:.6rem 1.4rem;font-size:1rem} audio{width:100%}
 #status{color:#555;white-space:pre-wrap} #status.err{color:#b00020}
</style></head><body>
<h2>Hebrew TTS test</h2>
<label for="text">Text</label>
<textarea id="text" dir="rtl">שלום, זהו מבחן קצר של המודל.</textarea>
<label for="voice">Voice prompt</label>
<select id="voice"></select>
<label for="file">…or upload one (wins over the list; first <span id="ps"></span>s are used)</label>
<input type="file" id="file" accept="audio/*">
<details><summary style="margin-top:1rem;cursor:pointer">Sampling settings</summary><div class="grid" id="params"></div></details>
<button id="go">Generate</button>
<div id="status"></div>
<audio id="player" controls></audio>
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
document.getElementById("go").onclick = async () => {
  const f = new FormData();
  f.append("text", document.getElementById("text").value);
  const file = document.getElementById("file").files[0];
  if (file) f.append("voice_wav", file); else f.append("voice", document.getElementById("voice").value);
  for (const k of [...Object.keys(D), "seed"]) {
    const v = document.getElementById("p_" + k).value;
    if (v !== "") f.append(k, v);
  }
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
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8002)
    parser.add_argument(
        "--threads", type=int, default=min(8, os.cpu_count() or 1), help="torch CPU threads"
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    torch.set_num_threads(args.threads)  # pocket_tts pins this to 1 at import
    config = args.config if args.config.is_absolute() else Path.cwd() / args.config
    if not config.exists():
        raise SystemExit(f"config not found: {config}")
    voices_dir = args.voices_dir.resolve()
    # The config's weights/tokenizer paths are relative to the repo root.
    os.chdir(REPO_ROOT)

    device = pick_device(args.device)
    logger.info("device: %s", device)
    engine = Engine(config, voices_dir, device)
    if not engine.list_voices():
        logger.warning("no voice prompts in %s; upload one per request or add files", voices_dir)
    uvicorn.run(create_app(engine), host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
