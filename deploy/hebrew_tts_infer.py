"""Standalone local inference for the Hebrew 24-layer model: text + voice prompt -> WAV.

This is the same pipeline as `deploy/hebrew_tts_server.py` (see inference_guide.md), without
the HTTP layer. Run it from the project root (the directory holding `artifacts/`,
`tokenizers/`, `pocket_tts/` and `training/`), because the paths inside
`artifacts/hebrew_24l.yaml` are relative to it:

    python hebrew_tts_infer.py --text "שלום, מה שלומך?" --voice my_voice.wav --out out.wav

or from Python:

    from hebrew_tts_infer import HebrewTTS, write_wav
    tts = HebrewTTS()                              # loads artifacts/hebrew_24l.yaml
    voice = tts.encode_voice("my_voice.wav")       # any speaker, >= 0.5 s, first 5 s are used
    wave = tts.synthesize("שלום", voice, seed=1)   # float32 tensor [samples], or None
    write_wav("out.wav", wave, tts.sample_rate)
"""

from __future__ import annotations

import argparse
import logging
import os
import time
from pathlib import Path

import soundfile
import torch

from pocket_tts import TTSModel
from pocket_tts.data.audio import audio_read
from pocket_tts.data.audio_utils import convert_audio
from pocket_tts.modules.stateful_module import init_states
from training.args import TrainArgs
from training.modules.model import TrainableTTS
from training.modules.samplers import LSD

logger = logging.getLogger("hebrew_tts_infer")

_HERE = Path(__file__).resolve().parent
# The project root: next to this file in the download bundle, one level up in the repo (deploy/).
ROOT = _HERE if (_HERE / "pocket_tts").is_dir() else _HERE.parent
MIN_FRAMES = 8  # a shorter generation is an immediate EOS, not speech

# What training and the training evals use. Every one is a keyword of `synthesize`.
DEFAULTS = {
    "temp": 0.3,
    "cfg": 2.0,
    "n_steps": 1,
    "eos_threshold": -1.0,
    "max_seconds": 30.0,
    "prompt_seconds": 5.0,
}


def pick_device(requested: str = "auto") -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class HebrewTTS:
    def __init__(self, config: Path | str = "artifacts/hebrew_24l.yaml", device: str = "auto"):
        # Paths inside the config (weights, tokenizer) are relative to the project root.
        os.chdir(ROOT)
        self.device = pick_device(device)
        started = time.perf_counter()
        # Builds FlowLM + Mimi from the config and loads artifacts/model.safetensors (strict).
        tts = TTSModel.load_model(config=Path(config))
        self.sample_rate: int = tts.sample_rate
        self.mimi = tts.mimi.to(self.device).eval()
        flow_lm = tts.flow_lm.to(self.device).eval()
        # The LSD sampler has no weights of its own at decode time; the wrapper only provides
        # `generate` (classifier-free guidance, EOS countdown).
        self.model = TrainableTTS(flow_lm, LSD(normalize=False), TrainArgs()).eval()
        self.tokenize = flow_lm.conditioner.tokenizer.sp.encode
        logger.info("loaded model on %s in %.1fs", self.device, time.perf_counter() - started)

    def encode_voice(
        self, path: Path | str, prompt_seconds: float = DEFAULTS["prompt_seconds"]
    ) -> torch.Tensor:
        """Voice prompt audio file -> Mimi latents [frames, channels] (the `voice` condition)."""
        audio, sr = audio_read(Path(path))  # [channels, samples]
        if prompt_seconds > 0:
            audio = audio[..., : int(prompt_seconds * sr)]
        if audio.shape[-1] < sr // 2:
            raise ValueError("voice prompt is shorter than 0.5 s")
        mono = convert_audio(audio, sr, self.sample_rate, 1)  # [1, T] at 24 kHz
        with torch.no_grad():
            latents = self.mimi.encode_to_latent(mono[None].to(self.device))  # [1, Tv, C]
        return latents[0].float()

    def synthesize(
        self,
        text: str,
        voice: torch.Tensor,
        *,
        temp: float = DEFAULTS["temp"],
        cfg: float = DEFAULTS["cfg"],
        n_steps: int = DEFAULTS["n_steps"],
        eos_threshold: float = DEFAULTS["eos_threshold"],
        max_seconds: float = DEFAULTS["max_seconds"],
        seed: int | None = None,
    ) -> torch.Tensor | None:
        """float32 waveform [samples] on the CPU at `sample_rate`, or None when the model ended
        speech at once (try another voice prompt or a higher `eos_threshold`). The text goes to
        the tokenizer as is: no normalization, no chunking, one pass of at most `max_seconds`."""
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
            return None
        if latents.shape[0] >= max_frames:
            logger.warning("hit max length (%.0f s): EOS never fired", max_seconds)
        ratio = round(self.mimi.encoder_frame_rate / self.mimi.frame_rate)
        state = init_states(self.mimi, 1, (latents.shape[0] + MIN_FRAMES) * ratio)
        with torch.no_grad():
            wave = self.mimi.decode_from_latent(latents[None].to(self.device), state)[0, 0]
        return wave.float().cpu()


def write_wav(path: Path | str, wave: torch.Tensor, sample_rate: int) -> None:
    soundfile.write(
        str(path), wave.clamp(-1.0, 1.0).numpy(), sample_rate, format="WAV", subtype="PCM_16"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--text", required=True)
    parser.add_argument("--voice", required=True, type=Path, help="voice prompt audio file")
    parser.add_argument("--out", type=Path, default=Path("out.wav"))
    parser.add_argument("--config", type=Path, default=Path("artifacts/hebrew_24l.yaml"))
    parser.add_argument("--device", default="auto", help="auto = cuda, mps, then cpu")
    parser.add_argument("--seed", type=int, default=None)
    for key, default in DEFAULTS.items():
        if key != "prompt_seconds":
            parser.add_argument(f"--{key.replace('_', '-')}", type=type(default), default=default)
    parser.add_argument("--prompt-seconds", type=float, default=DEFAULTS["prompt_seconds"])
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    voice_path, out_path = args.voice.resolve(), args.out.resolve()  # before the chdir to ROOT
    tts = HebrewTTS(args.config, args.device)
    voice = tts.encode_voice(voice_path, args.prompt_seconds)
    started = time.perf_counter()
    wave = tts.synthesize(
        args.text,
        voice,
        temp=args.temp,
        cfg=args.cfg,
        n_steps=args.n_steps,
        eos_threshold=args.eos_threshold,
        max_seconds=args.max_seconds,
        seed=args.seed,
    )
    if wave is None:
        raise SystemExit("the model ended speech immediately; try another voice prompt")
    elapsed = time.perf_counter() - started
    seconds = wave.shape[0] / tts.sample_rate
    write_wav(out_path, wave, tts.sample_rate)
    logger.info(
        "%.2fs audio in %.2fs (%.2fx realtime) -> %s", seconds, elapsed, seconds / elapsed, out_path
    )


if __name__ == "__main__":
    main()
