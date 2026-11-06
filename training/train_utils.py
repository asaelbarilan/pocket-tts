"""Helpers for the training entrypoint: compilation, lr schedule, samples."""

import json
import logging
import math
import random
import subprocess
import time
from pathlib import Path

import soundfile
import torch

from pocket_tts.conditioners.semantic import SEMANTIC_PREFIX
from pocket_tts.modules.stateful_module import init_states
from training.args import TrainArgs

logger = logging.getLogger("train")

LOG_FORMAT = "[%(asctime)s %(levelname)s %(name)s] %(message)s"
LOG_DATEFMT = "%d-%m %H:%M:%S"


def setup_logging(level: int = logging.INFO) -> None:
    logging.basicConfig(level=level, format=LOG_FORMAT, datefmt=LOG_DATEFMT)


def add_file_logging(run_dir: Path, rank: int = 0) -> Path:
    """Mirror the stdout logs into a timestamped file under run_dir/logs."""
    log_dir = Path(run_dir) / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    suffix = "" if rank == 0 else f"_rank{rank}"
    path = log_dir / f"train_{stamp}{suffix}.log"
    handler = logging.FileHandler(path)
    handler.setFormatter(logging.Formatter(LOG_FORMAT, datefmt=LOG_DATEFMT))
    logging.getLogger().addHandler(handler)
    return path


class ProgressLog:
    """Append-only jsonl of training events in the run dir, continued across restarts."""

    def __init__(self, path: Path, enabled: bool = True):
        self.path = Path(path)
        self.enabled = enabled

    def log(self, event: str, step: int, metrics: dict | None = None, **fields) -> None:
        if not self.enabled:
            return
        record = {
            "time": time.strftime("%Y-%m-%d %H:%M:%S%z"),
            "type": event,
            "step": step,
            **fields,
        }
        if metrics is not None:
            record["metrics"] = metrics
        with open(self.path, "a") as f:
            f.write(json.dumps(record) + "\n")


def git_commit() -> str | None:
    """HEAD's short sha, suffixed "-dirty" when the tree has uncommitted changes."""

    def run(*cmd):
        try:
            out = subprocess.run(cmd, capture_output=True, text=True, timeout=10, check=False)
        except (OSError, subprocess.SubprocessError):
            return None
        return out.stdout if out.returncode == 0 else None

    sha = run("git", "rev-parse", "--short", "HEAD")
    if sha is None:
        return None
    dirty = run("git", "status", "--porcelain")
    return sha.strip() + ("-dirty" if dirty else "")


def _compile_models(model, mimi) -> None:
    """Per-layer compilation: whole-module compile trips dynamo on the
    streaming-state plumbing, individual layers and the flow head are clean.
    In-place .compile() (not torch.compile(module)) so state_dict keys stay
    free of _orig_mod. prefixes and checkpoints load in uncompiled code.
    The frozen Mimi encoder runs under no_grad outside the DDP module, so
    compiling it is safe on any GPU count (~6ms/step)."""

    def _compile_backbone(fl):
        for layer in fl.transformer.layers:
            layer.compile(dynamic=True)

    _compile_backbone(model.flow_lm)
    model.flow_lm.flow_net.compile(dynamic=True)
    if model.distill_teacher is not None:
        _compile_backbone(model.distill_teacher)
    mimi.encoder.compile(dynamic=True)
    mimi.encoder_transformer.compile(dynamic=True)


def optimizer_param_groups(model: torch.nn.Module, args: TrainArgs):
    """model.parameters() as before; with semantic conditioning, two groups, the fusion's at
    `semantic_lr_scale` times the learning rate (read by the schedule as `lr_scale`)."""
    if getattr(model.flow_lm, "semantic_fusion", None) is None:
        return model.parameters()
    prefix = "flow_lm." + SEMANTIC_PREFIX
    named = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    return [
        {"params": [p for n, p in named if not n.startswith(prefix)]},
        {
            "params": [p for n, p in named if n.startswith(prefix)],
            "lr_scale": args.semantic_lr_scale,
        },
    ]


def lr_at(step: int, args: TrainArgs) -> float:
    lr = args.optim.lr
    if step < args.optim.warmup_steps:
        return lr * (step + 1) / args.optim.warmup_steps
    if args.optim.schedule == "cosine":
        progress = (step - args.optim.warmup_steps) / max(
            1, args.max_steps - args.optim.warmup_steps
        )
        floor = lr * args.optim.lr_min_ratio
        return floor + 0.5 * (lr - floor) * (1 + math.cos(math.pi * min(1.0, progress)))
    return lr


SAMPLE_VOICE_MIN_SEC = 2.0
SAMPLE_VOICE_MAX_SEC = 5.0


def pick_sample_voice(
    voice_prompt_latents: torch.Tensor,
    num_voice_prompt_frames: torch.Tensor,
    voice_audio: torch.Tensor,
    frame_rate: float,
    sample_rate: int,
    seed: int,
    step: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pick a random voice prompt from the batch (preferring 2-5s) and trim its padding."""
    max_frames = voice_prompt_latents.shape[1]
    frame_counts = [
        max(1, min(int(n), max_frames)) for n in num_voice_prompt_frames.detach().cpu().tolist()
    ]
    preferred = [
        i
        for i, n in enumerate(frame_counts)
        if SAMPLE_VOICE_MIN_SEC <= n / frame_rate <= SAMPLE_VOICE_MAX_SEC
    ]
    candidates = preferred or list(range(len(frame_counts)))
    idx = random.Random(f"sample_voice:{seed}:{step}").choice(candidates)
    n_frames = frame_counts[idx]
    valid_samples = min(voice_audio.shape[-1], round(n_frames * sample_rate / frame_rate))
    trimmed_latents = voice_prompt_latents[idx, :n_frames].detach().clone()
    trimmed_audio = voice_audio[idx, 0, :valid_samples].detach().cpu().clone()
    return trimmed_latents, trimmed_audio


@torch.no_grad()
def write_samples(
    model,
    mimi,
    tokenize,
    args,
    run_dir,
    step,
    voice_latents,
    device,
    voice_audio: torch.Tensor | None = None,
):
    """Synthesize the configured sentences and return their local WAV paths."""
    out_dir = run_dir / "samples"
    out_dir.mkdir(exist_ok=True)
    written = []
    if voice_audio is not None and voice_audio.numel() > 0:
        voice_path = out_dir / f"step{step:08d}_voice.wav"
        soundfile.write(str(voice_path), voice_audio.float().cpu().numpy(), mimi.sample_rate)
        written.append(("[voice prompt]", voice_path))
    model.eval()
    tokens = [torch.tensor(tokenize(s), dtype=torch.long) for s in args.sample_sentences]
    with torch.no_grad():
        outs = model.generate(
            tokens,
            [voice_latents] * len(tokens),
            temp=args.sample_temp,
            cfg_coef=args.sample_cfg_coef,
            texts=list(args.sample_sentences),  # read by semantic conditioning only
        )
        ratio = round(mimi.encoder_frame_rate / mimi.frame_rate)
        for i, latents in enumerate(outs):
            if latents.shape[0] < 8:  # mimi decoder needs a few frames of context
                logger.warning(f"sample {i} at step {step}: empty generation, skipped")
                continue
            state = init_states(mimi, 1, (latents.shape[0] + 8) * ratio)
            audio = mimi.decode_from_latent(latents[None].to(device), state)[0, 0]
            soundfile.write(
                str(out_dir / f"step{step:08d}_{i}.wav"),
                audio.float().cpu().numpy(),
                mimi.sample_rate,
            )
            written.append((args.sample_sentences[i], out_dir / f"step{step:08d}_{i}.wav"))
    model.train()
    logger.info(f"wrote {len(tokens)} samples at step {step}")
    return written
