from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path

import safetensors.torch
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

from hebrew_training.data import read_jsonl
from hebrew_training.lsd import AdaptiveLossWeight, flow_matching_lsd_loss
from hebrew_training.model_utils import (
    choose_device,
    install_tokenizer,
    require_voice_cloning_access,
)
from pocket_tts import TTSModel
from pocket_tts.conditioners.base import TokenizedText


class LatentDataset(Dataset):
    def __init__(self, manifest: Path, expected_base_language: str | None = None):
        self.rows = list(read_jsonl(manifest))
        if not self.rows:
            raise ValueError(f"No rows found in {manifest}")
        for row in self.rows:
            if "latent_path" not in row:
                raise ValueError(f"{manifest} is not a cached-latent manifest")
            if expected_base_language is None:
                continue
            cached_language = row.get("latent_base_language")
            # Manifests produced before this safety field was added used English. Keep
            # those existing experiments usable, but never accept them for another base.
            if cached_language is None and expected_base_language == "english":
                continue
            if cached_language != expected_base_language:
                found = cached_language or "legacy/english"
                raise ValueError(
                    f"{manifest} contains latents for {found}, but training requested "
                    f"--base-language {expected_base_language}. Recompute target and prompt "
                    "latents with that exact base model."
                )

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict:
        row = self.rows[index]
        tensors = safetensors.torch.load_file(row["latent_path"])
        return {"text": row["text"], "target": tensors["target"], "prompt": tensors["prompt"]}


def singleton_collate(batch: list[dict]) -> dict:
    if len(batch) != 1:
        raise ValueError("This experimental trainer currently requires --batch-size 1")
    return batch[0]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Experimental Pocket TTS Hebrew adaptation.")
    parser.add_argument("--train-manifest", type=Path, required=True)
    parser.add_argument("--validation-manifest", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--base-language", default="english")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--steps", type=int, default=20_000)
    parser.add_argument("--gradient-accumulation", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-steps", type=int, default=500)
    parser.add_argument("--head-samples", type=int, default=128)
    parser.add_argument(
        "--head-batch-multiplier",
        type=int,
        default=1,
        help=(
            "Reuse each backbone output across N independent noise/timestep draws in the "
            "flow head. The paper (Table 14) uses 8; this trainer was written with 1."
        ),
    )
    parser.add_argument(
        "--loss-mode",
        choices=("flow", "fm-lsd"),
        default="flow",
        help=(
            "flow preserves the original experimental trainer. fm-lsd enables the paper's "
            "flow-matching/Lagrangian-self-distillation split and adaptive weighting."
        ),
    )
    parser.add_argument(
        "--lsd-fraction",
        type=float,
        default=0.25,
        help="Fraction of flow-head examples assigned to LSD when --loss-mode=fm-lsd.",
    )
    parser.add_argument(
        "--adaptive-weight-channels",
        type=int,
        default=128,
        help="Fourier feature width for the training-only adaptive loss-weight module.",
    )
    parser.add_argument("--eos-weight", type=float, default=0.1)
    parser.add_argument(
        "--eos-label-frames",
        type=int,
        default=1,
        help="Supervise the stop signal on the last N frames instead of exactly one.",
    )
    parser.add_argument(
        "--freeze-eos-after",
        type=int,
        default=0,
        help="After this step the EOS term stops contributing gradient. 0 disables.",
    )
    parser.add_argument(
        "--eos-early-stop-patience",
        type=int,
        default=0,
        help=(
            "Stop the run after this many consecutive evaluations where validation EOS "
            "exceeds its best by --eos-early-stop-factor. 0 disables."
        ),
    )
    parser.add_argument("--eos-early-stop-factor", type=float, default=1.3)
    parser.add_argument("--eos-early-stop-min-step", type=int, default=2000)
    parser.add_argument("--save-every", type=int, default=1000)
    parser.add_argument(
        "--skip-final-checkpoint",
        action="store_true",
        help="Do not save at the final step unless it is also a periodic save; useful for smoke tests.",
    )
    parser.add_argument("--eval-every", type=int, default=250)
    # 8 was far too few: hebrew-20k validation was noisy enough that it could not be
    # used for model selection. Pass 0 to evaluate the whole validation set.
    parser.add_argument("--eval-samples", type=int, default=64)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--resume", type=Path)
    parser.add_argument(
        "--init-flow-checkpoint",
        type=Path,
        help=(
            "Initialize only FlowLM weights from a checkpoint and start a fresh optimizer/run. "
            "Use this, rather than --resume, when changing the loss objective."
        ),
    )
    parser.add_argument("--limit-train-samples", type=int)
    return parser.parse_args()


def compute_loss(
    model,
    item: dict,
    head_samples: int,
    eos_weight: float,
    device,
    eos_label_frames: int = 1,
    eos_trainable: bool = True,
    head_batch_multiplier: int = 1,
    lsd_fraction: float = 0.0,
    adaptive_loss_weight: AdaptiveLossWeight | None = None,
) -> tuple:
    flow_lm = model.flow_lm
    target = item["target"].to(device=device, dtype=flow_lm.dtype).unsqueeze(0)
    prompt = item["prompt"].to(device=device, dtype=flow_lm.dtype).unsqueeze(0)

    prepared = flow_lm.conditioner.prepare(item["text"])
    text_embeddings = flow_lm.conditioner(TokenizedText(prepared.tokens))
    prompt_embeddings = F.linear(prompt, flow_lm.speaker_proj_weight)
    if flow_lm.insert_bos_before_voice:
        prompt_embeddings = torch.cat([flow_lm.bos_before_voice, prompt_embeddings], dim=1)
    prefix_embeddings = torch.cat([prompt_embeddings, text_embeddings], dim=1)

    teacher_input = torch.cat(
        [
            torch.full((1, 1, flow_lm.ldim), float("nan"), device=device, dtype=flow_lm.dtype),
            target[:, :-1],
        ],
        dim=1,
    )
    teacher_input = torch.where(torch.isnan(teacher_input), flow_lm.bos_emb, teacher_input)
    # The released backbone implementation supports a None state, but its runtime
    # beartype annotation says dict. Call the same layers directly for full-sequence
    # causal teacher forcing.
    transformer_input = torch.cat([prefix_embeddings, flow_lm.input_linear(teacher_input)], dim=1)
    hidden = flow_lm.transformer(transformer_input, model_state=None)
    hidden = flow_lm.out_norm(hidden)
    hidden = hidden[:, -teacher_input.shape[1] :].float()

    frame_count = target.shape[1]
    selected = torch.randperm(frame_count, device=device)[: min(frame_count, head_samples)]
    condition = hidden[:, selected].reshape(-1, hidden.shape[-1])
    clean = target[:, selected].reshape(-1, target.shape[-1]).float()
    # Head batch multiplier (paper Table 14 uses 8; this reimplementation used 1).
    # The backbone pass that produced `hidden` is the expensive part, and the flow head is
    # a small MLP, so reusing the same conditioning across several independent (t, noise)
    # draws buys much more gradient signal per backbone pass for little extra compute.
    # The paper credits it with faster convergence and better final quality at comparable
    # cost. Default 1 keeps the previous behaviour exactly.
    if head_batch_multiplier > 1:
        condition = condition.repeat(head_batch_multiplier, 1)
        clean = clean.repeat(head_batch_multiplier, 1)
    if lsd_fraction > 0:
        head_loss, flow_loss, lsd_loss = flow_matching_lsd_loss(
            flow_lm.flow_net, condition, clean, lsd_fraction, adaptive_loss_weight
        )
    else:
        # Preserve the original flow-only path exactly for old experiments/checkpoints.
        noise = torch.randn_like(clean)
        time_value = torch.rand((clean.shape[0], 1), device=device, dtype=clean.dtype)
        interpolated = (1 - time_value) * noise + time_value * clean
        predicted_velocity = flow_lm.flow_net(condition, time_value, time_value, interpolated)
        # mse_loss averages over the expanded batch, so the loss scale is unchanged and
        # values stay comparable to runs made with multiplier 1.
        flow_loss = F.mse_loss(predicted_velocity, clean - noise)
        head_loss = flow_loss
        lsd_loss = torch.zeros((), device=device, dtype=flow_loss.dtype)

    # The stop signal is supervised on the last `eos_label_frames` frames rather than
    # exactly one. With a single positive per clip the head has almost nothing to learn
    # from and memorises clip lengths instead; widening the target is the cheapest way to
    # give it more signal. Default 1 reproduces the original behaviour.
    positives = max(1, min(eos_label_frames, frame_count))
    eos_target = torch.zeros((1, frame_count), device=device)
    eos_target[:, -positives:] = 1
    pos_weight = max(1.0, min((frame_count - positives) / positives, 20.0))
    eos_logits = flow_lm.out_eos(hidden).squeeze(-1)
    eos_loss = F.binary_cross_entropy_with_logits(
        eos_logits, eos_target, pos_weight=torch.tensor([pos_weight], device=device)
    )
    if not eos_trainable:
        # Freeze the stop signal: still reported so curves stay comparable, but it no
        # longer contributes gradient to the head or the backbone.
        return head_loss, flow_loss.detach(), lsd_loss.detach(), eos_loss.detach()
    return (
        head_loss + eos_weight * eos_loss,
        flow_loss.detach(),
        lsd_loss.detach(),
        eos_loss.detach(),
    )


def save_checkpoint(
    model,
    optimizer,
    scheduler,
    run_dir: Path,
    step: int,
    adaptive_loss_weight: AdaptiveLossWeight | None = None,
) -> Path:
    checkpoint_dir = run_dir / f"checkpoint-{step:07d}"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    safetensors.torch.save_file(
        {
            key: value.detach().cpu().contiguous()
            for key, value in model.flow_lm.state_dict().items()
        },
        checkpoint_dir / "flow_lm.safetensors",
    )
    if adaptive_loss_weight is not None:
        safetensors.torch.save_file(
            {
                key: value.detach().cpu().contiguous()
                for key, value in adaptive_loss_weight.state_dict().items()
            },
            checkpoint_dir / "adaptive_loss_weight.safetensors",
        )
    torch.save(
        {
            "step": step,
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "rng_state": torch.get_rng_state(),
        },
        checkpoint_dir / "trainer_state.pt",
    )
    return checkpoint_dir


def load_checkpoint(
    model,
    optimizer,
    scheduler,
    checkpoint: Path,
    adaptive_loss_weight: AdaptiveLossWeight | None = None,
) -> int:
    model.flow_lm.load_state_dict(
        safetensors.torch.load_file(checkpoint / "flow_lm.safetensors"), strict=True
    )
    adaptive_path = checkpoint / "adaptive_loss_weight.safetensors"
    if adaptive_loss_weight is not None:
        if not adaptive_path.exists():
            raise ValueError(
                f"{checkpoint} has no adaptive LSD weight state. Use "
                "--init-flow-checkpoint to start a fresh FM/LSD optimizer instead of --resume."
            )
        adaptive_loss_weight.load_state_dict(
            safetensors.torch.load_file(adaptive_path), strict=True
        )
    state = torch.load(checkpoint / "trainer_state.pt", map_location="cpu", weights_only=True)
    optimizer.load_state_dict(state["optimizer"])
    scheduler.load_state_dict(state["scheduler"])
    torch.set_rng_state(state["rng_state"])
    return int(state["step"])


def load_flow_weights(model, checkpoint: Path) -> None:
    """Load model weights without optimizer state when changing training objectives."""
    model.flow_lm.load_state_dict(
        safetensors.torch.load_file(checkpoint / "flow_lm.safetensors"), strict=True
    )


def write_metric(path: Path, record: dict) -> None:
    """
    Append one machine-readable metric line. Console logs are for watching a run; this
    file is for comparing runs. Flushed per line so a killed run keeps everything up to
    the last write.
    """
    record = {"wall_time": time.time(), **record}
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(record) + "\n")


def ensure_safe_run_directory(run_dir: Path, *, resume: Path | None) -> None:
    """Refuse to mix a fresh experiment into a directory that already has outputs."""
    if not run_dir.exists():
        return
    entries = list(run_dir.iterdir())
    if entries and resume is None:
        preview = ", ".join(entry.name for entry in entries[:5])
        raise FileExistsError(
            f"Refusing a fresh run in non-empty {run_dir} ({preview}). "
            "Choose a new --run-dir, or pass --resume for a matching checkpoint."
        )


@torch.no_grad()
def evaluate(
    model,
    dataset,
    sample_count: int,
    args,
    device,
    adaptive_loss_weight: AdaptiveLossWeight | None = None,
) -> tuple[float, float, float, float]:
    was_training = model.flow_lm.training
    model.flow_lm.eval()
    # sample_count <= 0 means evaluate everything.
    if sample_count <= 0 or sample_count >= len(dataset):
        indices = torch.arange(len(dataset))
    else:
        indices = torch.linspace(0, len(dataset) - 1, steps=sample_count).long()
    adaptive_was_training = None
    if adaptive_loss_weight is not None:
        adaptive_was_training = adaptive_loss_weight.training
        adaptive_loss_weight.eval()
    totals = torch.zeros(4)
    for index in indices:
        # Evaluation always keeps the EOS term in the reported total, even when training
        # has frozen it, so the validation curve stays comparable across experiments.
        loss, flow_loss, lsd_loss, eos_loss = compute_loss(
            model,
            dataset[int(index)],
            args.head_samples,
            args.eos_weight,
            device,
            eos_label_frames=args.eos_label_frames,
            eos_trainable=True,
            head_batch_multiplier=args.head_batch_multiplier,
            lsd_fraction=args.lsd_fraction if args.loss_mode == "fm-lsd" else 0.0,
            adaptive_loss_weight=adaptive_loss_weight,
        )
        totals += torch.tensor([float(loss), float(flow_loss), float(lsd_loss), float(eos_loss)])
    if was_training:
        model.flow_lm.train()
    if adaptive_loss_weight is not None and adaptive_was_training:
        adaptive_loss_weight.train()
    means = totals / len(indices)
    return tuple(float(value) for value in means)


def main() -> None:
    args = parse_args()
    if args.batch_size != 1:
        raise ValueError("Variable-length batching is not implemented; use --batch-size 1")
    if args.resume and args.init_flow_checkpoint:
        raise ValueError("--resume and --init-flow-checkpoint are mutually exclusive")
    if args.loss_mode == "fm-lsd" and not 0.0 < args.lsd_fraction < 1.0:
        raise ValueError("--lsd-fraction must be strictly between 0 and 1")
    if args.save_every < 0:
        raise ValueError("--save-every must be non-negative")
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = choose_device(args.device)
    ensure_safe_run_directory(args.run_dir, resume=args.resume)
    args.run_dir.mkdir(parents=True, exist_ok=True)
    (args.run_dir / "arguments.json").write_text(
        json.dumps(vars(args), default=str, indent=2), encoding="utf-8"
    )
    # Appended to, not truncated, so a resumed run keeps the earlier curve.
    metrics_path = args.run_dir / "metrics.jsonl"

    model = TTSModel.load_model(language=args.base_language)
    require_voice_cloning_access(model)
    copied_pieces = install_tokenizer(model, args.tokenizer)
    model.mimi.requires_grad_(False)
    model.mimi.eval()
    model.to(device)
    model.flow_lm.train()
    adaptive_loss_weight = None
    if args.loss_mode == "fm-lsd":
        adaptive_loss_weight = AdaptiveLossWeight(args.adaptive_weight_channels).to(device)
        adaptive_loss_weight.train()

    dataset = LatentDataset(args.train_manifest, expected_base_language=args.base_language)
    validation_dataset = LatentDataset(
        args.validation_manifest, expected_base_language=args.base_language
    )
    if args.limit_train_samples:
        dataset.rows = dataset.rows[: args.limit_train_samples]
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=singleton_collate,
        num_workers=0,
    )

    parameters = [parameter for parameter in model.flow_lm.parameters() if parameter.requires_grad]
    if adaptive_loss_weight is not None:
        parameters.extend(adaptive_loss_weight.parameters())
    optimizer = torch.optim.AdamW(parameters, lr=args.learning_rate, weight_decay=args.weight_decay)

    def lr_lambda(step: int) -> float:
        if step < args.warmup_steps:
            return max(step, 1) / max(args.warmup_steps, 1)
        progress = (step - args.warmup_steps) / max(args.steps - args.warmup_steps, 1)
        return 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * min(progress, 1.0)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    start_step = 0
    if args.resume:
        start_step = load_checkpoint(
            model, optimizer, scheduler, args.resume, adaptive_loss_weight=adaptive_loss_weight
        )
    elif args.init_flow_checkpoint:
        load_flow_weights(model, args.init_flow_checkpoint)

    amp_dtype = (
        torch.bfloat16
        if device.type == "cuda" and torch.cuda.is_bf16_supported()
        else torch.float16
    )
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp and amp_dtype == torch.float16)
    best_val_eos = float("inf")
    best_val_eos_step = 0
    eos_regressions = 0
    iterator = iter(loader)
    optimizer.zero_grad(set_to_none=True)
    print(
        f"device={device} samples={len(dataset)} copied_token_pieces={copied_pieces} "
        f"trainable_parameters={sum(p.numel() for p in parameters):,} "
        f"loss_mode={args.loss_mode} head_batch_multiplier={args.head_batch_multiplier}"
    )

    for step in range(start_step + 1, args.steps + 1):
        total = flow = lsd = eos = 0.0
        for _ in range(args.gradient_accumulation):
            try:
                item = next(iterator)
            except StopIteration:
                iterator = iter(loader)
                item = next(iterator)
            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                loss, flow_loss, lsd_loss, eos_loss = compute_loss(
                    model,
                    item,
                    args.head_samples,
                    args.eos_weight,
                    device,
                    eos_label_frames=args.eos_label_frames,
                    eos_trainable=(args.freeze_eos_after <= 0 or step <= args.freeze_eos_after),
                    head_batch_multiplier=args.head_batch_multiplier,
                    lsd_fraction=(args.lsd_fraction if args.loss_mode == "fm-lsd" else 0.0),
                    adaptive_loss_weight=adaptive_loss_weight,
                )
                scaled_loss = loss / args.gradient_accumulation
            scaler.scale(scaled_loss).backward()
            total += float(loss.detach())
            flow += float(flow_loss)
            lsd += float(lsd_loss)
            eos += float(eos_loss)

        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(parameters, max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)
        scheduler.step()

        if step % args.log_every == 0 or step == 1:
            divisor = args.gradient_accumulation
            print(
                f"step={step} loss={total / divisor:.5f} flow={flow / divisor:.5f} "
                f"lsd={lsd / divisor:.5f} eos={eos / divisor:.5f} "
                f"lr={scheduler.get_last_lr()[0]:.3e}"
            )
            write_metric(
                metrics_path,
                {
                    "step": step,
                    "split": "train",
                    "loss": total / divisor,
                    "flow": flow / divisor,
                    "lsd": lsd / divisor,
                    "eos": eos / divisor,
                    "lr": scheduler.get_last_lr()[0],
                },
            )
        if args.eval_every > 0 and (step % args.eval_every == 0 or step == args.steps):
            val_loss, val_flow, val_lsd, val_eos = evaluate(
                model,
                validation_dataset,
                args.eval_samples,
                args,
                device,
                adaptive_loss_weight=adaptive_loss_weight,
            )
            print(
                f"validation step={step} loss={val_loss:.5f} "
                f"flow={val_flow:.5f} lsd={val_lsd:.5f} eos={val_eos:.5f}"
            )
            write_metric(
                metrics_path,
                {
                    "step": step,
                    "split": "validation",
                    "loss": val_loss,
                    "flow": val_flow,
                    "lsd": val_lsd,
                    "eos": val_eos,
                    "eval_samples": min(args.eval_samples, len(validation_dataset))
                    if args.eval_samples > 0
                    else len(validation_dataset),
                },
            )
            # Early stop on the EOS term specifically. The flow term keeps improving long
            # after EOS has turned, so a stop rule on total loss would not fire. Only arms
            # after --eos-early-stop-min-step, because EOS is noisy early on.
            if args.eos_early_stop_patience > 0 and step >= args.eos_early_stop_min_step:
                if val_eos < best_val_eos:
                    best_val_eos = val_eos
                    best_val_eos_step = step
                    eos_regressions = 0
                elif val_eos > best_val_eos * args.eos_early_stop_factor:
                    eos_regressions += 1
                    print(
                        f"eos regression {eos_regressions}/{args.eos_early_stop_patience}"
                        f" (eos={val_eos:.5f} vs best {best_val_eos:.5f} @ {best_val_eos_step})"
                    )
                    if eos_regressions >= args.eos_early_stop_patience:
                        print(
                            f"EARLY STOP at step {step}: validation eos failed to improve "
                            f"on {best_val_eos:.5f} @ step {best_val_eos_step}"
                        )
                        write_metric(
                            metrics_path,
                            {
                                "step": step,
                                "split": "early_stop",
                                "loss": val_loss,
                                "flow": val_flow,
                                "eos": val_eos,
                                "best_eos": best_val_eos,
                                "best_eos_step": best_val_eos_step,
                            },
                        )
                        save_checkpoint(
                            model,
                            optimizer,
                            scheduler,
                            args.run_dir,
                            step,
                            adaptive_loss_weight=adaptive_loss_weight,
                        )
                        break
                else:
                    eos_regressions = 0
        periodic_save = args.save_every > 0 and step % args.save_every == 0
        final_save = step == args.steps and not args.skip_final_checkpoint
        if periodic_save or final_save:
            checkpoint = save_checkpoint(
                model,
                optimizer,
                scheduler,
                args.run_dir,
                step,
                adaptive_loss_weight=adaptive_loss_weight,
            )
            print(f"saved={checkpoint}")


if __name__ == "__main__":
    main()
