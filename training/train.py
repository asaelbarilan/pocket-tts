"""Train a CALM-style TTS on top of the pocket-tts modules.

Usage:
    torchrun --nproc-per-node 8 -m training.train training/configs/scratch.yaml
"""

import os

# The package-wide beartype claw must be disabled before pocket_tts is first
# imported: dynamo cannot trace the beartype wrappers and `compile` defaults
# to on. Export POCKET_TTS_NO_BEARTYPE=0 to force type checking back on
# (requires `compile: false`).
os.environ.setdefault("POCKET_TTS_NO_BEARTYPE", "1")

# Utterances are variable-length, so the default caching allocator fragments and
# a batch that fits can still fail to allocate; expandable segments keep the
# reserved memory flat at no throughput cost. Set only from the entry point,
# and before torch is imported: the allocator reads it when it initializes.
if __name__ == "__main__":
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import logging
import signal
import sys
import time
from contextlib import closing
from dataclasses import dataclass
from pathlib import Path

import torch
import wandb
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP

from training.args import TrainArgs, dump_args, load_args, save_args
from training.checkpointing import EMA, latest_checkpoint, load_checkpoint, save_checkpoint
from training.curriculum import (
    CurriculumLoader,
    MixSchedule,
    Stream,
    combine_validation,
    count_lines,
    plan_validation,
)
from training.dataloader import DataLoader, encode_batch
from training.distributed import (
    any_across_ranks,
    avg_across_ranks,
    get_rank,
    get_world_size,
    init_distributed,
    is_torchrun,
    shutdown_distributed,
)
from training.modules.builders import build_models
from training.train_utils import (
    ProgressLog,
    _compile_models,
    add_file_logging,
    git_commit,
    lr_at,
    optimizer_param_groups,
    setup_logging,
    write_samples,
)

logger = logging.getLogger("train")

VERBOSE_STEPS = 10  # log every step at the start of a run, then every log_freq


class GracefulStop:
    """Defer termination until the current optimizer step is complete."""

    def __init__(self) -> None:
        self.requested = False
        self._seen: set[int] = set()

    def __call__(self, signum: int, _frame) -> None:
        if signum in self._seen:
            signal.signal(signum, signal.SIG_DFL)
            os.kill(os.getpid(), signum)
        self._seen.add(signum)
        self.requested = True


@dataclass
class Run:
    """Everything the training loop needs, assembled before the first step."""

    args: TrainArgs
    model: nn.Module  # unwrapped, for EMA/checkpointing
    wrapped: nn.Module  # DDP-wrapped when distributed, else the model itself
    mimi: nn.Module
    optimizer: torch.optim.Optimizer
    ema: EMA | None
    start_step: int
    device: torch.device
    rank: int
    world_size: int
    progress: ProgressLog


def setup(config_path: str) -> Run:
    """Resolve the config, build the models, restore any checkpoint."""
    setup_logging()
    torch.backends.cuda.matmul.allow_tf32 = True  # fp32 islands (Mimi encode)
    args = load_args(config_path)
    device = init_distributed()
    rank, world_size = get_rank(), get_world_size()
    # All ranks must construct identical parameters so DDP can skip its very large
    # initial parameter broadcast. Rank-specific randomness is restored below.
    torch.manual_seed(args.seed)
    run_dir = args.run_dir
    log_path = add_file_logging(run_dir, rank)
    progress = ProgressLog(run_dir / "progress.jsonl", enabled=rank == 0)
    if rank == 0:
        logger.info(f"logging to {log_path}")
        gpu = torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu"
        logger.info(f"torch {torch.__version__} | {device} ({gpu}) | world size {world_size}")
        logger.info(f"resolved config from {config_path}:\n{dump_args(args).rstrip()}")

    manifests = [("train_jsonl", args.data.train_jsonl), ("valid_jsonl", args.data.valid_jsonl)]
    curriculum = args.data.resolve_curriculum()
    if curriculum is not None:
        manifests = [
            (f"datasets.{name}.train_jsonl", curriculum.datasets[name].train_jsonl)
            for name in curriculum.train_names
        ] + [
            (f"datasets.{name}.valid_jsonl", curriculum.datasets[name].valid_jsonl)
            for name in curriculum.valid_names
        ]
    for name, path in manifests:
        if path and not Path(path).exists():
            raise SystemExit(
                f"{name} not found: {path}\n"
                "Prepare a dataset first, e.g.:\n"
                "    python -m training.scripts.prepare_data --hours 200\n"
                "or point data.train_jsonl / data.valid_jsonl (or data.datasets) at your own "
                "manifests."
            )

    if rank == 0:
        save_args(args, run_dir / "args.yaml")
        wandb_kwargs = {
            "project": os.environ.get("WANDB_PROJECT", "pocket-tts"),
            "name": os.environ.get("WANDB_NAME", run_dir.name),
            "dir": str(run_dir),
            "config": {
                "config_path": config_path,
                "resolved_config": dump_args(args),
                "world_size": world_size,
                "torch_version": torch.__version__,
                "git_commit": git_commit(),
            },
        }
        if entity := os.environ.get("WANDB_ENTITY"):
            wandb_kwargs["entity"] = entity
        if run_id := os.environ.get("WANDB_RUN_ID"):
            wandb_kwargs.update(id=run_id, resume="allow")
        wandb.init(**wandb_kwargs)

    model, mimi, _config = build_models(args)
    model.to(device)
    mimi.to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if rank == 0:
        logger.info(f"flow_lm + objective: {n_params / 1e6:.1f}M trainable params")

    optimizer = torch.optim.AdamW(
        optimizer_param_groups(model, args),
        lr=args.optim.lr,
        betas=args.optim.betas,
        eps=args.optim.eps,
        weight_decay=args.optim.weight_decay,
        fused=device.type == "cuda",
    )
    ema = EMA(model, args.ema_decay) if args.ema_decay > 0 else None

    start_step = 0
    ckpt = latest_checkpoint(run_dir)
    if ckpt is not None:
        start_step = load_checkpoint(ckpt, model, optimizer, ema)
    progress.log(
        "run_start",
        start_step,
        config=config_path,
        world_size=world_size,
        max_steps=args.max_steps,
        commit=git_commit(),
    )

    wrapped: nn.Module = model
    if is_torchrun():
        if rank == 0:
            logger.info(f"initializing DDP for the {n_params / 1e6:.1f}M-parameter model")
        # Every rank used the same seed while building the model above. Avoid DDP's
        # multi-gigabyte init broadcast, which is excessively slow on this host.
        wrapped = DDP(
            model,
            device_ids=[device.index],
            find_unused_parameters=not args.compile,
            init_sync=False,
        )
        if rank == 0:
            logger.info("DDP initialization complete")
    if args.compile:
        if rank == 0:
            logger.info("compiling model kernels; this can take several minutes before step 1")
        _compile_models(model, mimi)
        if rank == 0:
            logger.info("model compilation complete; entering training loop")
    torch.manual_seed(args.seed + rank)

    return Run(
        args=args,
        model=model,
        wrapped=wrapped,
        mimi=mimi,
        optimizer=optimizer,
        ema=ema,
        start_step=start_step,
        device=device,
        rank=rank,
        world_size=world_size,
        progress=progress,
    )


def main(config_path: str) -> None:
    run = setup(config_path)
    stop = GracefulStop()
    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)
    # Unpacked for the hot loop; the thin ones stay run.* at their call sites.
    args, model, mimi = run.args, run.model, run.mimi
    optimizer, ema, device, rank = run.optimizer, run.ema, run.device, run.rank
    progress, start_step = run.progress, run.start_step

    tokenize = model.flow_lm.conditioner.tokenizer.sp.encode
    # Semantic conditioning reads each row's text; otherwise the forward call is unchanged.
    semantic = getattr(model.flow_lm, "semantic_fusion", None) is not None
    curriculum = args.data.resolve_curriculum()
    mixed: CurriculumLoader | None = None
    valid_plan: dict[str, int] = {}
    current_phase = -1
    if curriculum is None:
        train_loader = iter(
            make_loader(
                args.data.train_jsonl,
                args,
                tokenize,
                mimi,
                rank,
                run.world_size,
                seed=args.seed + rank,
                shuffle=args.data.shuffle,
            )
        )

        def next_batch(step: int, micro: int):
            return next(train_loader)

        has_valid = bool(args.data.valid_jsonl)
    else:
        schedule = MixSchedule(curriculum, args.grad_accum_steps * run.world_size)
        schedule.fast_forward(start_step)

        def open_stream(name: str) -> Stream:
            loader = make_loader(
                curriculum.datasets[name].train_jsonl,
                args,
                tokenize,
                mimi,
                rank,
                run.world_size,
                seed=args.seed + rank,
                shuffle=args.data.shuffle,
            )
            return iter(loader), len(loader.entries)

        mixed = CurriculumLoader(schedule, open_stream, rank, run.world_size, args.batch_size)
        next_batch = mixed.next_batch
        valid_plan = plan_validation(
            curriculum,
            {
                name: count_lines(curriculum.datasets[name].valid_jsonl)
                for name in curriculum.valid_names
            },
            args.batch_size,
            run.world_size,
            args.num_valid_batches,
        )
        has_valid = bool(valid_plan)
        if rank == 0:
            logger.info(f"curriculum, {curriculum.total_steps} steps:\n{curriculum.describe()}")
            if valid_plan:
                shown = ", ".join(f"{n}: {b}" for n, b in valid_plan.items())
                logger.info(f"validation benchmark, batches of {args.batch_size} per rank: {shown}")
        if start_step and rank == 0:
            logger.info(
                f"resuming at step {start_step}, in phase {curriculum.phase_index(start_step) + 1}"
                ": the mix continues, but each dataset's shuffle restarts from its beginning"
            )

    def close_train_loaders():
        if mixed is not None:
            mixed.close()
        else:
            train_loader.close()

    autocast = torch.autocast(
        device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"
    )
    model.train()
    if rank == 0:
        logger.info("starting training loop, first step can take a few minutes (compilation etc.)")
    last_log = time.time()
    steps_since_log = 0
    sample_voice = None
    for step in range(start_step, args.max_steps):
        first_step = step == start_step
        if rank == 0 and first_step:
            logger.info(f"starting optimizer step {step + 1}/{args.max_steps}; loading first batch")
        step_start = time.time()
        lr = lr_at(step, args)
        for group in optimizer.param_groups:
            group["lr"] = lr * group.get("lr_scale", 1.0)
        if curriculum is not None and curriculum.phase_index(step) != current_phase:
            current_phase = curriculum.phase_index(step)
            mix = curriculum.phases[current_phase].mix
            if rank == 0:
                shown = ", ".join(f"{name} {share:.1%}" for name, share in mix.items())
                logger.info(
                    f"curriculum phase {current_phase + 1}/{len(curriculum.phases)} "
                    f"from step {step + 1}: {shown}"
                )
                progress.log("phase", step + 1, phase=current_phase + 1, mix=mix)
        optimizer.zero_grad()
        for micro in range(args.grad_accum_steps):
            batch = next_batch(step, micro)
            if rank == 0 and first_step and micro == 0:
                logger.info(
                    "first batch loaded; running forward/backward (CUDA compilation may continue)"
                )
            latents, mask, voice_prompt_latents, num_voice_prompt_frames = encode_batch(
                mimi, batch, device
            )
            with autocast:
                loss, metrics = run.wrapped(
                    latents,
                    mask,
                    batch.text_tokens,
                    voice_prompt_latents,
                    update_stats=step < args.stats_ema_steps,
                    num_voice_prompt_frames=num_voice_prompt_frames,
                    **({"texts": batch.texts} if semantic else {}),
                )
            # Under DDP, allreduce only on the last micro-batch.
            last_micro = micro == args.grad_accum_steps - 1
            scaled = loss / args.grad_accum_steps
            if not last_micro and hasattr(run.wrapped, "no_sync"):
                with run.wrapped.no_sync():
                    scaled.backward()
            else:
                scaled.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.optim.max_norm)
        optimizer.step()
        if ema is not None:
            ema.update(model)
        if rank == 0 and first_step:
            logger.info(f"first optimizer step complete in {time.time() - step_start:.0f}s")

        if any_across_ranks(stop.requested, device):
            completed_step = step + 1
            if rank == 0:
                logger.info(
                    f"stop requested; saving checkpoint after completed step {completed_step}"
                )
                save_checkpoint(
                    args.run_dir, completed_step, model, optimizer, ema, args.num_ckpt_keep, mimi
                )
                progress.log("checkpoint", completed_step, interrupted=True)
                logger.info("graceful shutdown complete")
            close_train_loaders()
            shutdown_distributed()
            if rank == 0:
                wandb.finish()
            return

        steps_since_log += 1
        if rank == 0 and step == start_step and args.compile:
            logger.info(f"first step took {time.time() - step_start:.0f}s, including compilation")

        verbose = step - start_step < VERBOSE_STEPS
        if rank == 0 and (verbose or (step + 1) % args.log_freq == 0):
            now = time.time()
            speed = steps_since_log / (now - last_log)
            last_log, steps_since_log = now, 0
            values = {k: v.item() for k, v in metrics.items() if v.numel() == 1}
            shown = {k: f"{v:.4f}" for k, v in values.items()}
            logger.info(
                f"step {step + 1} | lr {lr:.2e} | grad {grad_norm:.2f} | {speed:.2f} it/s | {shown}"
            )
            # Passes over each dataset so far (this rank's shard): whole numbers are rollovers.
            epochs = {} if mixed is None else mixed.epochs()
            extra = {} if mixed is None else {"phase": current_phase + 1, "epochs": epochs}
            progress.log(
                "train", step + 1, values, lr=lr, grad_norm=grad_norm.item(), it_s=speed, **extra
            )
            wandb.log(
                {
                    **{f"train/{key}": value for key, value in values.items()},
                    "train/lr": lr,
                    "train/grad_norm": grad_norm.item(),
                    "train/iterations_per_second": speed,
                    **({"train/phase": current_phase + 1} if mixed is not None else {}),
                    **{f"data/epochs/{name}": value for name, value in epochs.items()},
                },
                step=step + 1,
            )
        if rank == 0 and step - start_step == VERBOSE_STEPS - 1:
            logger.info(f"per-step logging done, logging every {args.log_freq} steps from now on")

        if sample_voice is None:
            sample_voice = voice_prompt_latents[0].detach().clone()
        if (
            rank == 0
            and args.sample_sentences
            and args.sample_freq > 0
            and (step + 1) % args.sample_freq == 0
        ):
            samples = write_samples(
                model, mimi, tokenize, args, args.run_dir, step + 1, sample_voice, device
            )
            if samples:
                wandb.log(
                    {
                        "samples": wandb.Table(
                            columns=["text", "audio"],
                            data=[
                                [text, wandb.Audio(str(path), sample_rate=mimi.sample_rate)]
                                for text, path in samples
                            ],
                        )
                    },
                    step=step + 1,
                )

        # A phase's end is also evaluated and saved: with a fixed benchmark that is the
        # cleanest read on what the phase did to every dataset.
        phase_end = curriculum is not None and (step + 1) in curriculum.boundaries
        if ((step + 1) % args.valid_freq == 0 or phase_end) and has_valid:
            valid_metrics = validate(
                model, mimi, args, device, rank, run.world_size, step + 1, valid_plan
            )
            progress.log("valid", step + 1, valid_metrics)
            if rank == 0:
                wandb.log(
                    {f"valid/{key}": value for key, value in valid_metrics.items()}, step=step + 1
                )
            model.train()

        # The last step is saved after the loop, whatever the reason it is a boundary.
        save_phase_end = (
            phase_end and args.data.curriculum_checkpoint_at_phase_end and step + 1 < args.max_steps
        )
        if rank == 0 and ((step + 1) % args.ckpt_freq == 0 or save_phase_end):
            save_checkpoint(args.run_dir, step + 1, model, optimizer, ema, args.num_ckpt_keep, mimi)
            progress.log("checkpoint", step + 1)

    if rank == 0:
        save_checkpoint(
            args.run_dir, args.max_steps, model, optimizer, ema, args.num_ckpt_keep, mimi
        )
        progress.log("checkpoint", args.max_steps)
        logger.info("done")
        wandb.finish()
    close_train_loaders()
    shutdown_distributed()


def make_loader(
    jsonl: str,
    args: TrainArgs,
    tokenize,
    mimi,
    rank: int,
    world_size: int,
    *,
    seed: int,
    shuffle: bool,
    fixed_draws: bool = False,
) -> DataLoader:
    return DataLoader(
        jsonl,
        tokenize,
        args.batch_size,
        mimi.sample_rate,
        mimi.frame_rate,
        args.data.max_duration_sec,
        args.data.max_voice_prompt_sec,
        rank,
        world_size,
        seed=seed,
        shuffle=shuffle,
        alt_words_sample_prob=args.data.alt_words_sample_prob,
        alt_word_swap_prob=args.data.alt_word_swap_prob,
        nikud_word_prob=args.data.nikud_word_prob,
        fixed_draws=fixed_draws,
    )


@torch.no_grad()
def _valid_pass(
    model, mimi, args: TrainArgs, device, rank: int, world_size: int, jsonl: str, num_batches: int
) -> dict[str, float]:
    """Mean of each metric over the first `num_batches` batches of `jsonl`, averaged over ranks."""
    tokenize = model.flow_lm.conditioner.tokenizer.sp.encode
    loader = iter(
        make_loader(
            jsonl,
            args,
            tokenize,
            mimi,
            rank,
            world_size,
            seed=0,
            shuffle=False,
            # Same digit/niqqud draws for a row on every evaluation, so scores compare.
            fixed_draws=True,
        )
    )
    autocast = torch.autocast(
        device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"
    )
    totals: dict[str, float] = {}
    n = 0
    # The loader is endless and we stop after num_batches: closing it stops its
    # prefetch thread and I/O pool instead of leaving them blocked for the rest of the run.
    with closing(loader):
        for _ in range(num_batches):
            try:
                batch = next(loader)
            except StopIteration:
                break
            latents, mask, voice_prompt_latents, num_voice_prompt_frames = encode_batch(
                mimi, batch, device
            )
            semantic = getattr(model.flow_lm, "semantic_fusion", None) is not None
            with autocast:
                _, metrics = model(
                    latents,
                    mask,
                    batch.text_tokens,
                    voice_prompt_latents,
                    num_voice_prompt_frames=num_voice_prompt_frames,
                    **({"texts": batch.texts} if semantic else {}),
                )
            for k, v in metrics.items():
                if v.numel() == 1:
                    totals[k] = totals.get(k, 0.0) + v.item()
            n += 1
    return {k: avg_across_ranks(v / max(1, n)) for k, v in totals.items()}


@torch.no_grad()
def validate(
    model,
    mimi,
    args: TrainArgs,
    device,
    rank: int,
    world_size: int,
    step: int,
    plan: dict[str, int] | None = None,
) -> dict[str, float]:
    """Validation metrics. With a curriculum, `plan` is the number of batches per dataset:
    the plain keys are the whole benchmark and `<dataset>/<key>` are each dataset's."""
    model.eval()
    if plan:
        curriculum = args.data.resolve_curriculum()
        per_dataset = {
            name: _valid_pass(
                model,
                mimi,
                args,
                device,
                rank,
                world_size,
                curriculum.datasets[name].valid_jsonl,
                batches,
            )
            for name, batches in plan.items()
        }
        averaged = combine_validation(per_dataset, plan)
    else:
        averaged = _valid_pass(
            model,
            mimi,
            args,
            device,
            rank,
            world_size,
            args.data.valid_jsonl,
            args.num_valid_batches,
        )
    if rank == 0:
        shown = {k: f"{v:.4f}" for k, v in averaged.items()}
        logger.info(f"valid @ step {step}: {shown}")
    return averaged


if __name__ == "__main__":
    assert len(sys.argv) == 2, "usage: python -m training.train <config.yaml>"
    main(sys.argv[1])
