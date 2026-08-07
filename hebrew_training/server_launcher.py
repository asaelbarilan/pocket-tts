"""Portable, guarded launcher for Hebrew student and teacher experiments."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path


def model_slug(base_language: str) -> str:
    return base_language.replace("/", "-").replace("\\", "-")


def validate_model_kind(kind: str, base_language: str) -> None:
    is_teacher = base_language.endswith("_24l")
    if kind == "teacher" and not is_teacher:
        raise ValueError("Teacher runs require a released *_24l --base-language.")
    if kind == "student" and is_teacher:
        raise ValueError("Student runs require a 6-layer --base-language, normally english.")


def artifact_paths(artifacts_dir: Path, base_language: str) -> dict[str, Path]:
    slug = model_slug(base_language)
    return {
        "train": artifacts_dir / "train.jsonl",
        "validation": artifacts_dir / "validation.jsonl",
        "tokenizer": artifacts_dir / "tokenizer.model",
        "train_latents": artifacts_dir / f"train_latents_{slug}.jsonl",
        "validation_latents": artifacts_dir / f"validation_latents_{slug}.jsonl",
        "train_cache": artifacts_dir / "latents" / slug / "train",
        "validation_cache": artifacts_dir / "latents" / slug / "validation",
    }


def precompute_commands(args: argparse.Namespace) -> list[list[str]]:
    paths = artifact_paths(args.artifacts_dir, args.base_language)
    commands = []
    for split in ("train", "validation"):
        command = [
            sys.executable,
            "-m",
            "hebrew_training.precompute_latents",
            "--manifest",
            str(paths[split]),
            "--output-dir",
            str(paths[f"{split}_cache"]),
            "--output-manifest",
            str(paths[f"{split}_latents"]),
            "--base-language",
            args.base_language,
            "--device",
            args.device,
        ]
        if args.limit:
            command.extend(["--limit", str(args.limit)])
        commands.append(command)
    return commands


def training_command(args: argparse.Namespace) -> list[str]:
    validate_model_kind(args.kind, args.base_language)
    paths = artifact_paths(args.artifacts_dir, args.base_language)
    head_samples = args.head_samples or (500 if args.kind == "teacher" else 128)
    command = [
        sys.executable,
        "-m",
        "hebrew_training.train",
        "--train-manifest",
        str(paths["train_latents"]),
        "--validation-manifest",
        str(paths["validation_latents"]),
        "--tokenizer",
        str(paths["tokenizer"]),
        "--run-dir",
        str(args.run_dir),
        "--base-language",
        args.base_language,
        "--device",
        args.device,
        "--loss-mode",
        "flow",
        "--steps",
        str(args.steps),
        "--gradient-accumulation",
        str(args.gradient_accumulation),
        "--head-samples",
        str(head_samples),
        "--head-batch-multiplier",
        str(args.head_batch_multiplier),
        "--eval-every",
        str(args.eval_every),
        "--eval-samples",
        str(args.eval_samples),
        "--save-every",
        str(args.save_every),
    ]
    if args.smoke:
        command[command.index("--steps") + 1] = "30"
        for flag, value in (
            ("--gradient-accumulation", "1"),
            ("--eval-every", "10"),
            ("--eval-samples", "8"),
            ("--save-every", "0"),
        ):
            command[command.index(flag) + 1] = value
        command.extend(["--limit-train-samples", "16", "--skip-final-checkpoint"])
    return command


def missing_inputs(args: argparse.Namespace, *, require_latents: bool) -> list[Path]:
    paths = artifact_paths(args.artifacts_dir, args.base_language)
    keys = ["train", "validation", "tokenizer"]
    if require_latents:
        keys.extend(["train_latents", "validation_latents"])
    return [paths[key] for key in keys if not paths[key].is_file()]


def run_command(command: list[str], log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8", newline="\n") as log:
        log.write("\n$ " + subprocess.list2cmdline(command) + "\n")
        process = subprocess.Popen(
            command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="")
            log.write(line)
            log.flush()
        if process.wait() != 0:
            raise subprocess.CalledProcessError(process.returncode, command)


def print_commands(commands: list[list[str]]) -> None:
    for command in commands:
        print(subprocess.list2cmdline(command))


def doctor(args: argparse.Namespace) -> None:
    validate_model_kind(args.kind, args.base_language)
    missing = missing_inputs(args, require_latents=args.require_latents)
    disk = shutil.disk_usage(args.artifacts_dir.resolve().anchor or ".")
    report = {
        "python": sys.version.split()[0],
        "kind": args.kind,
        "base_language": args.base_language,
        "artifacts_dir": str(args.artifacts_dir.resolve()),
        "missing": [str(path) for path in missing],
        "free_disk_gb": round(disk.free / 1024**3, 2),
    }
    try:
        import torch

        report["torch"] = torch.__version__
        report["cuda_available"] = torch.cuda.is_available()
        report["gpu"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
        if torch.cuda.is_available():
            report["gpu_memory_gb"] = round(
                torch.cuda.get_device_properties(0).total_memory / 1024**3, 2
            )
    except ImportError:
        report["torch"] = None
        report["cuda_available"] = False
    print(json.dumps(report, indent=2))
    if missing:
        raise SystemExit(2)
    if not report["cuda_available"]:
        raise SystemExit("CUDA PyTorch is not available in this environment.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    def add_artifact_args(subparser: argparse.ArgumentParser) -> None:
        subparser.add_argument("--artifacts-dir", type=Path, required=True)
        subparser.add_argument("--base-language", default="english")

    doctor_parser = subparsers.add_parser("doctor", help="Validate environment and inputs.")
    add_artifact_args(doctor_parser)
    doctor_parser.add_argument("--kind", choices=("student", "teacher"), required=True)
    doctor_parser.add_argument("--require-latents", action="store_true")

    precompute_parser = subparsers.add_parser("precompute", help="Build model-specific latents.")
    add_artifact_args(precompute_parser)
    precompute_parser.add_argument("--device", default="cuda")
    precompute_parser.add_argument("--limit", type=int)
    precompute_parser.add_argument("--log-dir", type=Path, default=Path("logs"))
    precompute_parser.add_argument("--execute", action="store_true")

    train_parser = subparsers.add_parser("train", help="Smoke-test or train one model family.")
    add_artifact_args(train_parser)
    train_parser.add_argument("--kind", choices=("student", "teacher"), required=True)
    train_parser.add_argument("--run-dir", type=Path, required=True)
    train_parser.add_argument("--device", default="cuda")
    train_parser.add_argument("--steps", type=int, default=12_000)
    train_parser.add_argument("--gradient-accumulation", type=int, default=8)
    train_parser.add_argument("--head-samples", type=int)
    train_parser.add_argument("--head-batch-multiplier", type=int, default=8)
    train_parser.add_argument("--eval-every", type=int, default=250)
    train_parser.add_argument("--eval-samples", type=int, default=64)
    train_parser.add_argument("--save-every", type=int, default=3000)
    train_parser.add_argument("--log-dir", type=Path, default=Path("logs"))
    train_parser.add_argument("--smoke", action="store_true")
    train_parser.add_argument("--execute", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "doctor":
        doctor(args)
        return
    if args.command == "precompute":
        commands = precompute_commands(args)
        missing = missing_inputs(args, require_latents=False)
        if missing:
            raise FileNotFoundError("Missing inputs: " + ", ".join(map(str, missing)))
        if not args.execute:
            print_commands(commands)
            print("Dry run only. Add --execute to run these commands.")
            return
        for index, command in enumerate(commands, start=1):
            run_command(
                command, args.log_dir / f"precompute-{model_slug(args.base_language)}-{index}.log"
            )
        return
    command = training_command(args)
    missing = missing_inputs(args, require_latents=True)
    if missing:
        raise FileNotFoundError("Missing inputs: " + ", ".join(map(str, missing)))
    if not args.execute:
        print_commands([command])
        print("Dry run only. Add --execute to start training.")
        return
    run_command(command, args.log_dir / f"{args.run_dir.name}.log")


if __name__ == "__main__":
    main()
