"""Evaluate official trainer checkpoints with Hebrew text and optional ivrit.ai Whisper.

This uses the same checkpoint loader, batched generation path, and Mimi decoder as
``training.eval.librispeech``. The run's ``args.yaml`` supplies the model config and
tokenizer, so no export step is needed.

Example:
    python -m training.eval.ivritai runs/finetune_hebrew \
        --sentences eval_data/hebrew_sentences.txt \
        --voices eval_data/voice_prompt.wav \
        --checkpoints /data/runs/finetune_hebrew/checkpoint_00002000.pt \
                      /data/runs/finetune_hebrew/checkpoint_00004000.pt

Add --devices auto to evaluate checkpoints concurrently on all visible CUDA GPUs,
or --devices cuda:0 cuda:1 for a subset. --list-devices prints the device inventory
as JSON without loading any TTS/ASR models. The parent owns results.json and
evaluation.json; workers write audio and logs into isolated checkpoint folders.
"""

from __future__ import annotations

import argparse
import json
import logging
import multiprocessing
import queue
import re
import signal
import traceback
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from datetime import datetime, timezone
from pathlib import Path

import torch

logger = logging.getLogger("eval_ivritai")
DEFAULT_ASR = "ivrit-ai/whisper-large-v3-turbo-ct2"


def read_sentences(path: Path) -> list[str]:
    sentences = [line.strip() for line in path.read_text(encoding="utf-8").splitlines()]
    sentences = [sentence for sentence in sentences if sentence]
    if not sentences:
        raise ValueError(f"no non-empty sentences in {path}")
    return sentences


def checkpoint_tag(checkpoint: Path, step: int, use_ema: bool) -> str:
    run = re.sub(r"[^A-Za-z0-9_.-]+", "_", checkpoint.parent.name)
    weights = "ema" if use_ema else "raw"
    return f"{run}_step{step:08d}_{weights}"


def discover_devices() -> dict:
    """JSON-friendly device inventory, respecting CUDA_VISIBLE_DEVICES."""
    devices = [{"id": "cpu", "name": "CPU", "type": "cpu", "total_memory_bytes": None}]
    if torch.cuda.is_available():
        for index in range(torch.cuda.device_count()):
            properties = torch.cuda.get_device_properties(index)
            devices.append(
                {
                    "id": f"cuda:{index}",
                    "name": properties.name,
                    "type": "cuda",
                    "total_memory_bytes": properties.total_memory,
                }
            )
    return {"schema_version": 1, "devices": devices}


def resolve_devices(device: str, devices: list[str] | None) -> list[str]:
    available = [entry["id"] for entry in discover_devices()["devices"]]
    selected = devices or [device]
    if selected == ["auto"]:
        return [name for name in available if name.startswith("cuda:")] or ["cpu"]
    selected = ["cuda:0" if name == "cuda" else name for name in selected]
    if len(set(selected)) != len(selected):
        raise ValueError("devices must be distinct (cuda and cuda:0 refer to the same GPU)")
    if len(selected) > 1 and "cpu" in selected:
        raise ValueError("select CPU alone or one or more CUDA GPUs")
    missing = [name for name in selected if name not in available]
    if missing:
        raise ValueError(
            f"unavailable device(s): {', '.join(missing)}; available: {', '.join(available)}"
        )
    return selected


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("run_dir", type=Path, nargs="?", help="Training run containing args.yaml.")
    parser.add_argument(
        "--list-devices", action="store_true", help="Print available devices as JSON and exit."
    )
    parser.add_argument("--sentences", type=Path, help="UTF-8 text file, one sentence per line.")
    parser.add_argument(
        "--voices",
        type=Path,
        nargs="+",
        help="One or more audio prompts. Every sentence is generated in every voice.",
    )
    parser.add_argument(
        "--checkpoints", type=Path, nargs="+", help="One or more checkpoint_XXXXXXXX.pt paths."
    )
    parser.add_argument("--output-dir", type=Path, help="Defaults to <run-dir>/ivritai_eval.")
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--devices",
        nargs="+",
        help="GPU worker devices, e.g. cuda:0 cuda:1; auto selects all GPUs.",
    )
    parser.add_argument(
        "--asr",
        default=DEFAULT_ASR,
        help="ASR model name; pass an empty string to disable scoring.",
    )
    parser.add_argument("--asr-compute-type", default="int8_float16")
    parser.add_argument("--beam-size", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--temp", type=float, default=0.3)
    parser.add_argument("--cfg", type=float, default=1.0)
    parser.add_argument("--n-steps", type=int, default=1)
    parser.add_argument("--eos-threshold", type=float, default=-1.0)
    parser.add_argument("--max-seconds", type=float, default=30.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--use-ema", action="store_true")
    parser.add_argument("--bootstrap-samples", type=int, default=1_000)
    args = parser.parse_args(argv)
    if not args.list_devices:
        missing = [
            name
            for name in ("run_dir", "sentences", "voices", "checkpoints")
            if not getattr(args, name)
        ]
        if missing:
            parser.error("required for evaluation: " + ", ".join(missing))
    return args


def validate_inputs(args: argparse.Namespace) -> None:
    if not (args.run_dir / "args.yaml").is_file():
        raise FileNotFoundError(f"missing training metadata: {args.run_dir / 'args.yaml'}")
    for name, paths in (("voice", args.voices), ("checkpoint", args.checkpoints)):
        missing = [str(path) for path in paths if not path.is_file()]
        if missing:
            raise FileNotFoundError(f"missing {name} file(s): {', '.join(missing)}")
    if args.batch_size < 1:
        raise ValueError("--batch-size must be positive")
    if args.max_seconds <= 0:
        raise ValueError("--max-seconds must be positive")
    if len({path.resolve() for path in args.checkpoints}) != len(args.checkpoints):
        raise ValueError("duplicate checkpoint paths")


def evaluate_checkpoint(
    checkpoint: Path, sentences: list[str], args: argparse.Namespace, asr=None, progress=None
) -> dict:
    import soundfile

    from hebrew_training.evaluation import normalize_hebrew_for_asr, score_transcripts
    from training.eval.librispeech import latents_to_wav, load_mono, load_run

    model, mimi, step = load_run(
        args.run_dir, args.device, use_ema=args.use_ema, checkpoint=checkpoint
    )
    tag = checkpoint_tag(checkpoint, step, args.use_ema)
    audio_dir = args.output_dir / tag
    audio_dir.mkdir(parents=True, exist_ok=True)
    max_frames = int(args.max_seconds * mimi.frame_rate)
    tokenize = model.flow_lm.conditioner.tokenizer.sp.encode
    if asr is not None:
        import jiwer

    with torch.no_grad():
        voices = [
            mimi.encode_to_latent(
                load_mono(str(path), mimi.sample_rate)[None, None].to(args.device)
            )[0]
            for path in args.voices
        ]

    items = [
        (voice_index, text_index, sentence)
        for voice_index in range(len(voices))
        for text_index, sentence in enumerate(sentences)
    ]
    records = []

    for start in range(0, len(items), args.batch_size):
        chunk = items[start : start + args.batch_size]
        batch_seed = args.seed + start
        torch.manual_seed(batch_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(batch_seed)
        tokens = [torch.tensor(tokenize(sentence), dtype=torch.long) for _, _, sentence in chunk]
        voice_latents = [voices[voice_index] for voice_index, _, _ in chunk]
        with torch.no_grad():
            outputs = model.generate(
                tokens,
                voice_latents,
                max_frames=max_frames,
                temp=args.temp,
                n_steps=args.n_steps,
                cfg_coef=args.cfg,
                eos_threshold=args.eos_threshold,
            )

        for (voice_index, text_index, sentence), latents in zip(chunk, outputs, strict=True):
            reference = normalize_hebrew_for_asr(sentence)
            if not reference:
                raise ValueError(f"sentence normalizes to empty text: {sentence!r}")
            no_eos = latents.shape[0] >= max_frames
            audio = latents_to_wav(mimi, latents, args.device)
            hypothesis = ""
            relative_audio = None
            seconds = 0.0
            if audio is not None:
                name = f"voice{voice_index + 1:02d}_sentence{text_index + 1:03d}.wav"
                path = audio_dir / name
                waveform = audio.float().cpu().numpy()
                soundfile.write(str(path), waveform, mimi.sample_rate)
                if asr is not None:
                    segments, _ = asr.transcribe(str(path), language="he", beam_size=args.beam_size)
                    hypothesis = normalize_hebrew_for_asr(
                        "".join(segment.text for segment in segments)
                    )
                relative_audio = path.relative_to(args.output_dir).as_posix()
                seconds = len(waveform) / mimi.sample_rate

            record = {
                "voice_index": voice_index,
                "voice": str(args.voices[voice_index].resolve()),
                "sentence_index": text_index,
                "generation_seed": batch_seed,
                "text": sentence,
                "reference": reference,
                "hypothesis": hypothesis or None,
                "audio": relative_audio,
                "seconds": round(seconds, 3),
                "silent": audio is None,
                "no_eos": bool(no_eos),
            }
            if asr is not None:
                record["wer"] = float(jiwer.wer(reference, hypothesis))
                record["cer"] = float(jiwer.cer(reference, hypothesis))
            records.append(record)
        if progress is not None:
            progress(len(records))

    result = {
        "run_dir": str(args.run_dir.resolve()),
        "checkpoint": str(checkpoint.resolve()),
        "step": step,
        "weights": "ema" if args.use_ema else "raw",
        "asr": args.asr if asr is not None else None,
        "asr_settings": (
            {"compute_type": args.asr_compute_type, "beam_size": args.beam_size}
            if asr is not None
            else None
        ),
        "generation": {
            "seed": args.seed,
            "batch_size": args.batch_size,
            "temperature": args.temp,
            "cfg": args.cfg,
            "n_steps": args.n_steps,
            "eos_threshold": args.eos_threshold,
            "max_seconds": args.max_seconds,
        },
        "silent": sum(record["silent"] for record in records),
        "no_eos": sum(record["no_eos"] for record in records),
        "records": records,
    }
    if asr is not None:
        references = [record["reference"] for record in records]
        hypotheses = [record["hypothesis"] or "" for record in records]
        result.update(
            score_transcripts(
                references, hypotheses, bootstrap_samples=args.bootstrap_samples, seed=args.seed
            )
        )
    del model, mimi, voices, voice_latents, outputs
    if str(args.device).startswith("cuda"):
        torch.cuda.empty_cache()
    return result


def timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json(path: Path, data) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(
        json.dumps(data, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )
    temporary.replace(path)


def load_asr(args: argparse.Namespace):
    if not args.asr:
        logger.info("ASR disabled; generating audio without WER/CER scoring")
        return None
    try:
        from faster_whisper import WhisperModel
    except ImportError:
        logger.warning("faster-whisper is not installed; generating audio without WER/CER scoring")
        return None
    device = torch.device(args.device)
    logger.info("loading ASR %s on %s", args.asr, args.device)
    return WhisperModel(
        args.asr,
        device=device.type,
        device_index=device.index or 0,
        compute_type=args.asr_compute_type,
    )


def _device_worker(
    device: str, args: argparse.Namespace, sentences: list[str], tasks, events
) -> None:
    """A spawned, persistent GPU worker; model weights change, ASR is reused."""
    if device.startswith("cuda:"):
        torch.cuda.set_device(device)
    asr = None
    asr_loaded = False
    while True:
        index = tasks.get()
        if index is None:
            return
        checkpoint = args.checkpoints[index]
        folder = args.output_dir / "checkpoints" / f"cp{index + 1:03d}"
        folder.mkdir(parents=True, exist_ok=True)
        worker_args = argparse.Namespace(**vars(args))
        worker_args.device = device
        worker_args.output_dir = folder
        events.put({"event": "started", "index": index, "device": device, "at": timestamp()})
        with (
            (folder / "worker.log").open("w", encoding="utf-8") as log,
            redirect_stdout(log),
            redirect_stderr(log),
        ):
            logging.basicConfig(level=logging.INFO, stream=log, force=True)
            try:
                if not asr_loaded:
                    asr = load_asr(worker_args)
                    asr_loaded = True
                result = evaluate_checkpoint(
                    checkpoint,
                    sentences,
                    worker_args,
                    asr,
                    progress=lambda count, job_index=index: events.put(
                        {"event": "progress", "index": job_index, "generated": count}
                    ),
                )
                write_json(folder / "results.json", [result])
                events.put(
                    {"event": "completed", "index": index, "result": result, "at": timestamp()}
                )
            except Exception:
                error = traceback.format_exc()
                logger.exception("checkpoint failed: %s", checkpoint)
                events.put({"event": "failed", "index": index, "error": error, "at": timestamp()})
        # A failed inference traceback may retain CUDA tensors; release them before the next task.
        if device.startswith("cuda:"):
            torch.cuda.empty_cache()


class EvaluationCancelled(Exception):
    pass


@contextmanager
def cancellation_signals():
    """Let the orchestrator clean up its workers on CLI/server termination."""

    def cancel(signum, frame):
        raise EvaluationCancelled(f"received signal {signum}")

    previous = {}
    for name in ("SIGTERM", "SIGINT", "SIGBREAK"):
        if hasattr(signal, name):
            number = getattr(signal, name)
            previous[number] = signal.signal(number, cancel)
    try:
        yield
    finally:
        for number, handler in previous.items():
            signal.signal(number, handler)


class EvaluationOutput:
    """Single writer for ordered results and auditable, live evaluation metadata."""

    def __init__(self, args: argparse.Namespace, sentences: list[str], devices: list[str]):
        self.root = args.output_dir
        self.results = {}
        self.metadata = {
            "schema_version": 1,
            "status": "running",
            "started_at": timestamp(),
            "finished_at": None,
            "run_dir": str(args.run_dir.resolve()),
            "training_args": (args.run_dir / "args.yaml").read_text(encoding="utf-8"),
            "sentences_file": str(args.sentences.resolve()),
            "sentences": sentences,
            "voices": [str(path.resolve()) for path in args.voices],
            "settings": {
                key: value
                for key, value in vars(args).items()
                if key
                not in {
                    "run_dir",
                    "sentences",
                    "voices",
                    "checkpoints",
                    "output_dir",
                    "list_devices",
                }
            },
            "devices": devices,
            "worker_count": min(len(devices), len(args.checkpoints)),
            "jobs": [
                {
                    "index": index,
                    "checkpoint": str(checkpoint.resolve()),
                    "status": "queued",
                    "device": None,
                    "started_at": None,
                    "finished_at": None,
                    "generated": 0,
                    "expected": len(sentences) * len(args.voices),
                    "output_dir": f"checkpoints/cp{index + 1:03d}",
                    "log": f"checkpoints/cp{index + 1:03d}/worker.log"
                    if len(devices) > 1
                    else None,
                }
                for index, checkpoint in enumerate(args.checkpoints)
            ],
        }
        write_json(self.root / "results.json", [])
        self.save()

    def save(self) -> None:
        write_json(self.root / "evaluation.json", self.metadata)

    def consume(self, event: dict) -> None:
        job = self.metadata["jobs"][event["index"]]
        kind = event["event"]
        if kind == "started":
            job.update(status="running", device=event["device"], started_at=event["at"])
            logger.info("%s started on %s", Path(job["checkpoint"]).name, job["device"])
        elif kind == "progress":
            job["generated"] = event["generated"]
        elif kind in {"completed", "failed"}:
            job.update(status=kind, finished_at=event["at"])
            if kind == "completed":
                result = event["result"]
                result["execution"] = {
                    "device": job["device"],
                    "started_at": job["started_at"],
                    "finished_at": job["finished_at"],
                }
                for record in result["records"]:
                    if record.get("audio"):
                        record["audio"] = f"{job['output_dir']}/{record['audio']}"
                self.results[job["index"]] = result
                job["generated"] = len(result["records"])
                write_json(
                    self.root / "results.json",
                    [self.results[index] for index in sorted(self.results)],
                )
                logger.info(
                    "%s completed on %s: WER=%s CER=%s silent=%s no_eos=%s",
                    Path(job["checkpoint"]).name,
                    job["device"],
                    result.get("wer"),
                    result.get("cer"),
                    result.get("silent"),
                    result.get("no_eos"),
                )
            else:
                job["error"] = event["error"]
                logger.error(
                    "%s failed on %s: %s",
                    Path(job["checkpoint"]).name,
                    job["device"],
                    event["error"],
                )
        self.save()

    def finish(self, status: str, error: str | None = None) -> None:
        finished = timestamp()
        self.metadata.update(status=status, finished_at=finished)
        if error:
            self.metadata["error"] = error
        for job in self.metadata["jobs"]:
            if job["status"] in {"queued", "running"}:
                job.update(status=status, finished_at=finished)
                if error:
                    job["error"] = error
        self.save()


def run_evaluation(args: argparse.Namespace, sentences: list[str], devices: list[str]) -> int:
    output = EvaluationOutput(args, sentences, devices)
    processes = []
    tasks = events = None
    try:
        if len(devices) == 1:
            # Preserve the simple in-process CLI path for one device.
            args.device = devices[0]
            if args.device.startswith("cuda:"):
                torch.cuda.set_device(args.device)
            asr = None
            asr_loaded = False
            for index, checkpoint in enumerate(args.checkpoints):
                output.consume(
                    {"event": "started", "index": index, "device": args.device, "at": timestamp()}
                )
                worker_args = argparse.Namespace(**vars(args))
                worker_args.output_dir = (
                    args.output_dir / output.metadata["jobs"][index]["output_dir"]
                )
                worker_args.output_dir.mkdir(parents=True, exist_ok=True)
                try:
                    if not asr_loaded:
                        asr = load_asr(args)
                        asr_loaded = True
                    result = evaluate_checkpoint(
                        checkpoint,
                        sentences,
                        worker_args,
                        asr,
                        progress=lambda count, job_index=index: output.consume(
                            {"event": "progress", "index": job_index, "generated": count}
                        ),
                    )
                    write_json(worker_args.output_dir / "results.json", [result])
                    output.consume(
                        {"event": "completed", "index": index, "result": result, "at": timestamp()}
                    )
                except EvaluationCancelled:
                    raise
                except Exception:
                    logger.exception("checkpoint failed: %s", checkpoint)
                    output.consume(
                        {
                            "event": "failed",
                            "index": index,
                            "error": traceback.format_exc(),
                            "at": timestamp(),
                        }
                    )
        else:
            context = multiprocessing.get_context("spawn")
            tasks, events = context.Queue(), context.Queue()
            active_devices = devices[: output.metadata["worker_count"]]
            for index in range(len(args.checkpoints)):
                tasks.put(index)
            for _ in active_devices:
                tasks.put(None)
            for device in active_devices:
                process = context.Process(
                    target=_device_worker, args=(device, args, sentences, tasks, events)
                )
                process.start()
                processes.append(process)
            remaining = len(args.checkpoints)
            while remaining:
                try:
                    event = events.get(timeout=0.2)
                except queue.Empty:
                    crashed = [
                        process for process in processes if process.exitcode not in (None, 0)
                    ]
                    if crashed:
                        raise RuntimeError(
                            f"evaluation worker exited unexpectedly: {[process.exitcode for process in crashed]}"
                        )
                    if all(process.exitcode is not None for process in processes):
                        raise RuntimeError(
                            "evaluation workers exited without returning all checkpoint results"
                        )
                    continue
                output.consume(event)
                if event["event"] in {"completed", "failed"}:
                    remaining -= 1
        failed = any(job["status"] == "failed" for job in output.metadata["jobs"])
        output.finish("failed" if failed else "completed")
        return 1 if failed else 0
    except (EvaluationCancelled, KeyboardInterrupt):
        output.finish("cancelled")
        return 130
    except Exception:
        error = traceback.format_exc()
        logger.exception("evaluation failed")
        output.finish("failed", error)
        return 1
    finally:
        for process in processes:
            if process.is_alive():
                process.terminate()
        for process in processes:
            process.join(timeout=5)
            if process.is_alive():
                process.kill()
                process.join()
        for channel in (tasks, events):
            if channel is not None:
                channel.cancel_join_thread()
                channel.close()


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s %(levelname)s %(name)s] %(message)s",
        datefmt="%d-%m %H:%M:%S",
    )
    args = parse_args()
    if args.list_devices:
        print(json.dumps(discover_devices(), ensure_ascii=False))
        return 0
    args.output_dir = args.output_dir or args.run_dir / "ivritai_eval"
    validate_inputs(args)
    sentences = read_sentences(args.sentences)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    from training.args import load_args

    train_args = load_args(args.run_dir / "args.yaml")
    logger.info(
        "%d sentence(s) x %d voice(s), %d checkpoint(s), model_config=%s",
        len(sentences),
        len(args.voices),
        len(args.checkpoints),
        train_args.model_config,
    )

    args.asr = args.asr.strip()
    devices = resolve_devices(args.device, args.devices)
    with cancellation_signals():
        code = run_evaluation(args, sentences, devices)
    logger.info("wrote %s", args.output_dir / "results.json")
    return code


if __name__ == "__main__":
    raise SystemExit(main())
