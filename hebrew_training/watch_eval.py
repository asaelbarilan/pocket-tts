"""Score every checkpoint of a training run on a fixed set of Hebrew sentences.

Kyutai's trainer computes no WER. It writes a validation loss and a few sample wavs, and on
this project validation loss ranked checkpoints differently from what the audio actually
sounded like three separate times -- so the loss curve cannot be used to choose a checkpoint.

This closes that gap. It watches a run directory, and for every checkpoint at or past
--min-step it generates the same fixed Hebrew sentences, transcribes them with
`ivrit-ai/whisper-large-v3-turbo-ct2`, and appends WER/CER to `hebrew_eval.jsonl` in the run
directory. `build_wer_dashboard.py` renders that file as a chart with the audio beside it.

    # follow a live run, scoring each checkpoint as it lands
    python -m hebrew_training.watch_eval --run-dir runs/finetune_hebrew \
        --voice prompts/hebrew_voice.wav --watch

    # or score whatever is already on disk and exit
    python -m hebrew_training.watch_eval --run-dir runs/finetune_hebrew \
        --voice prompts/hebrew_voice.wav

Resumable: a checkpoint already present in hebrew_eval.jsonl is skipped, so this can be
killed and restarted, and a --watch run can be started midway through training.

The scores are only as good as the sentence count. Five sentences is a progress signal, not
a measurement -- our earlier 12-clip scores swung 0.1-0.3 between adjacent checkpoints on
sampling noise alone. Use score_wer.py against a few hundred clips to choose the final model.
"""

from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path

DEFAULT_SENTENCES = [
    "אדוני היושב ראש, אני מבקש להעלות את הנושא לסדר היום.",
    "חברי הכנסת הנכבדים, מדובר בהחלטה משמעותית עבור אזרחי המדינה.",
    "הוועדה תתכנס ביום שלישי הקרוב בשעה עשר בבוקר.",
    "אני מודה לכם על ההקשבה ומסיים כאן את דבריי.",
    "שלוש עשרה חברות הצביעו בעד ההצעה, וארבע התנגדו.",
]

_STEP = re.compile(r"checkpoint_(\d+)\.pt$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--voice", type=Path, required=True,
                        help="Hebrew voice prompt wav. The model clones this; a noisy prompt "
                             "makes every generation sound noisy.")
    parser.add_argument("--sentences", type=Path,
                        help="One sentence per line. Defaults to five Knesset-style lines.")
    parser.add_argument("--min-step", type=int, default=4000,
                        help="Skip earlier checkpoints; before this the model has nothing to say.")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--asr", default="ivrit-ai/whisper-large-v3-turbo-ct2")
    parser.add_argument("--compute-type", default="int8_float16")
    parser.add_argument("--beam-size", type=int, default=5)
    parser.add_argument("--temp", type=float, default=0.3)
    parser.add_argument("--cfg", type=float, default=2.0,
                        help="Kyutai's reference eval value. Use 1.0 for a distilled student, "
                             "which has guidance baked in.")
    parser.add_argument("--n-steps", type=int, default=1)
    parser.add_argument("--eos-threshold", type=float, default=-1.0)
    parser.add_argument("--use-ema", action="store_true",
                        help="Score the EMA shadow instead of the raw weights.")
    parser.add_argument("--max-seconds", type=float, default=30.0)
    parser.add_argument("--watch", action="store_true",
                        help="Keep polling for new checkpoints instead of exiting.")
    parser.add_argument("--poll-seconds", type=float, default=120.0)
    return parser.parse_args()


def checkpoint_step(path: Path) -> int | None:
    match = _STEP.search(path.name)
    return int(match.group(1)) if match else None


def scored_steps(results: Path) -> set[int]:
    """Steps already in the results file, so a restart does not redo them."""
    if not results.exists():
        return set()
    done = set()
    with results.open(encoding="utf-8") as handle:
        for line in handle:
            try:
                done.add(int(json.loads(line)["step"]))
            except (json.JSONDecodeError, KeyError, ValueError):
                continue
    return done


def pending(run_dir: Path, min_step: int, done: set[int]) -> list[tuple[int, Path]]:
    found = []
    for path in run_dir.glob("checkpoint_*.pt"):
        step = checkpoint_step(path)
        if step is not None and step >= min_step and step not in done:
            found.append((step, path))
    return sorted(found)


def score_checkpoint(path: Path, step: int, sentences: list[str], args, asr) -> dict:
    """Generate every sentence from one checkpoint and score the transcriptions."""
    import jiwer
    import soundfile
    import torch

    from hebrew_training.evaluation import normalize_hebrew_for_asr
    from training.eval.librispeech import latents_to_wav, load_mono, load_run

    model, mimi, _ = load_run(args.run_dir, args.device, use_ema=args.use_ema, checkpoint=path)
    out_dir = args.run_dir / "hebrew_eval" / f"step{step:08d}"
    out_dir.mkdir(parents=True, exist_ok=True)

    tokenize = model.flow_lm.conditioner.tokenizer.sp.encode
    tokens = [torch.tensor(tokenize(s), dtype=torch.long) for s in sentences]
    with torch.no_grad():
        voice = load_mono(str(args.voice), mimi.sample_rate)
        voice_latents = mimi.encode_to_latent(voice[None, None].to(args.device))[0]
        outs = model.generate(
            tokens,
            [voice_latents] * len(tokens),
            max_frames=int(args.max_seconds * mimi.frame_rate),
            temp=args.temp,
            n_steps=args.n_steps,
            cfg_coef=args.cfg,
            eos_threshold=args.eos_threshold,
        )

    references, hypotheses, clips = [], [], []
    for index, (sentence, latents) in enumerate(zip(sentences, outs, strict=True)):
        wav = latents_to_wav(mimi, latents, args.device)
        if wav is None:
            # An empty generation is a real failure mode, not a missing data point: score it
            # as a total miss rather than dropping it and flattering the checkpoint.
            references.append(normalize_hebrew_for_asr(sentence))
            hypotheses.append("")
            clips.append({"index": index, "file": None, "reference": sentence,
                          "hypothesis": "", "wer": 1.0, "seconds": 0.0, "empty": True})
            continue
        audio = wav.float().cpu().numpy()
        name = f"{index}.wav"
        soundfile.write(str(out_dir / name), audio, mimi.sample_rate)
        segments, _ = asr.transcribe(str(out_dir / name), language="he", beam_size=args.beam_size)
        hypothesis = normalize_hebrew_for_asr("".join(s.text for s in segments))
        reference = normalize_hebrew_for_asr(sentence)
        references.append(reference)
        hypotheses.append(hypothesis)
        clips.append({
            "index": index, "file": f"hebrew_eval/step{step:08d}/{name}",
            "reference": sentence, "hypothesis": hypothesis,
            "wer": jiwer.wer(reference, hypothesis) if reference else None,
            "seconds": len(audio) / mimi.sample_rate, "empty": False,
        })

    del model
    if args.device.startswith("cuda"):
        torch.cuda.empty_cache()
    return {
        "step": step,
        "checkpoint": path.name,
        "ema": bool(args.use_ema),
        "wer": jiwer.wer(references, hypotheses),
        "cer": jiwer.cer(references, hypotheses),
        "empty_outputs": sum(1 for c in clips if c["empty"]),
        "mean_seconds": sum(c["seconds"] for c in clips) / len(clips),
        "clips": clips,
    }


def main() -> None:
    args = parse_args()
    sentences = (
        [line.strip() for line in args.sentences.read_text(encoding="utf-8").splitlines()
         if line.strip()]
        if args.sentences else list(DEFAULT_SENTENCES)
    )
    if not args.voice.exists():
        raise SystemExit(f"voice prompt not found: {args.voice}")
    results = args.run_dir / "hebrew_eval.jsonl"

    from faster_whisper import WhisperModel

    print(f"loading {args.asr} ...", flush=True)
    asr = WhisperModel(args.asr, device=args.device, compute_type=args.compute_type)
    print(f"scoring {len(sentences)} sentences per checkpoint, from step {args.min_step}",
          flush=True)

    while True:
        todo = pending(args.run_dir, args.min_step, scored_steps(results))
        for step, path in todo:
            print(f"step {step}: scoring {path.name} ...", flush=True)
            try:
                record = score_checkpoint(path, step, sentences, args, asr)
            except Exception as exc:  # noqa: BLE001 -- a half-written checkpoint must not stop the watch
                print(f"  failed: {exc}", flush=True)
                continue
            with results.open("a", encoding="utf-8", newline="\n") as handle:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            print(f"  WER {record['wer']:.3f}  CER {record['cer']:.3f}  "
                  f"empty {record['empty_outputs']}/{len(sentences)}  "
                  f"mean {record['mean_seconds']:.1f}s", flush=True)
        if not args.watch:
            break
        time.sleep(args.poll_seconds)

    print(f"\n{results} holds every scored checkpoint")
    print("render it with: python -m hebrew_training.build_wer_dashboard "
          f"--run-dir {args.run_dir}")


if __name__ == "__main__":
    main()
