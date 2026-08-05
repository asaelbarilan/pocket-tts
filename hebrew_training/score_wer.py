from __future__ import annotations

import argparse
import json
from pathlib import Path

from hebrew_training.evaluation import normalize_hebrew_for_asr


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Transcribe generated samples and score WER/CER against the prompt text."
    )
    parser.add_argument("--runs-dir", type=Path, default=Path("runs"))
    parser.add_argument("--model", default="ivrit-ai/whisper-large-v3-turbo-ct2")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--compute-type", default="int8_float16")
    parser.add_argument("--beam-size", type=int, default=5)
    parser.add_argument("--output", type=Path, default=Path("runs/wer_scores.json"))
    parser.add_argument(
        "--only", help="Substring filter on the sample directory path, for a quick subset."
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    import jiwer
    from faster_whisper import WhisperModel

    sample_dirs = sorted(path.parent for path in args.runs_dir.glob("*/samples/step*/samples.json"))
    if args.only:
        needle = args.only.replace("\\", "/")
        sample_dirs = [directory for directory in sample_dirs if needle in directory.as_posix()]
    if not sample_dirs:
        raise SystemExit(f"no sample directories under {args.runs_dir}")

    print(f"loading {args.model} on {args.device} ...", flush=True)
    model = WhisperModel(args.model, device=args.device, compute_type=args.compute_type)
    results = []
    for directory in sample_dirs:
        items = json.loads((directory / "samples.json").read_text(encoding="utf-8"))
        references, hypotheses, per_clip = [], [], []
        for item in items:
            wav = directory / item["file"]
            if not wav.exists():
                continue
            segments, _ = model.transcribe(str(wav), language="he", beam_size=args.beam_size)
            hypothesis = normalize_hebrew_for_asr("".join(segment.text for segment in segments))
            reference = normalize_hebrew_for_asr(item["text"])
            if not reference:
                continue
            references.append(reference)
            hypotheses.append(hypothesis)
            per_clip.append(
                {
                    "file": item["file"],
                    "reference": reference,
                    "hypothesis": hypothesis,
                    "wer": jiwer.wer(reference, hypothesis),
                }
            )
        if not references:
            continue
        wer = jiwer.wer(references, hypotheses)
        cer = jiwer.cer(references, hypotheses)
        run = directory.parent.parent.name
        step = directory.name
        results.append(
            {
                "run": run,
                "step": step,
                "wer": wer,
                "cer": cer,
                "clips": len(references),
                "empty_outputs": sum(not hypothesis for hypothesis in hypotheses),
                "per_clip": per_clip,
            }
        )
        print(f"{run}/{step}: WER {wer:.3f} CER {cer:.3f} ({len(references)} clips)", flush=True)

    args.output.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nwrote {args.output}")
    print(f"\n{'run':<18}{'step':<12}{'WER':>8}{'CER':>8}{'empty':>7}")
    for result in sorted(results, key=lambda item: item["wer"]):
        print(
            f"{result['run']:<18}{result['step']:<12}{result['wer']:>8.3f}"
            f"{result['cer']:>8.3f}{result['empty_outputs']:>7}"
        )


if __name__ == "__main__":
    main()
