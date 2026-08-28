"""Transcribe the samples a training run wrote and score WER/CER against their text.

Reads two layouts:

  * Kyutai's trainer (training/train_utils.py:write_samples) writes flat wavs,
    `runs/<run>/samples/step00010000_<i>.wav`, with no transcript beside them. The text is
    recoverable because `<i>` indexes `sample_sentences` in the `args.yaml` the run saves.
  * Our earlier trainer wrote `runs/<run>/samples/step<N>/samples.json` carrying the text
    per clip. Still read, so old runs stay scoreable.

If a Kyutai run's `sample_sentences` are still the English defaults, every sample is English
text pushed through a Hebrew tokenizer and the resulting WER is meaningless. This refuses to
score those rather than reporting a number nobody should trust.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from hebrew_training.evaluation import normalize_hebrew_for_asr

_HEBREW = re.compile(r"[֐-׿]")
_STEP_WAV = re.compile(r"^step(\d+)_(\d+)\.wav$")


def kyutai_sample_groups(runs_dir: Path) -> list[tuple[str, str, list[dict]]]:
    """(run, step, items) for each step of each run written by Kyutai's trainer."""
    import yaml

    groups: list[tuple[str, str, list[dict]]] = []
    for samples_dir in sorted(runs_dir.glob("*/samples")):
        run_dir = samples_dir.parent
        config = run_dir / "args.yaml"
        if not config.exists():
            continue
        sentences = (yaml.safe_load(config.read_text(encoding="utf-8")) or {}).get(
            "sample_sentences"
        )
        if not sentences:
            continue
        if not any(_HEBREW.search(sentence) for sentence in sentences):
            print(
                f"skipping {run_dir.name}: sample_sentences in args.yaml are not Hebrew, so "
                f"these samples cannot be scored. Set sample_sentences in the training config.",
                flush=True,
            )
            continue
        by_step: dict[str, list[dict]] = {}
        for wav in sorted(samples_dir.glob("step*_*.wav")):
            match = _STEP_WAV.match(wav.name)
            if not match:
                continue
            index = int(match.group(2))
            if index >= len(sentences):
                continue
            by_step.setdefault(f"step{match.group(1)}", []).append(
                {"file": wav.name, "text": sentences[index]}
            )
        for step, items in sorted(by_step.items()):
            groups.append((run_dir.name, step, items))
    return groups


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

    # Our own layout first, then Kyutai's; a runs/ directory can hold both.
    groups: list[tuple[str, str, Path, list[dict]]] = []
    for path in sorted(args.runs_dir.glob("*/samples/step*/samples.json")):
        directory = path.parent
        items = json.loads(path.read_text(encoding="utf-8"))
        groups.append((directory.parent.parent.name, directory.name, directory, items))
    for run, step, items in kyutai_sample_groups(args.runs_dir):
        groups.append((run, step, args.runs_dir / run / "samples", items))
    if args.only:
        needle = args.only.replace("\\", "/")
        groups = [g for g in groups if needle in (g[2] / g[1]).as_posix()]
    if not groups:
        raise SystemExit(
            f"no scoreable samples under {args.runs_dir}. Kyutai's trainer writes "
            f"runs/<run>/samples/step<N>_<i>.wav and needs args.yaml beside them."
        )

    print(f"loading {args.model} on {args.device} ...", flush=True)
    model = WhisperModel(args.model, device=args.device, compute_type=args.compute_type)
    results = []
    for run, step, directory, items in groups:
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
