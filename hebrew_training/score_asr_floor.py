from __future__ import annotations

import argparse
import json
from pathlib import Path

from hebrew_training.evaluation import normalize_hebrew_for_asr, score_transcripts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Measure Hebrew ASR error on genuine held-out audio.")
    parser.add_argument("--eval-set", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--model", default="ivrit-ai/whisper-large-v3-turbo-ct2")
    parser.add_argument("--compute-type", default="int8_float16")
    parser.add_argument("--beam-size", type=int, default=5)
    parser.add_argument("--bootstrap-samples", type=int, default=1_000)
    parser.add_argument("--limit", type=int)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    from faster_whisper import WhisperModel

    spec = json.loads(args.eval_set.read_text(encoding="utf-8"))
    rows = spec.get("asr_floor", [])
    if args.limit:
        rows = rows[: args.limit]
    if not rows:
        raise SystemExit("evaluation set has no asr_floor rows")
    print(f"loading {args.model} on {args.device} ...", flush=True)
    model = WhisperModel(args.model, device=args.device, compute_type=args.compute_type)
    references, hypotheses, per_clip = [], [], []
    for index, row in enumerate(rows, start=1):
        audio_path = Path(row["audio_path"])
        if not audio_path.exists():
            raise FileNotFoundError(audio_path)
        segments, _ = model.transcribe(str(audio_path), language="he", beam_size=args.beam_size)
        reference = normalize_hebrew_for_asr(row["text"])
        hypothesis = normalize_hebrew_for_asr("".join(segment.text for segment in segments))
        references.append(reference)
        hypotheses.append(hypothesis)
        per_clip.append(
            {
                "speaker_id": row["speaker_id"],
                "audio_path": str(audio_path),
                "reference": reference,
                "hypothesis": hypothesis,
            }
        )
        if index % 10 == 0 or index == len(rows):
            print(f"transcribed {index}/{len(rows)}", flush=True)
    result = {
        "kind": "genuine_held_out_asr_floor",
        "model": args.model,
        **score_transcripts(
            references,
            hypotheses,
            bootstrap_samples=args.bootstrap_samples,
            seed=spec.get("seed", 1234),
        ),
        "per_clip": per_clip,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        f"ASR floor: WER {result['wer']:.4f} {result['wer_ci95']}  "
        f"CER {result['cer']:.4f} {result['cer_ci95']}"
    )
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
