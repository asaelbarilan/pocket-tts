from __future__ import annotations

import argparse
import json
import re
import unicodedata
from pathlib import Path

# Score generated speech by transcribing it and comparing to the text we asked for.
#
# This exists because validation loss lied. The EOS term improved while generated speech
# got shorter and unintelligible, so loss ranked the worst model first. Word Error Rate is
# what the Pocket TTS paper reports, and it cannot be gamed by a model that stops early:
# stopping early deletes words, which raises WER.


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


_PUNCT = re.compile(r"[^\w\s֐-׿]", flags=re.UNICODE)
_SPACE = re.compile(r"\s+")


def normalize(text: str) -> str:
    """Compare words, not typography: drop punctuation, niqqud and spacing differences."""
    text = unicodedata.normalize("NFC", text)
    # Hebrew diacritics are not pronounced differently by the ASR and vary in the source.
    text = "".join(c for c in text if not ("֑" <= c <= "ׇ"))
    text = _PUNCT.sub(" ", text)
    return _SPACE.sub(" ", text).strip()


def main() -> None:
    args = parse_args()
    from faster_whisper import WhisperModel
    import jiwer

    sample_dirs = sorted(
        p.parent for p in args.runs_dir.glob("*/samples/step*/samples.json")
    )
    if args.only:
        # Compare with forward slashes so the filter works the same on Windows.
        needle = args.only.replace("\\", "/")
        sample_dirs = [d for d in sample_dirs if needle in d.as_posix()]
    if not sample_dirs:
        raise SystemExit(f"no sample directories under {args.runs_dir}")

    print(f"loading {args.model} on {args.device} ...", flush=True)
    model = WhisperModel(args.model, device=args.device, compute_type=args.compute_type)

    results = []
    for directory in sample_dirs:
        items = json.loads((directory / "samples.json").read_text(encoding="utf-8"))
        refs, hyps, per_clip = [], [], []
        for item in items:
            wav = directory / item["file"]
            if not wav.exists():
                continue
            segments, _ = model.transcribe(
                str(wav), language="he", beam_size=args.beam_size
            )
            hypothesis = normalize("".join(s.text for s in segments))
            reference = normalize(item["text"])
            if not reference:
                continue
            refs.append(reference)
            hyps.append(hypothesis)
            per_clip.append(
                {
                    "file": item["file"],
                    "reference": reference,
                    "hypothesis": hypothesis,
                    "wer": jiwer.wer(reference, hypothesis),
                }
            )
        if not refs:
            continue
        # An empty hypothesis gives WER 1.0 (every word deleted), which is the correct
        # penalty for a model that produced nothing intelligible.
        wer = jiwer.wer(refs, hyps)
        cer = jiwer.cer(refs, hyps)
        run = directory.parent.parent.name
        step = directory.name
        results.append(
            {
                "run": run,
                "step": step,
                "wer": wer,
                "cer": cer,
                "clips": len(refs),
                "empty_outputs": sum(1 for h in hyps if not h),
                "per_clip": per_clip,
            }
        )
        print(f"{run}/{step}: WER {wer:.3f}  CER {cer:.3f}  ({len(refs)} clips)", flush=True)

    args.output.write_text(
        json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"\nwrote {args.output}")
    print(f"\n{'run':<18}{'step':<12}{'WER':>8}{'CER':>8}{'empty':>7}")
    for r in sorted(results, key=lambda r: r["wer"]):
        print(
            f"{r['run']:<18}{r['step']:<12}{r['wer']:>8.3f}{r['cer']:>8.3f}"
            f"{r['empty_outputs']:>7}"
        )


if __name__ == "__main__":
    main()
