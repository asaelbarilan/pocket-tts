from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
import time
import unicodedata
from pathlib import Path

# Generate the fixed evaluation set from a checkpoint and score it by WER, in one pass.
#
# Replaces the old flow of "generate 12 clips, score separately". 12 clips could not
# separate checkpoints: adjacent steps differed by 0.1-0.3 WER purely from sampling. The
# fixed set from build_eval_set.py is the same sentences and the same voice prompts for
# every model, which is what makes two numbers comparable at all.


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate and WER-score checkpoints.")
    parser.add_argument("--eval-set", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--checkpoints", type=Path, nargs="+", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-tokens", type=int, default=250)
    parser.add_argument("--asr-model", default="ivrit-ai/whisper-large-v3-turbo-ct2")
    parser.add_argument("--asr-compute-type", default="int8_float16")
    parser.add_argument("--beam-size", type=int, default=5)
    parser.add_argument("--limit-clips", type=int, help="Debug: cap clips per checkpoint.")
    parser.add_argument("--keep-wav", action="store_true",
                        help="Keep wavs. By default only mp3 is kept, to save disk.")
    return parser.parse_args()


_PUNCT = re.compile(r"[^\w\s֐-׿]", flags=re.UNICODE)
_WS = re.compile(r"\s+")


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFC", text)
    text = "".join(c for c in text if not ("֑" <= c <= "ׇ"))
    return _WS.sub(" ", _PUNCT.sub(" ", text)).strip()


def main() -> None:
    args = parse_args()
    import scipy.io.wavfile
    import torch
    import jiwer
    from faster_whisper import WhisperModel
    from pocket_tts import TTSModel

    spec = json.loads(args.eval_set.read_text(encoding="utf-8"))
    sentences, speakers = spec["sentences"], spec["speakers"]
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"loading ASR {args.asr_model} ...", flush=True)
    asr = WhisperModel(args.asr_model, device=args.device, compute_type=args.asr_compute_type)

    results_path = args.output_dir / "eval_results.json"
    results = json.loads(results_path.read_text(encoding="utf-8")) if results_path.exists() else []
    done = {(r["run"], r["step"]) for r in results}

    for checkpoint in args.checkpoints:
        run = checkpoint.parent.name
        step = int(checkpoint.name.replace("checkpoint-", ""))
        if (run, step) in done:
            print(f"{run}/{step}: already scored, skipping")
            continue

        started = time.time()
        export_dir = args.output_dir / f"_export_{run}_{step}"
        # sys.executable, not "python": the venv interpreter is the one with pocket_tts.
        export = subprocess.run(
            [sys.executable, "-m", "hebrew_training.export", "--checkpoint", str(checkpoint),
             "--tokenizer", str(args.tokenizer), "--output-dir", str(export_dir)],
            capture_output=True, text=True,
        )
        if export.returncode != 0:
            raise SystemExit(f"export failed for {checkpoint}:\n{export.stderr[-800:]}")
        model = TTSModel.load_model(config=export_dir / "hebrew.yaml")
        model.to(args.device).eval()

        clip_dir = args.output_dir / f"{run}_step{step:07d}"
        clip_dir.mkdir(parents=True, exist_ok=True)
        refs, hyps, per_clip = [], [], []
        made = 0
        for si, speaker in enumerate(speakers, start=1):
            # Encode each voice once, not once per sentence.
            voice = model.get_state_for_audio_prompt(speaker["prompt_audio_path"])
            for ti, sentence in enumerate(sentences, start=1):
                if args.limit_clips and made >= args.limit_clips:
                    break
                audio = model.generate_audio(voice, sentence, max_tokens=args.max_tokens)
                name = f"spk{si:02d}_s{ti:02d}.wav"
                scipy.io.wavfile.write(
                    clip_dir / name, model.sample_rate, audio.detach().cpu().numpy()
                )
                segments, _ = asr.transcribe(
                    str(clip_dir / name), language="he", beam_size=args.beam_size
                )
                hypothesis = normalize("".join(s.text for s in segments))
                reference = normalize(sentence)
                refs.append(reference)
                hyps.append(hypothesis)
                per_clip.append({"file": name, "speaker": speaker["speaker_id"][:8],
                                 "reference": reference, "hypothesis": hypothesis})
                made += 1
            if args.limit_clips and made >= args.limit_clips:
                break

        wer = jiwer.wer(refs, hyps)
        cer = jiwer.cer(refs, hyps)
        empty = sum(1 for h in hyps if not h)
        results.append({
            "run": run, "step": step, "wer": wer, "cer": cer,
            "clips": len(refs), "empty_outputs": empty,
            "seconds": round(time.time() - started, 1), "per_clip": per_clip,
        })
        results_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")

        for wav in clip_dir.glob("*.wav"):
            subprocess.run(
                ["ffmpeg", "-nostdin", "-loglevel", "error", "-y", "-i", str(wav),
                 "-codec:a", "libmp3lame", "-b:a", "96k", "-ac", "1",
                 str(wav.with_suffix(".mp3"))], check=False,
            )
            if not args.keep_wav:
                wav.unlink()
        shutil.rmtree(export_dir, ignore_errors=True)
        del model
        torch.cuda.empty_cache()
        print(f"{run}/{step}: WER {wer:.4f}  CER {cer:.4f}  empty {empty}/{len(refs)}"
              f"  ({time.time()-started:.0f}s)", flush=True)

    print(f"\n{'run':<18}{'step':>8}{'WER':>9}{'CER':>9}{'empty':>7}")
    for r in sorted(results, key=lambda r: r["wer"]):
        print(f"{r['run']:<18}{r['step']:>8}{r['wer']:>9.4f}{r['cer']:>9.4f}{r['empty_outputs']:>7}")
    print(f"\nwrote {results_path}")


if __name__ == "__main__":
    main()
