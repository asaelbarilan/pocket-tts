from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

from hebrew_training.evaluation import normalize_hebrew_for_asr, score_transcripts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate and score fixed checkpoint evaluations.")
    parser.add_argument("--eval-set", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--checkpoints", type=Path, nargs="+", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-tokens", type=int, default=250)
    parser.add_argument("--asr-model", default="ivrit-ai/whisper-large-v3-turbo-ct2")
    parser.add_argument("--asr-compute-type", default="int8_float16")
    parser.add_argument("--beam-size", type=int, default=5)
    parser.add_argument("--bootstrap-samples", type=int, default=1_000)
    parser.add_argument("--limit-clips", type=int)
    parser.add_argument("--keep-wav", action="store_true")
    return parser.parse_args()


def evaluation_groups(spec: dict) -> list[dict]:
    if "groups" in spec:
        return spec["groups"]
    return [
        {
            "name": "legacy_unseen_speaker_unseen_text",
            "speaker_status": "unseen",
            "text_status": "unseen",
            "sentences": spec["sentences"],
            "speakers": spec["speakers"],
            "clips": len(spec["sentences"]) * len(spec["speakers"]),
        }
    ]


def seed_generation(torch, seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main() -> None:
    args = parse_args()
    import scipy.io.wavfile
    import torch
    from faster_whisper import WhisperModel
    from pocket_tts import TTSModel

    spec = json.loads(args.eval_set.read_text(encoding="utf-8"))
    groups = evaluation_groups(spec)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    print(f"loading ASR {args.asr_model} ...", flush=True)
    asr = WhisperModel(args.asr_model, device=args.device, compute_type=args.asr_compute_type)

    results_path = args.output_dir / "eval_results.json"
    results = json.loads(results_path.read_text(encoding="utf-8")) if results_path.exists() else []
    done = {(result["run"], result["step"]) for result in results}

    for checkpoint in args.checkpoints:
        run = checkpoint.parent.name
        step = int(checkpoint.name.replace("checkpoint-", ""))
        if (run, step) in done:
            print(f"{run}/{step}: already scored, skipping")
            continue
        started = time.time()
        export_dir = args.output_dir / f"_export_{run}_{step}"
        export = subprocess.run(
            [
                sys.executable,
                "-m",
                "hebrew_training.export",
                "--checkpoint",
                str(checkpoint),
                "--tokenizer",
                str(args.tokenizer),
                "--output-dir",
                str(export_dir),
            ],
            capture_output=True,
            text=True,
        )
        if export.returncode != 0:
            raise SystemExit(f"export failed for {checkpoint}:\n{export.stderr[-800:]}")
        model = TTSModel.load_model(config=export_dir / "hebrew.yaml")
        model.to(args.device).eval()

        clip_dir = args.output_dir / f"{run}_step{step:07d}"
        clip_dir.mkdir(parents=True, exist_ok=True)
        all_references, all_hypotheses, per_clip = [], [], []
        grouped_transcripts: dict[str, tuple[list[str], list[str]]] = {}
        made = 0
        voice_cache = {}
        stop = False
        for group_index, group in enumerate(groups):
            group_references, group_hypotheses = [], []
            for speaker_index, speaker in enumerate(group["speakers"]):
                prompt = speaker["prompt_audio_path"]
                if prompt not in voice_cache:
                    voice_cache[prompt] = model.get_state_for_audio_prompt(prompt)
                voice = voice_cache[prompt]
                for text_index, sentence in enumerate(group["sentences"]):
                    if args.limit_clips and made >= args.limit_clips:
                        stop = True
                        break
                    generation_seed = (
                        int(spec.get("seed", 1234))
                        + group_index * 1_000_000
                        + speaker_index * 10_000
                        + text_index
                    )
                    seed_generation(torch, generation_seed)
                    audio = model.generate_audio(voice, sentence, max_tokens=args.max_tokens)
                    name = (
                        f"g{group_index + 1}_spk{speaker_index + 1:02d}_"
                        f"s{text_index + 1:02d}.wav"
                    )
                    wav_path = clip_dir / name
                    scipy.io.wavfile.write(
                        wav_path, model.sample_rate, audio.detach().cpu().numpy()
                    )
                    segments, _ = asr.transcribe(
                        str(wav_path), language="he", beam_size=args.beam_size
                    )
                    reference = normalize_hebrew_for_asr(sentence)
                    hypothesis = normalize_hebrew_for_asr(
                        "".join(segment.text for segment in segments)
                    )
                    group_references.append(reference)
                    group_hypotheses.append(hypothesis)
                    all_references.append(reference)
                    all_hypotheses.append(hypothesis)
                    per_clip.append(
                        {
                            "file": name,
                            "group": group["name"],
                            "speaker_id": speaker["speaker_id"],
                            "generation_seed": generation_seed,
                            "reference": reference,
                            "hypothesis": hypothesis,
                        }
                    )
                    made += 1
                if stop:
                    break
            if group_references:
                grouped_transcripts[group["name"]] = (group_references, group_hypotheses)
            if stop:
                break

        score = score_transcripts(
            all_references,
            all_hypotheses,
            bootstrap_samples=args.bootstrap_samples,
            seed=int(spec.get("seed", 1234)),
        )
        group_scores = {
            name: score_transcripts(
                references,
                hypotheses,
                bootstrap_samples=args.bootstrap_samples,
                seed=int(spec.get("seed", 1234)) + index + 10,
            )
            for index, (name, (references, hypotheses)) in enumerate(grouped_transcripts.items())
        }
        result = {
            "run": run,
            "step": step,
            **score,
            "groups": group_scores,
            "seconds": round(time.time() - started, 1),
            "per_clip": per_clip,
        }
        results.append(result)
        results_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")

        for wav in clip_dir.glob("*.wav"):
            subprocess.run(
                [
                    "ffmpeg",
                    "-nostdin",
                    "-loglevel",
                    "error",
                    "-y",
                    "-i",
                    str(wav),
                    "-codec:a",
                    "libmp3lame",
                    "-b:a",
                    "96k",
                    "-ac",
                    "1",
                    str(wav.with_suffix(".mp3")),
                ],
                check=False,
            )
            if not args.keep_wav:
                wav.unlink()
        shutil.rmtree(export_dir, ignore_errors=True)
        del model
        torch.cuda.empty_cache()
        print(
            f"{run}/{step}: WER {score['wer']:.4f} CER {score['cer']:.4f} "
            f"empty {score['empty_outputs']}/{score['clips']} ({time.time() - started:.0f}s)",
            flush=True,
        )
        for name, group_score in group_scores.items():
            print(
                f"  {name}: WER {group_score['wer']:.4f} CER {group_score['cer']:.4f} "
                f"n={group_score['clips']}"
            )

    print(f"\n{'run':<28}{'step':>8}{'WER':>9}{'CER':>9}{'clips':>8}")
    for result in sorted(results, key=lambda item: item["wer"]):
        print(
            f"{result['run']:<28}{result['step']:>8}{result['wer']:>9.4f}"
            f"{result['cer']:>9.4f}{result['clips']:>8}"
        )
    print(f"\nwrote {results_path}")


if __name__ == "__main__":
    main()
