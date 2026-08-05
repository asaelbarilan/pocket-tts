from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from hebrew_training.data import read_jsonl

# Fixed sentences, so samples from different checkpoints are directly comparable.
# Held-out text: none of these come from the training corpus.
DEFAULT_TEXTS = [
    "שלום, זהו מבחן של מודל דיבור בעברית.",
    "היום מזג האוויר נאה, ואפשר לצאת לטיול בגן.",
    "המחקר החדש מראה תוצאות מפתיעות בתחום הבריאות.",
    "הוא סיפר לי שהוא נוסע לירושלים בשבוע הבא.",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate fixed Hebrew sentences in held-out validation voices, so a "
            "checkpoint can be listened to rather than only scored."
        )
    )
    parser.add_argument(
        "--export-dir",
        type=Path,
        help="Exported checkpoint directory. Omit and pass --base-model for the step-0 baseline.",
    )
    parser.add_argument(
        "--base-model",
        action="store_true",
        help=(
            "Generate from the untrained Kyutai model with the Hebrew tokenizer installed. "
            "This is exactly where training started, so it is the honest 'before' baseline. "
            "Requires --tokenizer."
        ),
    )
    parser.add_argument("--tokenizer", type=Path, help="Hebrew tokenizer for --base-model.")
    parser.add_argument("--base-language", default="english")
    parser.add_argument("--validation-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--speakers", type=int, default=3)
    parser.add_argument("--texts", type=Path, help="Optional file, one sentence per line.")
    parser.add_argument(
        "--max-tokens", type=int, default=250, help="Frames at 12.5 Hz; 250 is about 20 s."
    )
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def pick_voices(manifest: Path, speaker_count: int) -> list[dict]:
    """One prompt clip per speaker, longest first so the voice prompt is well conditioned."""
    rows = list(read_jsonl(manifest))
    by_speaker: dict[str, dict] = {}
    for row in rows:
        current = by_speaker.get(row["speaker_id"])
        if current is None or row["duration"] > current["duration"]:
            by_speaker[row["speaker_id"]] = row
    ordered = sorted(by_speaker.values(), key=lambda r: -r["duration"])
    return ordered[:speaker_count]


def main() -> None:
    args = parse_args()
    import scipy.io.wavfile

    from pocket_tts import TTSModel

    if args.base_model:
        if not args.tokenizer:
            raise SystemExit("--base-model requires --tokenizer")
    elif not args.export_dir:
        raise SystemExit("pass --export-dir, or --base-model with --tokenizer")

    texts = DEFAULT_TEXTS
    if args.texts:
        texts = [t.strip() for t in args.texts.read_text(encoding="utf-8").splitlines() if t.strip()]

    voices = pick_voices(args.validation_manifest, args.speakers)
    if not voices:
        raise RuntimeError(f"No rows in {args.validation_manifest}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.base_model:
        from hebrew_training.model_utils import install_tokenizer

        model = TTSModel.load_model(language=args.base_language)
        copied = install_tokenizer(model, args.tokenizer)
        print(f"base model, Hebrew tokenizer installed, {copied} token pieces reused")
    else:
        model = TTSModel.load_model(config=args.export_dir / "hebrew.yaml")
    model.to(args.device).eval()

    index = []
    for speaker_number, voice in enumerate(voices, start=1):
        prompt_path = Path(voice["audio_path"])
        # Copy the reference clip in too, so the voice can be compared against the output.
        reference_name = f"speaker{speaker_number}_reference.wav"
        shutil.copyfile(prompt_path, args.output_dir / reference_name)

        voice_state = model.get_state_for_audio_prompt(str(prompt_path))
        for text_number, text in enumerate(texts, start=1):
            # generate_audio defaults to max_tokens=50, which is only ~4 s at the 12.5 Hz
            # frame rate and would clip these sentences mid-word.
            audio = model.generate_audio(voice_state, text, max_tokens=args.max_tokens)
            name = f"speaker{speaker_number}_text{text_number}.wav"
            # .cpu() is required on GPU: numpy() cannot read a cuda tensor.
            scipy.io.wavfile.write(
                args.output_dir / name, model.sample_rate, audio.detach().cpu().numpy()
            )
            index.append(
                {
                    "file": name,
                    "text": text,
                    "speaker_id": voice["speaker_id"],
                    "speaker_number": speaker_number,
                    "reference": reference_name,
                    "reference_duration": voice["duration"],
                }
            )
            print(f"wrote {name}")

    (args.output_dir / "samples.json").write_text(
        json.dumps(index, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"\n{len(index)} samples in {args.output_dir}")


if __name__ == "__main__":
    main()
