from __future__ import annotations

import argparse
from pathlib import Path

import scipy.io.wavfile

from pocket_tts import TTSModel


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate one Hebrew sample from an export.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--voice", required=True, help="A consented local voice WAV or voice state.")
    parser.add_argument("--output", type=Path, default=Path("hebrew_test.wav"))
    parser.add_argument("--text", default="שלום, זהו מבחן של מודל דיבור בעברית.")
    args = parser.parse_args()

    model = TTSModel.load_model(config=args.config)
    voice_state = model.get_state_for_audio_prompt(args.voice)
    audio = model.generate_audio(voice_state, args.text)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    scipy.io.wavfile.write(args.output, model.sample_rate, audio.numpy())
    print(args.output.resolve())


if __name__ == "__main__":
    main()
