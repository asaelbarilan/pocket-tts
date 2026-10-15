"""Run isolated Silero VAD endpoint analysis for the dataset viewer.

This file is intentionally runnable with a separate Python environment. The viewer invokes it
as a subprocess so its VAD/Torch dependencies do not become dependencies of pocket-tts.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("audio", type=Path)
    parser.add_argument("--method", choices=("vad", "envelope"), default="vad")
    parser.add_argument("--word-start", type=float, help="Last-word start in audio seconds")
    parser.add_argument("--forced-end", type=float, required=True, help="Endpoint in audio seconds")
    parser.add_argument("--sample-end", type=float, required=True, help="Original sample end")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--min-speech-ms", type=int, default=20)
    parser.add_argument("--min-silence-ms", type=int, default=40)
    parser.add_argument("--max-extension-ms", type=int, default=120)
    parser.add_argument("--safety-ms", type=int, default=5)
    parser.add_argument("--quiet-fraction", type=float, default=0.10)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.method == "envelope":
        import numpy as np
        import soundfile as sf

        audio, sample_rate = sf.read(str(args.audio), dtype="float32", always_2d=True)
        audio = np.abs(audio.mean(axis=1))
        hop = max(1, round(0.005 * sample_rate))  # 5 ms, matching tmp_correct_mms.py
        frame_count = len(audio) // hop
        envelope = audio[: frame_count * hop].reshape(frame_count, hop).max(axis=1)
        start = max(0, int((args.word_start or args.forced_end) * sample_rate / hop))
        end = min(len(envelope), max(start + 1, int(args.forced_end * sample_rate / hop)))
        peak = float(envelope[start:end].max())
        threshold = args.quiet_fraction * peak
        limit = min(
            len(envelope), int((args.forced_end + args.max_extension_ms / 1000) * sample_rate / hop)
        )
        cursor = end
        while cursor < limit and envelope[cursor] > threshold:
            cursor += 1
        candidate = cursor * hop / sample_rate
        corrected_end = min(
            candidate + args.safety_ms / 1000, args.forced_end + args.max_extension_ms / 1000
        )
        extension = max(0.0, corrected_end - args.forced_end)
        applied = 0.020 <= extension <= args.max_extension_ms / 1000
        if not applied:
            corrected_end = args.forced_end
            extension = 0.0
        print(
            json.dumps(
                {
                    "method": "envelope",
                    "forced_end": args.forced_end,
                    "candidate_end": candidate,
                    "corrected_end": corrected_end,
                    "extension": extension,
                    "applied": applied,
                    "peak": peak,
                    "threshold": threshold,
                    "quiet_fraction": args.quiet_fraction,
                }
            )
        )
        return

    from silero_vad import get_speech_timestamps, load_silero_vad, read_audio

    model = load_silero_vad()
    audio = read_audio(str(args.audio), sampling_rate=16_000)
    timestamps = get_speech_timestamps(
        audio,
        model,
        threshold=args.threshold,
        sampling_rate=16_000,
        min_speech_duration_ms=args.min_speech_ms,
        min_silence_duration_ms=args.min_silence_ms,
        speech_pad_ms=0,
        return_seconds=False,
    )

    forced_end = max(0.0, args.forced_end)
    sample_end = max(forced_end, args.sample_end)
    max_end = min(len(audio) / 16_000, sample_end + args.max_extension_ms / 1000)
    candidate = forced_end
    for span in timestamps:
        span_start = span["start"] / 16_000
        span_end = span["end"] / 16_000
        # Require the VAD span to touch the MMS endpoint. This avoids extending from a
        # separate noise burst after a real silence.
        if span_start <= forced_end + 0.030 and span_end > forced_end:
            candidate = max(candidate, min(span_end, max_end))

    corrected_end = min(max_end, candidate + args.safety_ms / 1000)
    extension = max(0.0, corrected_end - forced_end)
    applied = 0.020 <= extension <= args.max_extension_ms / 1000
    if not applied:
        corrected_end = forced_end
        extension = 0.0

    print(
        json.dumps(
            {
                "forced_end": forced_end,
                "vad_end": candidate,
                "corrected_end": corrected_end,
                "extension": extension,
                "applied": applied,
                "speech_spans": [
                    {"start": item["start"] / 16_000, "end": item["end"] / 16_000}
                    for item in timestamps
                ],
            }
        )
    )


if __name__ == "__main__":
    main()
