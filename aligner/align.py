"""Run MMS CTC forced alignment from the dedicated external environment.

The project environment intentionally does not need transformers, uroman, or the
MMS-specific torch build.  This module passes a one-row manifest to ``ctc.py``
using the Python interpreter in ``/home/yoadsnapir/time-align/mms-ctc``.

Examples:
    python aligner/align.py audio.wav "text to align"
    python aligner/align.py audio.wav "text to align" --output timings.json
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import uuid
import wave
from pathlib import Path
from typing import Any

DEFAULT_ENV = Path("/home/yoadsnapir/time-align/mms-ctc/ctc-env")
MODEL = "MahmoudAshraf/mms-300m-1130-forced-aligner"
CTC_SCRIPT = Path(__file__).with_name("ctc.py")


def _audio_duration(audio_path: Path) -> float:
    try:
        with wave.open(str(audio_path), "rb") as audio:
            return audio.getnframes() / audio.getframerate()
    except (wave.Error, ZeroDivisionError) as exc:
        raise ValueError(f"Could not read WAV metadata from {audio_path}: {exc}") from exc


def align(
    audio_path: str | os.PathLike[str],
    text: str,
    *,
    device: str = "cuda",
    environment: str | os.PathLike[str] = DEFAULT_ENV,
) -> dict[str, Any]:
    """Return MMS word timings for one WAV file and transcript.

    The returned object has the shape ``{"words": [{"word", "start", "end"}, ...]}``.
    Words the model cannot spell are omitted by the underlying CTC aligner.
    """
    audio = Path(audio_path).expanduser().resolve()
    if not audio.is_file():
        raise FileNotFoundError(audio)
    if not text.strip():
        raise ValueError("text must not be empty")
    if not CTC_SCRIPT.is_file():
        raise FileNotFoundError(CTC_SCRIPT)

    python = Path(environment).expanduser() / "bin" / "python"
    if not python.is_file():
        raise FileNotFoundError(
            f"Dedicated CTC Python was not found at {python}. "
            "Create it with: uv venv --python 3.11 /home/yoadsnapir/time-align/mms-ctc/ctc-env"
        )

    row_id = uuid.uuid4().hex
    row = {
        "id": row_id,
        "audio": str(audio),
        "duration": _audio_duration(audio),
        "text": text,
        "words": text.split(),
    }
    with tempfile.TemporaryDirectory(prefix="mms-ctc-") as temp_dir:
        temp = Path(temp_dir)
        manifest = temp / "input.jsonl"
        output = temp / "output.jsonl"
        manifest.write_text(json.dumps(row, ensure_ascii=False) + "\n", encoding="utf-8")
        command = [
            str(python),
            str(CTC_SCRIPT),
            "--manifest",
            str(manifest),
            "--out",
            str(output),
            "--model",
            MODEL,
            "--device",
            device,
        ]
        completed = subprocess.run(command, text=True, capture_output=True, check=False)
        if completed.returncode:
            details = completed.stderr.strip() or completed.stdout.strip()
            raise RuntimeError(f"MMS CTC alignment failed ({completed.returncode}): {details}")
        if not output.exists():
            raise RuntimeError(
                "MMS CTC alignment produced no output: "
                + (completed.stdout.strip() or "unknown error")
            )
        result = json.loads(output.read_text(encoding="utf-8").splitlines()[0])
        return {"words": result["words"]}


def main() -> None:
    parser = argparse.ArgumentParser(description="Align WAV words with the MMS CTC aligner")
    parser.add_argument("audio", type=Path, help="Input WAV file")
    parser.add_argument("text", help="Transcript to align")
    parser.add_argument("--output", type=Path, help="Write JSON here instead of stdout")
    parser.add_argument("--device", default="cuda", help="Torch device used by MMS (default: cuda)")
    parser.add_argument(
        "--environment",
        type=Path,
        default=Path(os.environ.get("MMS_CTC_ENV", DEFAULT_ENV)),
        help="Dedicated uv environment (default: /home/yoadsnapir/time-align/mms-ctc/ctc-env)",
    )
    args = parser.parse_args()
    result = align(args.audio, args.text, device=args.device, environment=args.environment)
    serialized = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized, encoding="utf-8")
    else:
        sys.stdout.write(serialized)


if __name__ == "__main__":
    main()
