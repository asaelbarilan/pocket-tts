"""Create the isolated Viter runtime and download the pretrained Hebrew grapheme assets."""

from __future__ import annotations

import argparse
import hashlib
import os
import subprocess
import urllib.request
from pathlib import Path

DEFAULT_ENV = Path("/home/yoadsnapir/time-align/viter-env")
ASSETS = {
    "hebrew_graphemes.viter": "8bf5a759058e9ad12fe833d55063e7bcdae5cd905300777774d70b0e65019631",
    "hebrew_graphemes.dict": "4c8b97e2b707fd72f144ef34476b2625bee2bc3fa2cf0153543d263f730596c4",
}
RELEASE = "https://github.com/thewh1teagle/viter/releases/download/models-v1.0"


def digest(path: Path) -> str:
    checksum = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            checksum.update(block)
    return checksum.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--environment", type=Path, default=Path(os.environ.get("VITER_ENV", DEFAULT_ENV))
    )
    args = parser.parse_args()
    environment = args.environment.expanduser().resolve()
    python = environment / "bin/python"
    if not python.is_file():
        subprocess.run(["uv", "venv", "--python", "3.11", str(environment)], check=True)
    subprocess.run(
        [
            "uv",
            "pip",
            "install",
            "--python",
            str(python),
            "-r",
            str(Path(__file__).with_name("viter_requirements.txt")),
        ],
        check=True,
    )
    models = environment / "models"
    models.mkdir(parents=True, exist_ok=True)
    for name, expected in ASSETS.items():
        destination = models / name
        if destination.is_file() and digest(destination) == expected:
            print(f"Verified {destination}", flush=True)
            continue
        temporary = destination.with_suffix(destination.suffix + ".download")
        try:
            print(f"Downloading {name}", flush=True)
            urllib.request.urlretrieve(f"{RELEASE}/{name}", temporary)
            if digest(temporary) != expected:
                raise ValueError(f"SHA-256 mismatch for {name}")
            temporary.replace(destination)
        finally:
            temporary.unlink(missing_ok=True)
    print(f"Viter ready: {python}", flush=True)


if __name__ == "__main__":
    main()
