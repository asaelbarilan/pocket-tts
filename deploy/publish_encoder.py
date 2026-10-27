"""Stage a semantic encoder for the encoder store (the helper of deploy/publish_encoder.sh).

    python -m deploy.publish_encoder dictabert    --out DIR
    python -m deploy.publish_encoder renikud_plus --out DIR [--revision <commit>]

Downloads the encoder's files from Hugging Face at the pinned revision (the defaults of
`SemanticConditionerConfig`, the revision the shipped configs pin), hashes them and writes
DIR/manifest.json, the layout described in deploy/encoder_store.py. Prints the encoder's sha256
(the hash of its main file) on the first line and then each file to upload, one per line. The
shell script uploads them; nothing here touches the store.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

from deploy.encoder_store import MANIFEST, main_file, required_files
from pocket_tts.utils.config import SemanticConditionerConfig

CHUNK = 1 << 22


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(CHUNK):
            digest.update(chunk)
    return digest.hexdigest()


def stage(config: SemanticConditionerConfig, out: Path, note: str = "") -> tuple[str, list[str]]:
    """Download `config`'s encoder into `out`, write the manifest, return (sha256, file names)."""
    from huggingface_hub import snapshot_download

    names = list(required_files(config))
    snapshot_download(
        config.model_name, revision=config.revision, allow_patterns=names, local_dir=out
    )
    files = {}
    for name in names:
        path = out / name
        if not path.is_file() or path.stat().st_size == 0:
            raise SystemExit(f"{config.model_name}@{config.revision} has no {name}")
        files[name] = {"sha256": sha256_of(path), "size": path.stat().st_size}
    manifest = {
        "format": 1,
        "created": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "note": note,
        "encoder": config.encoder,
        "model_name": config.model_name,
        "revision": config.revision,
        "main_file": main_file(config),
        "files": files,
    }
    (out / MANIFEST).write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return files[main_file(config)]["sha256"], names


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("encoder", choices=["dictabert", "renikud_plus"])
    parser.add_argument("--out", type=Path, required=True, help="staging directory")
    parser.add_argument("--revision", help="a Hugging Face commit (default: the pinned one)")
    parser.add_argument("--note", default="")
    args = parser.parse_args()
    overrides = {"revision": args.revision} if args.revision else {}
    config = SemanticConditionerConfig(encoder=args.encoder, **overrides)
    args.out.mkdir(parents=True, exist_ok=True)
    sha, names = stage(config, args.out, args.note)
    print(sha)
    print(*names, sep="\n")
    print(f"{args.encoder} {config.model_name}@{config.revision} -> {sha}", file=sys.stderr)


if __name__ == "__main__":
    main()
