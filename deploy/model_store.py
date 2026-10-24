"""Model store: the weights, config and tokenizer live outside the image, in a directory
`<store>/<weights sha256>/` (on Cloud Run a read-only mounted Cloud Storage bucket).

    <store>/<sha256>/manifest.json
    <store>/<sha256>/model.safetensors
    <store>/<sha256>/hebrew_24l.yaml
    <store>/<sha256>/tokenizer.model

The sha256 of the weights is the model's identity: the server is pinned to one (env
MODEL_SHA256) and refuses to serve bytes that hash to anything else. `deploy/publish_model.sh`
writes the layout; a directory is never overwritten, so a hash always means the same model.

At startup the files are copied to the repo-relative paths the config expects
(`artifacts/`, `tokenizers/`), hashing every byte on the way, so nothing is read twice and
the model is read from local disk rather than through the FUSE mount.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
from pathlib import Path

logger = logging.getLogger("model_store")

MANIFEST = "manifest.json"
WEIGHTS = "model.safetensors"
# store file name -> path relative to the repo root, where the config looks for it
LOCAL_PATHS = {
    WEIGHTS: "artifacts/model.safetensors",
    "hebrew_24l.yaml": "artifacts/hebrew_24l.yaml",
    "tokenizer.model": "tokenizers/tokenizer.model",
}
SMALL_FILES = ("hebrew_24l.yaml", "tokenizer.model")
SHA_RE = re.compile(r"[0-9a-f]{64}")
CHUNK = 1 << 22


def normalize_sha(value: str) -> str:
    sha = value.strip().lower()
    if not SHA_RE.fullmatch(sha):
        raise ValueError(f"not a sha256: {value!r}")
    return sha


def copy_verified(source: Path, dest: Path, expected_sha256: str) -> None:
    """Copies `source` to `dest` and hashes it in the same pass. `dest` only appears once its
    hash matches; a mismatch raises and leaves nothing behind."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    partial = dest.with_name(dest.name + ".partial")
    digest = hashlib.sha256()
    try:
        with source.open("rb") as src, partial.open("wb") as out:
            while chunk := src.read(CHUNK):
                digest.update(chunk)
                out.write(chunk)
        if digest.hexdigest() != expected_sha256:
            raise ValueError(
                f"{source.name}: sha256 is {digest.hexdigest()}, the manifest says {expected_sha256}"
            )
        os.replace(partial, dest)
    finally:
        partial.unlink(missing_ok=True)


class ModelStore:
    """One pinned model in a store directory."""

    def __init__(self, store: Path, sha256: str, root: Path):
        self.sha256 = normalize_sha(sha256)
        self.root = root
        self.dir = store / self.sha256
        if not (self.dir / MANIFEST).is_file():
            raise FileNotFoundError(
                f"model {self.sha256} is not in the store: {self.dir / MANIFEST} is missing "
                "(publish it with deploy/publish_model.sh)"
            )
        manifest = json.loads((self.dir / MANIFEST).read_text(encoding="utf-8"))
        self.files: dict[str, dict] = manifest["files"]
        for name in LOCAL_PATHS:
            if name not in self.files:
                raise ValueError(f"manifest lists no {name}")
        if self.files[WEIGHTS]["sha256"] != self.sha256:
            raise ValueError(
                f"{self.dir}: manifest weights hash {self.files[WEIGHTS]['sha256']} is not the "
                f"directory name"
            )
        self.manifest = manifest

    def local_path(self, name: str) -> Path:
        return self.root / LOCAL_PATHS[name]

    def fetch(self, name: str) -> Path:
        dest = self.local_path(name)
        copy_verified(self.dir / name, dest, self.files[name]["sha256"])
        return dest

    def fetch_small(self) -> None:
        """Config and tokenizer: tiny, needed before the server starts."""
        for name in SMALL_FILES:
            self.fetch(name)

    def fetch_weights(self) -> Path:
        return self.fetch(WEIGHTS)
