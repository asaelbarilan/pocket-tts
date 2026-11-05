"""Encoder store: the frozen semantic encoders (DictaBERT, RenikudPlus) live next to the models,
in the same store, keyed by the sha256 of their main weights file.

    <store>/encoders/<sha256>/manifest.json
    <store>/encoders/<sha256>/<the encoder's files>

    dictabert     config.json model.safetensors tokenizer.json tokenizer_config.json vocab.txt
    renikud_plus  model.onnx                                   (the fp32 file, never the int8 one)

The manifest names what the files are:

    {"format": 1, "encoder": "dictabert", "model_name": "dicta-il/dictabert-large",
     "revision": "<commit>", "main_file": "model.safetensors",
     "files": {"<name>": {"sha256": ..., "size": ...}, ...}}

A model says which encoder it needs in its own config (`flow_lm.semantic_conditioner`: the
encoder, `model_name`, `revision` and, for RenikudPlus, `filename`), because its fusion weights
are tied to exactly that encoder. So the server needs no extra setting: it finds the published
encoder that matches the config, copies its files to local disk hashing every byte, and loads
the encoder from there. A model without a `semantic_conditioner` section needs none.

A combined model (`flow_lm.semantic_conditioners`, two entries) needs both: `EncoderStoreSet`
holds one `EncoderStore` per entry, in the config's order, and hands the model one directory
per encoder.

Environment (both optional):

    SEMANTIC_ENCODER_STORE   where encoders are copied from; default `<MODEL_STORE>/encoders`
    SEMANTIC_ENCODER_SHA256  pin one encoder by hash instead of finding it by identity. It must
                             still be the encoder the model's config names: a pin cannot make a
                             model run with another encoder. For a combined model, one hash
                             per encoder in the config's order, comma-separated (an empty
                             entry leaves that encoder to be found by identity).

`deploy/publish_encoder.sh` writes the layout; a directory is never overwritten.

Pure Python plus `pocket_tts.utils.config`: no torch, so the tests need no model.
"""

from __future__ import annotations

import json
import logging
import shutil
from pathlib import Path

import yaml

from deploy.model_store import copy_verified, normalize_sha
from pocket_tts.utils.config import SemanticConditionerConfig, SemanticSections

logger = logging.getLogger("encoder_store")

ENCODERS_DIR = "encoders"
MANIFEST = "manifest.json"
# Where the files are copied to, relative to the repo root.
LOCAL_DIR = "semantic-encoders"

# encoder -> (main file, every file the encoder needs). The main file's sha256 is the encoder's
# identity. DictaBERT's are the files `from_pretrained` reads (not the duplicate
# pytorch_model.bin).
DICTABERT_FILES = (
    "config.json",
    "model.safetensors",
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.txt",
)


def main_file(config: SemanticConditionerConfig) -> str:
    return config.filename if config.encoder == "renikud_plus" else "model.safetensors"


def required_files(config: SemanticConditionerConfig) -> tuple[str, ...]:
    if config.encoder == "renikud_plus":
        return (config.filename,)
    return DICTABERT_FILES


def semantic_configs_of(config_path: Path) -> list[SemanticConditionerConfig]:
    """The encoders a model config names, in order: none (the model runs without semantic
    conditioning), one (`flow_lm.semantic_conditioner`, or a list of one) or two
    (`flow_lm.semantic_conditioners`, the combined mode). Reads the YAML only: the weights and
    the other sections are not touched."""
    raw = yaml.safe_load(Path(config_path).read_text(encoding="utf-8")) or {}
    flow_lm = raw.get("flow_lm") or {}
    keys = ("semantic_conditioner", "semantic_conditioners")
    return SemanticSections(
        **{k: flow_lm[k] for k in keys if flow_lm.get(k) is not None}
    ).semantic_encoders


def semantic_config_of(config_path: Path) -> SemanticConditionerConfig | None:
    """The single encoder of a model config, None when it has none. A combined config has
    several: use `semantic_configs_of`."""
    configs = semantic_configs_of(config_path)
    if len(configs) > 1:
        raise ValueError(f"{config_path} names {len(configs)} encoders; use semantic_configs_of")
    return configs[0] if configs else None


def describe(config: SemanticConditionerConfig) -> str:
    where = f"{config.model_name}@{config.revision}"
    if config.encoder == "renikud_plus":
        where += f"/{config.filename}"
    return f"{config.encoder} ({where})"


def _manifest_matches(manifest: dict, config: SemanticConditionerConfig) -> bool:
    return (
        manifest.get("encoder") == config.encoder
        and manifest.get("model_name") == config.model_name
        and manifest.get("revision") == config.revision
        and manifest.get("main_file") == main_file(config)
    )


class EncoderStore:
    """The published encoder one model's config needs, found in a store directory."""

    def __init__(
        self, store: Path, config: SemanticConditionerConfig, root: Path, pin: str | None = None
    ):
        self.config = config
        self.root = root
        self.store = Path(store)
        self.dir = self._resolve(pin)
        manifest = json.loads((self.dir / MANIFEST).read_text(encoding="utf-8"))
        self.manifest = manifest
        self.files: dict[str, dict] = manifest["files"]
        for name in required_files(config):
            if name not in self.files:
                raise ValueError(f"{self.dir}: the manifest lists no {name}")
        self.sha256 = self.files[main_file(config)]["sha256"]
        if self.dir.name != self.sha256:
            raise ValueError(
                f"{self.dir}: the manifest's {main_file(config)} hash {self.sha256} is not the "
                "directory name"
            )

    def _resolve(self, pin: str | None) -> Path:
        wanted = describe(self.config)
        if pin:
            directory = self.store / normalize_sha(pin)
            manifest = self._read(directory)
            if manifest is None:
                raise FileNotFoundError(
                    f"encoder {pin} is not in the store: {directory / MANIFEST} is missing "
                    "(publish it with deploy/publish_encoder.sh)"
                )
            if not _manifest_matches(manifest, self.config):
                have = (
                    f"{manifest.get('encoder')} "
                    f"({manifest.get('model_name')}@{manifest.get('revision')})"
                )
                raise ValueError(
                    f"SEMANTIC_ENCODER_SHA256 {pin} is {have}, but the model's config needs "
                    f"{wanted}: a pin selects the bytes, not another encoder"
                )
            return directory
        found = []
        if self.store.is_dir():
            for directory in sorted(self.store.iterdir()):
                manifest = self._read(directory)  # a directory without one is an interrupted upload
                if manifest is not None and _manifest_matches(manifest, self.config):
                    found.append(directory)
        if not found:
            raise FileNotFoundError(
                f"no published encoder matches the model's config: {wanted} is not in "
                f"{self.store} (publish it with deploy/publish_encoder.sh {self.config.encoder})"
            )
        if len(found) > 1:
            names = ", ".join(d.name for d in found)
            raise ValueError(
                f"{len(found)} published encoders match {wanted} ({names}): "
                "pin one with SEMANTIC_ENCODER_SHA256"
            )
        return found[0]

    @staticmethod
    def _read(directory: Path) -> dict | None:
        path = directory / MANIFEST
        if not path.is_file():
            return None
        return json.loads(path.read_text(encoding="utf-8"))

    @property
    def local_dir(self) -> Path:
        return self.root / LOCAL_DIR / self.sha256

    def fetch(self) -> Path:
        """Copy the encoder's files to local disk, verifying each against the manifest, and
        return the directory (what `TTSModel.load_model(semantic_encoder_dir=...)` takes). A
        file that does not match raises and leaves nothing under its name."""
        for name in required_files(self.config):
            copy_verified(self.dir / name, self.local_dir / name, self.files[name]["sha256"])
        return self.local_dir

    def discard(self) -> None:
        """Delete the local copy once the encoder is loaded: Cloud Run's disk is memory, and
        the copy is up to 1.6 GB of it."""
        shutil.rmtree(self.local_dir, ignore_errors=True)

    def info(self) -> dict:
        """What `/health` reports."""
        return {
            "encoder": self.config.encoder,
            "model_name": self.config.model_name,
            "revision": self.config.revision,
            "sha256": self.sha256,
        }


class EncoderStoreSet:
    """The published encoders a combined model's config needs: one EncoderStore per encoder,
    in the config's order. Offers what the server uses of a single EncoderStore."""

    def __init__(
        self,
        store: Path,
        configs: list[SemanticConditionerConfig],
        root: Path,
        pins: list[str | None] | None = None,
    ):
        pins = pins or [None] * len(configs)
        if len(pins) != len(configs):
            raise ValueError(
                f"{len(pins)} encoder hashes pinned for {len(configs)} encoders: give one per "
                "encoder in the config's order (an empty one to find that encoder by identity)"
            )
        self.stores = [EncoderStore(store, c, root, p) for c, p in zip(configs, pins, strict=True)]
        if len({s.sha256 for s in self.stores}) < len(self.stores):
            raise ValueError("the combined model's encoders resolve to the same published files")

    @property
    def sha256(self) -> str:
        return ",".join(s.sha256 for s in self.stores)

    def fetch(self) -> list[Path]:
        """What `TTSModel.load_model(semantic_encoder_dir=...)` takes: one directory per encoder."""
        return [s.fetch() for s in self.stores]

    def discard(self) -> None:
        for s in self.stores:
            s.discard()

    def info(self) -> dict:
        return {"encoders": [s.info() for s in self.stores]}
