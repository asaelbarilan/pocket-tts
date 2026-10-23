"""Files the preview server offers for local inference: weights, tokenizer, config, the code
needed to sample like the server, and `inference_guide.md`."""

from __future__ import annotations

import html
import io
import zipfile
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import yaml

DEPLOY_DIR = Path(__file__).resolve().parent

# The part of `training/` that `hebrew_tts_infer.py` imports (checked in a clean environment).
TRAINING_FILES = [
    "__init__.py",
    "args.py",
    "curriculum.py",
    "text_augment.py",
    "modules/__init__.py",
    "modules/model.py",
    "modules/samplers.py",
    "modules/conditioner.py",
    "modules/utils.py",
]

CODE_ZIP = "hebrew_tts_inference_code.zip"
GUIDE = "inference_guide.md"


@dataclass
class StaticFile:
    name: str
    path: Path
    description: str


def human_size(size: int) -> str:
    if size >= 10**9:
        return f"{size / 1e9:.2f} GB"
    if size >= 10**6:
        return f"{size / 1e6:.0f} MB"
    return f"{max(1, round(size / 1e3))} KB"


class Downloads:
    def __init__(
        self,
        root: Path,
        config: Path,
        sentences: Path,
        defaults: dict,
        model_sha256: Callable[[], str | None],
    ):
        self.root = root
        self.defaults = defaults
        self.model_sha256 = model_sha256
        cfg = yaml.safe_load(config.read_text(encoding="utf-8"))
        self.static = [
            StaticFile(
                "model.safetensors",
                root / cfg["weights_path"],
                "model weights (FlowLM + Mimi) -> artifacts/model.safetensors",
            ),
            StaticFile(
                "tokenizer.model",
                root / cfg["flow_lm"]["lookup_table"]["tokenizer_path"],
                "SentencePiece tokenizer -> tokenizers/tokenizer.model",
            ),
            StaticFile("hebrew_24l.yaml", config, "model config -> artifacts/hebrew_24l.yaml"),
            StaticFile(
                "eval_sentences_he.jsonl", sentences, "optional: the 40 validation sentences"
            ),
        ]
        self._by_name = {f.name: f for f in self.static}

    def file(self, name: str) -> StaticFile | None:
        found = self._by_name.get(name)
        return found if found is not None and found.path.is_file() else None

    # -- generated files ----------------------------------------------------------------

    def guide(self) -> str | None:
        """The guide with this model's hash and defaults filled in; None while the hash is
        still being computed."""
        sha = self.model_sha256()
        if sha is None:
            return None
        weights = self._by_name["model.safetensors"].path
        table = "| setting | default |\n|---|---|\n" + "\n".join(
            f"| `{key}` | `{value}` |" for key, value in self.defaults.items()
        )
        return (
            (DEPLOY_DIR / GUIDE)
            .read_text(encoding="utf-8")
            .replace("{{MODEL_SHA256}}", sha)
            .replace("{{WEIGHTS_SIZE}}", human_size(weights.stat().st_size))
            .replace("{{DEFAULTS_TABLE}}", table)
        )

    def code_zip(self, guide: str) -> bytes:
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
            for path in sorted((self.root / "pocket_tts").rglob("*")):
                if path.is_file() and "__pycache__" not in path.parts and path.suffix != ".pyc":
                    archive.write(path, path.relative_to(self.root).as_posix())
            for name in TRAINING_FILES:
                archive.write(self.root / "training" / name, f"training/{name}")
            archive.write(DEPLOY_DIR / "hebrew_tts_infer.py", "hebrew_tts_infer.py")
            archive.write(DEPLOY_DIR / "requirements-inference.txt", "requirements.txt")
            archive.writestr(GUIDE, guide)
        return buffer.getvalue()

    # -- the start page -----------------------------------------------------------------

    def html(self) -> str:
        def item(name: str, description: str, size: str = "") -> str:
            return (
                f'<li><a href="/download/{html.escape(name)}" download>{html.escape(name)}</a>'
                f"{' <span class=meta>(' + size + ')</span>' if size else ''}"
                f' <span class="meta">- {html.escape(description)}</span></li>'
            )

        items = [
            item(GUIDE, "start here: what a coding agent needs to run inference like this server")
        ]
        for f in self.static[:3]:
            size = human_size(f.path.stat().st_size) if f.path.is_file() else ""
            items.append(item(f.name, f.description, size))
        items.append(
            item(
                CODE_ZIP,
                "pocket_tts + sampler code, hebrew_tts_infer.py (example), requirements.txt",
            )
        )
        items.append(item(self.static[3].name, self.static[3].description))
        return "<ul>" + "".join(items) + "</ul>"
