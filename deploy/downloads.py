"""Files the preview server offers for local inference: weights, tokenizer, config, the code
needed to sample like the server, `inference_guide.md`, and the semantic encoder versions
the model requires locally.

On Cloud Run `model.safetensors` lives in the model store bucket (`gs://<MODELS_BUCKET>/<sha256>/`),
so `/download/model.safetensors` redirects (302) to a 30-minute V4-signed GCS URL instead of
proxying 1.4 GB through the web server.
"""

from __future__ import annotations

import base64
import datetime
import hashlib
import html
import io
import json
import logging
import os
import urllib.parse
import urllib.request
import zipfile
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import yaml

from deploy.encoder_store import semantic_configs_of

logger = logging.getLogger("hebrew_tts_server.downloads")

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
WEIGHTS_NAME = "model.safetensors"
SIGNED_URL_EXPIRES_SECONDS = 1800  # 30 minutes
_METADATA_BASE = "http://metadata.google.internal/computeMetadata/v1"
_METADATA_HEADERS = {"Metadata-Flavor": "Google"}


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


def _metadata_get(path: str, timeout: float = 2.0) -> str:
    req = urllib.request.Request(f"{_METADATA_BASE}/{path.lstrip('/')}", headers=_METADATA_HEADERS)
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read().decode("utf-8").strip()


def sign_gcs_url(
    bucket: str,
    object_path: str,
    filename: str,
    expires_seconds: int = SIGNED_URL_EXPIRES_SECONDS,
    *,
    now: datetime.datetime | None = None,
    credentials_fn: Callable[[], tuple[str, str]] | None = None,
    sign_blob_fn: Callable[[str, str, bytes], bytes] | None = None,
) -> str:
    """Build a V4-signed `https://storage.googleapis.com/...` GET URL valid for `expires_seconds`
    (default 30 minutes) using the Cloud Run / GCE metadata server and the IAM Credentials
    `signBlob` API. Needs no private key file on disk."""
    if credentials_fn is not None:
        service_account, token = credentials_fn()
    else:
        service_account = _metadata_get("instance/service-accounts/default/email")
        token_raw = _metadata_get("instance/service-accounts/default/token")
        token = json.loads(token_raw)["access_token"]

    now = now or datetime.datetime.now(datetime.timezone.utc)
    datestamp = now.strftime("%Y%m%d")
    timestamp = now.strftime("%Y%m%dT%H%M%SZ")
    credential_scope = f"{datestamp}/auto/storage/goog4_request"
    params = {
        "X-Goog-Algorithm": "GOOG4-RSA-SHA256",
        "X-Goog-Credential": f"{service_account}/{credential_scope}",
        "X-Goog-Date": timestamp,
        "X-Goog-Expires": str(expires_seconds),
        "X-Goog-SignedHeaders": "host",
        "response-content-disposition": f'attachment; filename="{filename}"',
    }
    canonical_uri = (
        "/" + bucket.strip("/") + "/" + urllib.parse.quote(object_path.lstrip("/"), safe="/")
    )
    canonical_query = "&".join(
        f"{urllib.parse.quote(k, safe='')}={urllib.parse.quote(v, safe='')}"
        for k, v in sorted(params.items())
    )
    canonical_request = f"GET\n{canonical_uri}\n{canonical_query}\nhost:storage.googleapis.com\n\nhost\nUNSIGNED-PAYLOAD"
    digest = hashlib.sha256(canonical_request.encode("utf-8")).hexdigest()
    string_to_sign = f"GOOG4-RSA-SHA256\n{timestamp}\n{credential_scope}\n{digest}"
    if sign_blob_fn is not None:
        signature_bytes = sign_blob_fn(service_account, token, string_to_sign.encode("utf-8"))
    else:
        payload = base64.b64encode(string_to_sign.encode("utf-8")).decode("ascii")
        req = urllib.request.Request(
            f"https://iamcredentials.googleapis.com/v1/projects/-/serviceAccounts/{service_account}:signBlob",
            data=json.dumps({"payload": payload}).encode("utf-8"),
            headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=10.0) as resp:
            signature_bytes = base64.b64decode(json.loads(resp.read())["signedBlob"])
    signature = signature_bytes.hex()
    return f"https://storage.googleapis.com{canonical_uri}?{canonical_query}&X-Goog-Signature={signature}"


class Downloads:
    def __init__(
        self,
        root: Path,
        config: Path,
        sentences: Path,
        defaults: dict,
        model_sha256: Callable[[], str | None],
        encoder_info: Callable[[], dict | None] | None = None,
        models_bucket: str | None = None,
        weights_size: int | None = None,
        signer: Callable[[str, str, str], str] | None = None,
    ):
        self.root = root
        self.config = config
        self.defaults = defaults
        self.model_sha256 = model_sha256
        self.encoder_info = encoder_info
        self._models_bucket = models_bucket or os.environ.get("MODELS_BUCKET") or None
        self._weights_size = weights_size
        self._signer = signer
        self.semantic_configs = semantic_configs_of(config)
        cfg = yaml.safe_load(config.read_text(encoding="utf-8"))
        self.static = [
            StaticFile(
                WEIGHTS_NAME,
                root / cfg["weights_path"],
                "model weights (FlowLM + Mimi + semantic fusion) -> artifacts/model.safetensors",
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

    def _resolve_models_bucket(self) -> str | None:
        if self._models_bucket:
            return self._models_bucket
        if not os.environ.get("MODEL_STORE"):
            return None
        project = os.environ.get("GOOGLE_CLOUD_PROJECT") or os.environ.get("GCLOUD_PROJECT")
        if not project:
            try:
                project = _metadata_get("project/project-id")
            except (OSError, ValueError):
                return None
        if project:
            self._models_bucket = f"{project}-hebrew-tts-models"
        return self._models_bucket

    def signed_url(self, name: str) -> str | None:
        """For `model.safetensors` when served from a GCS model store: a 30-minute V4-signed
        GCS download URL (`None` for local files or when no store bucket is configured)."""
        if name != WEIGHTS_NAME:
            return None
        sha = self.model_sha256()
        if not sha:
            return None
        if self._signer is not None:
            bucket = self._models_bucket or "models"
            return self._signer(bucket, f"{sha}/{WEIGHTS_NAME}", WEIGHTS_NAME)
        bucket = self._resolve_models_bucket()
        if not bucket:
            return None
        try:
            return sign_gcs_url(
                bucket,
                f"{sha}/{WEIGHTS_NAME}",
                WEIGHTS_NAME,
                expires_seconds=SIGNED_URL_EXPIRES_SECONDS,
            )
        except (OSError, ValueError, KeyError) as exc:
            logger.warning("could not sign GCS URL for %s/%s: %s", bucket, sha, exc)
            return None

    def file(self, name: str) -> StaticFile | None:
        found = self._by_name.get(name)
        return found if found is not None and found.path.is_file() else None

    def _weights_human_size(self) -> str:
        weights = self._by_name[WEIGHTS_NAME].path
        if weights.is_file():
            return human_size(weights.stat().st_size)
        if self._weights_size:
            return human_size(self._weights_size)
        return ""

    def _semantic_entries(self) -> list[dict[str, str | None]]:
        raw_info = self.encoder_info() if self.encoder_info is not None else None
        info_list: list[dict] = []
        if isinstance(raw_info, dict):
            info_list = (
                list(raw_info["encoders"])
                if isinstance(raw_info.get("encoders"), list)
                else [raw_info]
            )
        entries: list[dict[str, str | None]] = []
        for idx, c in enumerate(self.semantic_configs):
            sha = None
            if idx < len(info_list) and isinstance(info_list[idx], dict):
                sha = info_list[idx].get("sha256")
            ref = f"{c.model_name}@{c.revision}" + (f"/{c.filename}" if c.filename else "")
            url = (
                f"https://huggingface.co/{c.model_name}/blob/{c.revision}/{c.filename}"
                if c.filename
                else f"https://huggingface.co/{c.model_name}/tree/{c.revision}"
            )
            layers = f"{c.first_layer}–{c.last_layer}"
            if c.encoder == "renikud_plus":
                details = (
                    f"~1.2 GB fp32 ONNX, file: {c.filename}, layers {layers}, "
                    f"onnx + onnxruntime, onnx_provider: {c.onnx_provider}"
                )
                main_file = c.filename or "model.onnx"
            else:
                details = (
                    f"~1.6 GB, files: config.json, model.safetensors, tokenizer.json, "
                    f"tokenizer_config.json, vocab.txt, layers {layers}, transformers"
                )
                main_file = "model.safetensors"
            entries.append(
                {
                    "encoder": c.encoder,
                    "model_name": c.model_name,
                    "revision": c.revision,
                    "filename": c.filename,
                    "main_file": main_file,
                    "ref": ref,
                    "url": url,
                    "layers": layers,
                    "details": details,
                    "sha256": sha,
                }
            )
        return entries

    def _semantic_markdown(self) -> str:
        entries = self._semantic_entries()
        if not entries:
            return (
                "Semantic encoders required by this model: **none** "
                "(no `semantic_conditioner` section in `artifacts/hebrew_24l.yaml`)."
            )
        key = (
            "flow_lm.semantic_conditioners" if len(entries) > 1 else "flow_lm.semantic_conditioner"
        )
        lines = [
            f"Semantic encoder(s) required by this model (`{key}` in `artifacts/hebrew_24l.yaml`"
            + ("; branch order is part of the checkpoint identity):" if len(entries) > 1 else "):")
        ]
        for idx, e in enumerate(entries, 1):
            sha_note = f", `{e['main_file']}` SHA-256: `{e['sha256']}`" if e["sha256"] else ""
            lines.append(f"{idx}. **`{e['encoder']}`** — `{e['ref']}` ({e['details']}{sha_note})")
        return "\n".join(lines)

    # -- generated files ----------------------------------------------------------------

    def guide(self) -> str | None:
        """The guide with this model's hash, semantic encoders and defaults filled in; None
        while the hash is still being computed."""
        sha = self.model_sha256()
        if sha is None:
            return None
        size_str = self._weights_human_size()
        if not size_str:
            return None
        table = "| setting | default |\n|---|---|\n" + "\n".join(
            f"| `{key}` | `{value}` |" for key, value in self.defaults.items()
        )
        return (
            (DEPLOY_DIR / GUIDE)
            .read_text(encoding="utf-8")
            .replace("{{MODEL_SHA256}}", sha)
            .replace("{{SEMANTIC_ENCODERS}}", self._semantic_markdown())
            .replace("{{WEIGHTS_SIZE}}", size_str)
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
            if f.name == WEIGHTS_NAME:
                size = self._weights_human_size()
            else:
                size = human_size(f.path.stat().st_size) if f.path.is_file() else ""
            items.append(item(f.name, f.description, size))
        items.append(
            item(
                CODE_ZIP,
                "pocket_tts + semantic conditioners + sampler code, hebrew_tts_infer.py (example), requirements.txt",
            )
        )
        items.append(item(self.static[3].name, self.static[3].description))

        entries = self._semantic_entries()
        if entries:
            enc_items = []
            for e in entries:
                sha_html = (
                    f"; <code>{html.escape(e['main_file'] or '')}</code> sha256: "
                    f"<code>{html.escape(e['sha256'])}</code>"
                    if e["sha256"]
                    else ""
                )
                enc_items.append(
                    f"<li><code>{html.escape(e['encoder'] or '')}</code>: "
                    f'<a href="{html.escape(e["url"] or "")}" target="_blank" rel="noopener">'
                    f"<code>{html.escape(e['ref'] or '')}</code></a> "
                    f'<span class="meta">({html.escape(e["details"] or "")}{sha_html})</span></li>'
                )
            mode = (
                f"combined: {' + '.join(str(e['encoder']) for e in entries)}"
                if len(entries) > 1
                else str(entries[0]["encoder"])
            )
            sem_html = (
                f'<div style="margin-top:.55rem"><b>Semantic encoder(s) required locally '
                f"({html.escape(mode)})</b> "
                '<span class="meta">- pinned in <code>hebrew_24l.yaml</code>; fetched '
                "automatically from Hugging Face on first run, or pass "
                "<code>--semantic-encoder-dir</code></span></div>"
                f"<ul>{''.join(enc_items)}</ul>"
            )
        else:
            sem_html = (
                '<div class="meta" style="margin-top:.45rem">'
                "<b>Semantic encoders required locally:</b> none (SentencePiece lookup table only)."
                "</div>"
            )
        return "<ul>" + "".join(items) + "</ul>" + sem_html
