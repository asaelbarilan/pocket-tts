"""Export batches of the training dataloader as a static site, and upload it to S3.

    uv run --with boto3 python -m hebrew_training.data_preview.export_s3 \\
        training/configs/finetune_hebrew_curriculum.yaml \\
        --s3 s3://my-bucket/previews/hebrew-phase0 --phase 0 --batches 20

Draws `--batches` batches from the same loader the preview server uses (so the same cuts,
mixing and digit/niqqud draws as training), writes every sample's prompt and target audio plus
one `index.html` that steps through the batches with Prev / Next, and uploads the folder. The
site is plain files: serve the prefix as is (S3 static website hosting, CloudFront, or any web
server). Alpine.js is loaded from a CDN by the page, there is nothing to build.

    index.html  style.css  audio/b0001/s0-prompt.wav  audio/b0001/s0-target.wav  ...

The loader is set up as on the live page: `--split`, `--dataset`, and every loader parameter
(`--seed`, `--batch-size`, `--nikud-word-prob`, ...) default to the config's values for that
split, and `--skip N` drops the first N batches. `--max-rows N` gives the loader a random subset
of each manifest (see manifest_sample.py): use it for manifests of millions of rows.

Credentials come from the usual environment variables: AWS_ACCESS_KEY_ID and
AWS_SECRET_ACCESS_KEY (plus AWS_SESSION_TOKEN for temporary credentials, and AWS_DEFAULT_REGION
or AWS_REGION). They are checked before any batch is built. No ACL is set on the objects:
who may read the site is up to the bucket policy. Files of an earlier export under the same
prefix are overwritten, not deleted.

Without `--s3`, only the site is written (to `--out-dir`), and no credentials are needed.
"""

from __future__ import annotations

import argparse
import io
import json
import logging
import os
import shutil
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse

from pydantic import ValidationError

from hebrew_training.data_preview.server import (
    DEFAULT_CACHE_DIR,
    REPO_ROOT,
    STYLE_CSS,
    NextRequest,
    Params,
    PreviewError,
    Session,
    Spec,
)

logger = logging.getLogger("data_preview")

TEMPLATE = Path(__file__).resolve().parent / "static_site.html"
UPLOAD_WORKERS = 8
CONTENT_TYPES = {
    ".html": "text/html; charset=utf-8",
    ".css": "text/css; charset=utf-8",
    ".wav": "audio/wav",
    ".flac": "audio/flac",
}
# CLI flag -> Params field, for the loader parameters that can be set from the command line.
PARAM_FLAGS = {
    "batch_size": int,
    "seed": int,
    "max_duration_sec": float,
    "max_voice_prompt_sec": float,
    "alt_words_sample_prob": float,
    "alt_word_swap_prob": float,
    "nikud_word_prob": float,
    "max_rows": int,
    "sample_seed": int,
}
SPLITS = {"train": "train", "valid": "valid", "eval": "valid"}


class ExportError(Exception):
    """A problem with the request, reported without a traceback."""


def parse_s3_uri(uri: str) -> tuple[str, str]:
    """`s3://bucket/some/prefix` -> ("bucket", "some/prefix/"); the prefix is "" at the root."""
    parsed = urlparse(uri)
    if parsed.scheme != "s3" or not parsed.netloc:
        raise ExportError(f"--s3 must look like s3://bucket/prefix, got {uri!r}")
    prefix = parsed.path.strip("/")
    return parsed.netloc, f"{prefix}/" if prefix else ""


def check_credentials() -> None:
    missing = [
        name for name in ("AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY") if not os.environ.get(name)
    ]
    if missing:
        raise ExportError(f"set {' and '.join(missing)} in the environment to upload to S3")


def s3_client(endpoint_url: str | None):
    try:
        import boto3
    except ImportError:
        raise ExportError(
            "uploading needs boto3, which is not installed: run this with "
            "`uv run --with boto3 python -m hebrew_training.data_preview.export_s3 ...`"
        ) from None
    return boto3.client(
        "s3",
        region_name=os.environ.get("AWS_REGION") or os.environ.get("AWS_DEFAULT_REGION"),
        endpoint_url=endpoint_url,
    )


def encode_audio(batch, index: int, kind: str, audio_format: str) -> bytes:
    if audio_format == "wav":
        return batch.wav(index, kind)
    import soundfile

    audio = (batch.prompts if kind == "prompt" else batch.targets)[index]
    buffer = io.BytesIO()
    soundfile.write(buffer, audio, batch.sample_rate, format="FLAC", subtype="PCM_16")
    return buffer.getvalue()


def build_params(session: Session, split: str, args: argparse.Namespace) -> Params:
    values = dict(session.info()["defaults"][split])
    for field in PARAM_FLAGS:
        if getattr(args, field) is not None:
            values[field] = getattr(args, field)
    for field in ("shuffle", "fixed_draws"):
        if getattr(args, field) is not None:
            values[field] = getattr(args, field)
    try:
        return Params(**values)
    except ValidationError as exc:
        problems = "; ".join(f"{e['loc'][0]}: {e['msg']}" for e in exc.errors())
        raise ExportError(f"bad loader parameter: {problems}") from exc


def build_site(
    session: Session, spec: Spec, batches: int, skip: int, out_dir: Path, audio_format: str
) -> Path:
    """Draw the batches and write the site into `out_dir`; returns its index.html."""
    out_dir.mkdir(parents=True, exist_ok=True)
    exported, last = [], None
    for n in range(batches):
        request = NextRequest(spec=spec, skip=skip if n == 0 else 0, reload=n == 0)
        try:
            last = session.next(request)
        except PreviewError as exc:
            raise ExportError(str(exc)) from exc
        loaded = session.batch
        folder = f"audio/b{n + 1:04d}"
        (out_dir / folder).mkdir(parents=True, exist_ok=True)
        samples = last["samples"]
        for sample in samples:
            sample.pop("prompt_url")
            sample.pop("target_url")
            for kind in ("prompt", "target"):
                name = f"{folder}/s{sample['index']}-{kind}.{audio_format}"
                (out_dir / name).write_bytes(
                    encode_audio(loaded, sample["index"], kind, audio_format)
                )
                sample[f"{kind}_file"] = name
        exported.append({"number": last["number"], "dataset": last["dataset"], "samples": samples})
        print(f"batch {n + 1}/{batches} from {last['dataset']}", flush=True)

    site = {
        "config": session.config_path.name,
        "generated": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        "phase": spec.phase,
        "split": spec.split,
        "dataset": spec.dataset,
        "skip": skip,
        "mix": session.mix,
        "params": spec.params.model_dump(),
        "sample_rate": session.sample_rate,
        "tokenizer": session.tokenizer_path,
        "rows": last["status"]["rows"] if last else {},
        "batches": exported,
    }
    # Inside a <script> element the only thing that must not appear in the data is `</`.
    data = json.dumps(site, ensure_ascii=False).replace("</", "<\\/")
    index = out_dir / "index.html"
    index.write_text(TEMPLATE.read_text().replace("__DATA__", data), encoding="utf-8")
    shutil.copyfile(STYLE_CSS, out_dir / "style.css")
    return index


def upload_site(out_dir: Path, bucket: str, prefix: str, endpoint_url: str | None) -> list[str]:
    """Upload every file under `out_dir`, index.html last so a half-uploaded site is not served."""
    client = s3_client(endpoint_url)
    files = sorted(p for p in out_dir.rglob("*") if p.is_file())
    files.sort(key=lambda p: p.name == "index.html")  # stable: index.html moves to the end
    keys = [prefix + p.relative_to(out_dir).as_posix() for p in files]

    def put(path: Path, key: str) -> None:
        extra = {"ContentType": CONTENT_TYPES.get(path.suffix, "application/octet-stream")}
        if path.name == "index.html":
            extra["CacheControl"] = "no-cache"  # a re-export under the same prefix shows at once
        client.upload_file(str(path), bucket, key, ExtraArgs=extra)

    assets = [(p, k) for p, k in zip(files, keys, strict=True) if p.name != "index.html"]
    done = 0
    with ThreadPoolExecutor(max_workers=UPLOAD_WORKERS) as pool:
        for _ in pool.map(lambda item: put(*item), assets):
            done += 1
            if done % 25 == 0 or done == len(assets):
                print(f"uploaded {done}/{len(assets)} files", flush=True)
    put(out_dir / "index.html", prefix + "index.html")
    return keys


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__.split("\n\n")[0], formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("config", type=Path, help="training config yaml")
    parser.add_argument("--s3", metavar="s3://BUCKET/PREFIX", help="where to upload the site")
    parser.add_argument(
        "--out-dir",
        type=Path,
        help="also keep the site in this folder (with no --s3: the only output)",
    )
    parser.add_argument("--batches", type=int, required=True, help="number of batches to export")
    parser.add_argument("--phase", type=int, default=0, help="curriculum phase, 0-based")
    parser.add_argument("--split", choices=sorted(SPLITS), default="train")
    parser.add_argument("--dataset", help="draw from this dataset only, not the phase's mix")
    parser.add_argument("--skip", type=int, default=0, help="batches to draw and drop first")
    parser.add_argument("--audio-format", choices=["wav", "flac"], default="wav")
    parser.add_argument("--tokenizer", help="sentencepiece model to use instead of the config's")
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--endpoint-url", help="S3-compatible endpoint (MinIO, R2, ...)")
    loader = parser.add_argument_group("loader parameters (default: the config's, per split)")
    for field, kind in PARAM_FLAGS.items():
        loader.add_argument(f"--{field.replace('_', '-')}", type=kind, default=None)
    loader.add_argument("--shuffle", action=argparse.BooleanOptionalAction, default=None)
    loader.add_argument("--fixed-draws", action=argparse.BooleanOptionalAction, default=None)
    args = parser.parse_args(argv)
    if args.batches < 1:
        parser.error("--batches must be at least 1")
    if args.skip < 0:
        parser.error("--skip must be >= 0")
    if not args.s3 and not args.out_dir:
        parser.error("give --s3 to upload, --out-dir to only write the site, or both")
    return args


def run(args: argparse.Namespace) -> None:
    bucket = prefix = None
    if args.s3:
        bucket, prefix = parse_s3_uri(args.s3)
        check_credentials()
        s3_client(args.endpoint_url)  # fails now if boto3 is missing, not after the batches

    config_path = args.config.resolve()
    tokenizer = str(Path(args.tokenizer).resolve()) if args.tokenizer else None
    cache_dir = args.cache_dir.expanduser().resolve()
    out_dir = args.out_dir.expanduser().resolve() if args.out_dir else None
    os.chdir(REPO_ROOT)  # the config's relative paths are relative to the repo root

    try:
        session = Session(config_path, tokenizer, 0, cache_dir)
    except (FileNotFoundError, ValueError, KeyError) as exc:
        raise ExportError(f"cannot set up {config_path.name}: {exc}") from exc
    split = SPLITS[args.split]
    try:
        spec = Spec(
            phase=args.phase,
            split=split,
            dataset=args.dataset,
            params=build_params(session, split, args),
        )
    except ValidationError as exc:
        raise ExportError(f"bad request: {exc}") from exc

    scratch = None if out_dir else Path(tempfile.mkdtemp(prefix="data-preview-"))
    site_dir = out_dir or scratch
    try:
        index = build_site(session, spec, args.batches, args.skip, site_dir, args.audio_format)
        print(f"site written to {site_dir}")
        if bucket is not None:
            keys = upload_site(site_dir, bucket, prefix, args.endpoint_url)
            print(f"uploaded {len(keys)} files to s3://{bucket}/{prefix}")
            print(f"entry point: s3://{bucket}/{prefix}index.html")
        else:
            print(f"open {index} (or serve the folder) to browse it")
    finally:
        session.close()
        if scratch is not None:
            shutil.rmtree(scratch, ignore_errors=True)


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(
        level=logging.WARNING, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    args = parse_args(argv)
    try:
        run(args)
    except ExportError as exc:
        sys.exit(f"error: {exc}")


if __name__ == "__main__":
    main()
