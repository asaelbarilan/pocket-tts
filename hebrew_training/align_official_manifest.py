"""Align official Pocket TTS manifests with Viter (default) or MMS CTC.

The public entry point re-executes itself in the selected dedicated environment. Persistent
workers load models once, while manifests and output files remain streaming.

Example:
    uv run python hebrew_training/align_official_manifest.py /data/plenum

Reads ``train_diac.jsonl`` (the diacritize pass's output) when the folder has one, else
``train.jsonl`` (the manifest builder's), likewise ``valid``, and writes ``train_aligned.jsonl``
and ``valid_aligned.jsonl``. Source manifests are never modified.

The row goes out as it came in with new word times: `words` keeps its text and its length and
order, `transcript` is rebuilt from it, and every other key -- `nikud`, `alt_words`, ... -- is
carried over untouched, so this pass and the diacritize pass can run in either order. A row
that would come out with a different number of words is kept as it was instead.

This file is run from its own virtualenv by path and imports nothing from `hebrew_training`:
the stage file names below are a copy of hebrew_training/manifest_files.py, and a test keeps
the two equal.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import math
import multiprocessing as mp
import os
import queue
import re
import select
import signal
import subprocess
import sys
import tempfile
import time
import unicodedata
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

BUILT = "{split}.jsonl"
DIACRITIZED = "{split}_diac.jsonl"
ALIGNED = "{split}_aligned.jsonl"
# A row with either of these has been through this stage, and aligning it again would
# overwrite `original_words` with the already-aligned words.
ALREADY_ALIGNED_KEYS = ("alignment_backend", "original_words")
VERIFY_ROWS = 20

SAMPLE_RATE = 16_000
PRE_PADDING = 0.5
POST_PADDING = 1.0
DEFAULT_ENV = Path("/home/yoadsnapir/time-align/mms-ctc/ctc-env")
DEFAULT_MODEL = "MahmoudAshraf/mms-300m-1130-forced-aligner"
DEFAULT_VITER_ENV = Path("/home/yoadsnapir/time-align/viter-env")
_WS = re.compile(r"\s+")
_PUNCT = re.compile(r"[^\w֐-׿']+", re.UNICODE)
_FINAL_LETTERS = str.maketrans("םןץףך", "מנצפכ")

_PROCESSOR: Any = None
_MODEL: Any = None
_ROMANIZER: Any = None
_VOCAB: dict[str, int] = {}
_BLANK = 0
_DELIMITER: int | None = None
_DEVICE = ""
_VITER: Any = None
_VITER_WORDS: set[str] = set()


def clean(text: str) -> str:
    return _WS.sub(" ", unicodedata.normalize("NFC", text)).strip()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "folder",
        type=Path,
        help="Directory holding train(_diac).jsonl and valid(_diac).jsonl; the aligned files "
        "are written beside them",
    )
    parser.add_argument(
        "--input-stage",
        choices=("auto", "built", "diac"),
        default="auto",
        help="Which file to read: auto takes {split}_diac.jsonl when it exists, else "
        "{split}.jsonl; built or diac insists on one.",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        choices=("train", "valid"),
        default=("valid", "train"),
        help="Splits to process, in order (default: valid train)",
    )
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument(
        "--workers-per-gpu",
        type=int,
        default=1,
        help="Persistent workers per selected GPU (default: 1); overlaps CPU/decode stages",
    )
    parser.add_argument(
        "--devices",
        default="all",
        help="Comma-separated CUDA indices, 'all', or 'cpu' (default: all)",
    )
    parser.add_argument(
        "--decode-workers", type=int, default=4, help="ffmpeg threads per GPU worker"
    )
    parser.add_argument("--dtype", choices=("float16", "float32"), default="float16")
    parser.add_argument("--aligner", choices=("viter", "mms"), default="viter")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="MMS model, including fallback")
    parser.add_argument(
        "--environment", type=Path, help="Override the selected aligner's Python environment"
    )
    parser.add_argument(
        "--mms-environment", type=Path, default=Path(os.environ.get("MMS_CTC_ENV", DEFAULT_ENV))
    )
    parser.add_argument(
        "--viter-environment",
        type=Path,
        default=Path(os.environ.get("VITER_ENV", DEFAULT_VITER_ENV)),
    )
    parser.add_argument("--viter-model", type=Path, help="Hebrew grapheme .viter model")
    parser.add_argument(
        "--viter-dictionary", type=Path, help="Matching grapheme dictionary (read-only)"
    )
    parser.add_argument("--beam", type=float, default=100)
    parser.add_argument("--retry-beam", type=float, default=800)
    parser.add_argument(
        "--viter-threads", type=int, default=4, help="Compute threads per Viter worker"
    )
    parser.add_argument("--mms-worker", help=argparse.SUPPRESS)
    parser.add_argument(
        "--limit",
        type=int,
        help="Process at most this many source rows per split (for benchmarking)",
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("--batch-size must be at least 1")
    if args.decode_workers < 1:
        parser.error("--decode-workers must be at least 1")
    if args.workers_per_gpu < 1:
        parser.error("--workers-per-gpu must be at least 1")
    if args.devices == "cpu" and args.workers_per_gpu != 1:
        parser.error("--workers-per-gpu requires GPU devices")
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be at least 1")
    if args.viter_threads < 1 or args.beam <= 0 or args.retry_beam < args.beam:
        parser.error("Viter threads/beam must be positive and retry-beam must be >= beam")
    if args.environment:
        if args.aligner == "mms":
            args.mms_environment = args.environment
        else:
            args.viter_environment = args.environment
    args.environment = args.mms_environment if args.aligner == "mms" else args.viter_environment
    args.viter_model = args.viter_model or args.viter_environment / "models/hebrew_graphemes.viter"
    args.viter_dictionary = (
        args.viter_dictionary or args.viter_environment / "models/hebrew_graphemes.dict"
    )
    return args


def check_source_head(path: Path) -> None:
    """Fail up front, naming the file, if the first rows are not what this pass reads."""
    seen = 0
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            seen += 1
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise SystemExit(f"{path} is not a JSON-lines manifest: {error}") from error
            problem = source_row_problem(row)
            if problem:
                raise SystemExit(
                    f"{path} is not an input for alignment: row {seen} {problem}. Expected the "
                    "output of build_official_manifest or diacritize_official_manifest."
                )
            if seen == VERIFY_ROWS:
                return
    if not seen:
        raise SystemExit(f"{path} is empty")


def source_row_problem(row: Any) -> str | None:
    if not isinstance(row, dict):
        return "is not a JSON object"
    if not isinstance(row.get("path"), str):
        return "has no string `path`"
    for key in ("start", "duration"):
        if isinstance(row.get(key), bool) or not isinstance(row.get(key), int | float):
            return f"has no numeric `{key}`"
    words = row.get("words")
    if words:
        if not isinstance(words, list) or not all(
            isinstance(item, dict) and isinstance(item.get("word"), str) for item in words
        ):
            return "has a `words` entry without a string `word`"
    elif not isinstance(row.get("transcript"), str):
        return "has neither `words` nor a `transcript`"
    for key in ALREADY_ALIGNED_KEYS:
        if key in row:
            return f"already has `{key}`, so it has been aligned"
    nikud = row.get("nikud")
    if nikud is not None and (not isinstance(nikud, list) or len(nikud) != len(words or [])):
        return "has a `nikud` list that is not one entry per word"
    return None


def source_path(folder: Path, split: str, stage: str = "auto") -> Path:
    """The manifest to align for `split`, checked: see `check_source_head`."""
    built = folder / BUILT.format(split=split)
    diacritized = folder / DIACRITIZED.format(split=split)
    if stage == "built":
        path = built
    elif stage == "diac" or diacritized.is_file():
        path = diacritized
    else:
        path = built
    if not path.is_file():
        raise FileNotFoundError(path)
    check_source_head(path)
    return path


def dedicated_python(environment: Path) -> Path:
    python = environment.expanduser().resolve() / "bin" / "python"
    if not python.is_file():
        raise FileNotFoundError(f"Dedicated aligner Python was not found at {python}")
    return python


def reexec_in_dedicated_environment(environment: Path) -> None:
    python = dedicated_python(environment)
    # Different virtualenvs may symlink the same base interpreter.
    if Path(sys.prefix).resolve() != environment.expanduser().resolve():
        os.execv(str(python), [str(python), str(Path(__file__).resolve()), *sys.argv[1:]])


def blank_id(tokenizer: Any) -> int:
    vocab = tokenizer.get_vocab()
    for name in ("<blank>", "[PAD]", "<pad>"):
        if name in vocab:
            return vocab[name]
    return tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0


def decode_audio(row: dict[str, Any]) -> tuple[Any, float]:
    """Decode only the padded source span to mono float32 at 16 kHz."""
    import numpy as np

    source = Path(row["path"])
    if not source.is_file():
        raise FileNotFoundError(source)
    start = float(row["start"])
    duration = float(row["duration"])
    if start < 0 or duration <= 0:
        raise ValueError(f"invalid span start={start}, duration={duration}")
    origin = max(0.0, start - PRE_PADDING)
    padded_duration = start + duration + POST_PADDING - origin
    command = [
        "ffmpeg",
        "-nostdin",
        "-loglevel",
        "error",
        "-ss",
        f"{origin:.6f}",
        "-i",
        str(source),
        "-t",
        f"{padded_duration:.6f}",
        "-map",
        "0:a:0",
        "-vn",
        "-ac",
        "1",
        "-ar",
        str(SAMPLE_RATE),
        "-f",
        "f32le",
        "-acodec",
        "pcm_f32le",
        "pipe:1",
    ]
    completed = subprocess.run(command, capture_output=True, check=False)
    if completed.returncode:
        detail = completed.stderr.decode("utf-8", errors="replace").strip()
        raise RuntimeError(f"ffmpeg exited {completed.returncode}: {detail[-500:]}")
    audio = np.frombuffer(completed.stdout, dtype="<f4").copy()
    if not len(audio):
        raise ValueError("ffmpeg decoded no audio")
    return audio, origin


def word_targets(row: dict[str, Any]) -> tuple[list[str], list[list[int]], list[int]]:
    words = source_words(row)
    token_ids = []
    for word in words:
        text = _ROMANIZER.romanize_string(word).lower()
        token_ids.append([_VOCAB[character] for character in text if character in _VOCAB])
    if any(not ids for ids in token_ids):
        raise ValueError("MMS cannot spell every source word; refusing to drop words")
    keep = [index for index, ids in enumerate(token_ids) if ids]
    return words, token_ids, keep


def source_words(row: dict[str, Any]) -> list[str]:
    return (
        [clean(str(item["word"])) for item in row["words"]]
        if row.get("words")
        else clean(str(row["transcript"])).split()
    )


def align_emission(row: dict[str, Any], audio: Any, origin: float, emission: Any) -> dict[str, Any]:
    import torch
    import torchaudio.functional as audio_functional

    words, token_ids, keep = word_targets(row)
    if not keep:
        raise ValueError("no word the model can spell")
    flat = []
    for position, index in enumerate(keep):
        if position and _DELIMITER is not None:
            flat.append(_DELIMITER)
        flat.extend(token_ids[index])
    aligned, scores = audio_functional.forced_align(
        emission[None], torch.tensor([flat], dtype=torch.int32), blank=_BLANK
    )
    spans = audio_functional.merge_tokens(aligned[0], scores[0].exp(), blank=_BLANK)
    if len(spans) != len(flat):
        raise ValueError(f"{len(spans)} spans for {len(flat)} targets")

    ratio = len(audio) / emission.shape[0] / SAMPLE_RATE
    timed = []
    cursor = 0
    for position, index in enumerate(keep):
        if position and _DELIMITER is not None:
            cursor += 1
        chunk = spans[cursor : cursor + len(token_ids[index])]
        cursor += len(token_ids[index])
        if chunk:
            timed.append(
                {
                    "word": words[index],
                    "start": chunk[0].start * ratio,
                    "end": chunk[-1].end * ratio,
                }
            )
    if not timed:
        raise ValueError("alignment returned no timed words")

    return update_timings(row, origin, timed, "mms")


def original_snapshot(row: dict[str, Any]) -> dict[str, Any]:
    updated = dict(row)
    updated["original_start"] = row["start"]
    updated["original_duration"] = row["duration"]
    updated["original_words"] = row.get("words")
    return updated


def update_timings(
    row: dict[str, Any], origin: float, timed: list[dict[str, Any]], aligner: str
) -> dict[str, Any]:
    if not timed or any(not math.isfinite(word[key]) for word in timed for key in ("start", "end")):
        raise ValueError("alignment returned empty/non-finite timings")
    previous_end = 0.0
    for word in timed:
        if word["start"] < previous_end - 1e-4 or word["end"] < word["start"]:
            raise ValueError("alignment returned invalid or overlapping word spans")
        previous_end = word["end"]
    first, last = timed[0]["start"], timed[-1]["end"]
    if last <= first:
        raise ValueError("alignment returned zero duration")
    # `nikud` and `alt_words` are indexed by position in `words`, and are carried over as-is.
    if row.get("words") and len(timed) != len(row["words"]):
        raise ValueError(f"alignment returned {len(timed)} words for {len(row['words'])}")
    updated = original_snapshot(row)
    updated["alignment_backend"] = aligner
    updated["start"] = round(origin + first, 4)
    updated["duration"] = round(last - first, 4)
    updated["words"] = [
        {
            "word": word["word"],
            "start": round(word["start"] - first, 4),
            "end": round(word["end"] - first, 4),
        }
        for word in timed
    ]
    # The official loader trains from words, so keep its fallback transcript in sync.
    updated["transcript"] = " ".join(word["word"] for word in updated["words"])
    return updated


def initialize_worker(device: str, model_name: str, dtype_name: str) -> None:
    global _BLANK, _DELIMITER, _DEVICE, _MODEL, _PROCESSOR, _ROMANIZER, _VOCAB

    import torch
    import uroman
    from transformers import AutoProcessor, Wav2Vec2ForCTC

    _DEVICE = device
    torch.set_num_threads(1)
    if device != "cpu":
        torch.cuda.set_device(device)
    _PROCESSOR = AutoProcessor.from_pretrained(model_name)
    dtype = torch.float16 if dtype_name == "float16" and device != "cpu" else torch.float32
    _MODEL = Wav2Vec2ForCTC.from_pretrained(model_name, torch_dtype=dtype).to(device).eval()
    _VOCAB = {
        key: value for key, value in _PROCESSOR.tokenizer.get_vocab().items() if len(key) == 1
    }
    _BLANK = blank_id(_PROCESSOR.tokenizer)
    token = _PROCESSOR.tokenizer.word_delimiter_token
    _DELIMITER = _VOCAB.get(token) if token else None
    _ROMANIZER = uroman.Uroman()


def align_batch(
    rows: list[tuple[int, dict[str, Any]]], decode_workers: int
) -> list[dict[str, Any]]:
    import torch

    outcomes: list[dict[str, Any] | None] = [None] * len(rows)

    def decode(item: tuple[int, dict[str, Any]]) -> tuple[Any, float] | Exception:
        try:
            return decode_audio(item[1])
        except Exception as exc:  # noqa: BLE001 - one bad source must not stop a multi-day run
            return exc

    with ThreadPoolExecutor(max_workers=min(decode_workers, len(rows))) as executor:
        decoded = list(executor.map(decode, rows))

    valid_positions = [
        index for index, value in enumerate(decoded) if not isinstance(value, Exception)
    ]
    for index, value in enumerate(decoded):
        if isinstance(value, Exception):
            outcomes[index] = {"error": f"{type(value).__name__}: {value}"[:1000]}
    if valid_positions:
        arrays = [decoded[index][0] for index in valid_positions]
        lengths = torch.tensor([len(array) for array in arrays])
        input_values = torch.nn.utils.rnn.pad_sequence(
            [torch.from_numpy(array) for array in arrays], batch_first=True
        ).to(device=_DEVICE, dtype=next(_MODEL.parameters()).dtype)
        attention_mask = torch.arange(input_values.shape[1])[None] < lengths[:, None]
        with torch.inference_mode():
            logits = _MODEL(input_values, attention_mask=attention_mask.to(_DEVICE)).logits.float()
        output_lengths = _MODEL._get_feat_extract_output_lengths(lengths).tolist()
        for batch_position, row_position in enumerate(valid_positions):
            _, row = rows[row_position]
            audio, origin = decoded[row_position]
            try:
                emission = torch.log_softmax(
                    logits[batch_position, : output_lengths[batch_position]], dim=-1
                ).cpu()
                outcomes[row_position] = {"row": align_emission(row, audio, origin, emission)}
            except Exception as exc:  # noqa: BLE001 - record deterministic alignment failures
                outcomes[row_position] = {"error": f"{type(exc).__name__}: {exc}"[:1000]}
    return [outcome for outcome in outcomes if outcome is not None]


def clean_viter_word(word: str) -> str:
    return _PUNCT.sub("", word).strip()


def split_graphemes(word: str) -> list[str] | None:
    """Match the released Hebrew dictionary's final-letter and geresh conventions."""
    phones = []
    index = 0
    while index < len(word):
        letter = word[index]
        if letter == "'":
            index += 1
            continue
        if not "א" <= letter <= "ת":
            return None
        letter = letter.translate(_FINAL_LETTERS)
        if index + 1 < len(word) and word[index + 1] == "'" and letter in "גזצת":
            letter += "'"
            index += 1
        phones.append(letter)
        index += 1
    return phones or None


def prepare_viter_dictionary(args: argparse.Namespace, destination: Path) -> None:
    """Extend a scratch dictionary, leaving the downloaded dictionary untouched."""
    entries = {}
    with args.viter_dictionary.open(encoding="utf-8") as handle:
        for line in handle:
            fields = line.strip().split("\t")
            if len(fields) > 1:
                entries[fields[0]] = " ".join(fields[1:])
    allowed = set("אבגדהוזחטיכלמנסעפצקרשת") | {"ג'", "ז'", "צ'", "ת'", "sil", "spn"}
    if not entries or any(
        phone not in allowed for phones in entries.values() for phone in phones.split()
    ):
        raise ValueError("--viter-dictionary must be a Hebrew grapheme dictionary")
    added = 0
    for split in args.splits:
        source = source_path(args.folder, split, args.input_stage)
        for batch in iter_batches(source, 0, 1024, None):
            for _, row in batch:
                for original in source_words(row):
                    word = clean_viter_word(original)
                    if word not in entries and (phones := split_graphemes(word)):
                        entries[word] = " ".join(phones)
                        added += 1
    destination.write_text(
        "".join(f"{word}\t{phones}\n" for word, phones in sorted(entries.items())), encoding="utf-8"
    )
    print(f"Viter dictionary: {len(entries)} entries ({added} added in scratch copy)", flush=True)


def configure_viter_gpu(device: str) -> None:
    """Viter uses Vulkan, so CUDA_VISIBLE_DEVICES alone does not select its adapter."""
    if device == "cpu":
        return
    if not Path("/usr/share/vulkan/implicit_layer.d/VkLayer_MESA_device_select.json").is_file():
        raise RuntimeError("Viter GPU selection needs the Mesa Vulkan device-select layer")
    logical_index = int(device.split(":")[1])
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    physical_id = visible.split(",")[logical_index].strip() if visible else str(logical_index)
    bus = (
        subprocess.check_output(
            ["nvidia-smi", "-i", physical_id, "--query-gpu=pci.bus_id", "--format=csv,noheader"],
            text=True,
        )
        .strip()
        .lower()
    )
    domain, bus_number, slot = bus.split(":")
    os.environ["DRI_PRIME"] = f"pci-{int(domain, 16):04x}_{bus_number}_{slot.replace('.', '_')}!"
    os.environ["MESA_VK_DEVICE_SELECT_FORCE_DEFAULT_DEVICE"] = "1"


class MMSFallback:
    """Persistent JSON-lines subprocess keeps MMS dependencies in their own environment."""

    def __init__(self, args: argparse.Namespace, device: str):
        self.args = args
        self.device = device
        self.process: Any = None

    def align(self, rows: list[Any]) -> list[dict[str, Any]]:
        try:
            return self._align(rows)
        except Exception:
            self.close()
            raise

    def _align(self, rows: list[Any]) -> list[dict[str, Any]]:
        if self.process is None:
            self.process = subprocess.Popen(
                [
                    str(dedicated_python(self.args.mms_environment)),
                    str(Path(__file__).resolve()),
                    str(self.args.folder),
                    "--mms-worker",
                    self.device,
                    "--aligner",
                    "mms",
                    "--model",
                    self.args.model,
                    "--dtype",
                    self.args.dtype,
                    "--decode-workers",
                    str(self.args.decode_workers),
                ],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                text=True,
                bufsize=1,
            )
        self.process.stdin.write(json.dumps(rows, ensure_ascii=False) + "\n")
        self.process.stdin.flush()
        if not select.select([self.process.stdout], [], [], 600)[0]:
            self.close()
            raise RuntimeError("MMS fallback timed out after 600 seconds")
        reply = self.process.stdout.readline()
        if not reply:
            self.close()
            raise RuntimeError("MMS fallback worker exited without a result")
        outcomes = json.loads(reply)
        if len(outcomes) != len(rows):
            raise RuntimeError("MMS fallback returned an incorrect number of rows")
        return outcomes

    def close(self) -> None:
        if self.process is not None:
            with contextlib.suppress(OSError):
                self.process.stdin.close()
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait()
            self.process.stdout.close()
            self.process = None


def align_viter_batch(
    rows: list[Any], args: argparse.Namespace, fallback: MMSFallback
) -> list[dict[str, Any]]:
    outcomes: list[Any] = [None] * len(rows)
    candidates = []
    reasons = {}
    for index, (_, row) in enumerate(rows):
        words = source_words(row)
        cleaned = [clean_viter_word(word) for word in words]
        if any(
            character.isdigit() for character in str(row.get("transcript", "")) + " ".join(words)
        ):
            reasons[index] = "digits are unsupported by Viter"
        elif not cleaned or any(word not in _VITER_WORDS for word in cleaned):
            reasons[index] = "source word is unsupported by the Viter dictionary"
        else:
            candidates.append((index, words, cleaned))

    def decode(candidate: Any) -> Any:
        try:
            return decode_audio(rows[candidate[0]][1])
        except Exception as exc:  # noqa: BLE001
            return exc

    if candidates:
        with ThreadPoolExecutor(max_workers=args.decode_workers) as executor:
            decoded = list(executor.map(decode, candidates))
        valid = []
        for candidate, value in zip(candidates, decoded, strict=True):
            if isinstance(value, Exception):
                reasons[candidate[0]] = f"audio decode failed: {value}"
            else:
                valid.append((candidate, value))
        if valid:
            try:
                alignments = _VITER.align_many(
                    [
                        (audio, " ".join(candidate[2]), SAMPLE_RATE)
                        for candidate, (audio, _) in valid
                    ],
                    speakers=[
                        str(rows[candidate[0]][1].get("speaker") or rows[candidate[0]][1]["path"])
                        for candidate, _ in valid
                    ],
                    beam=args.beam,
                    retry_beam=args.retry_beam,
                )
            except Exception as exc:  # noqa: BLE001
                alignments = [None] * len(valid)
                for candidate, _ in valid:
                    reasons[candidate[0]] = f"Viter batch failed: {exc}"
            for (candidate, (audio, origin)), alignment in zip(valid, alignments, strict=True):
                index, words, cleaned = candidate
                try:
                    if alignment is None:
                        raise ValueError("Viter returned no alignment")
                    spans = alignment.words
                    if [span.label for span in spans] != cleaned:
                        raise ValueError("Viter words do not match source words")
                    timed = [
                        {"word": word, "start": span.start, "end": span.end}
                        for word, span in zip(words, spans, strict=True)
                    ]
                    if timed[-1]["end"] > len(audio) / SAMPLE_RATE + 0.02:
                        raise ValueError("Viter timings extend beyond decoded audio")
                    outcomes[index] = {
                        "row": update_timings(rows[index][1], origin, timed, "viter")
                    }
                except Exception as exc:  # noqa: BLE001
                    reasons.setdefault(index, f"Viter failed: {exc}")

    fallback_indices = [index for index, outcome in enumerate(outcomes) if outcome is None]
    if fallback_indices:
        try:
            replies = fallback.align([rows[index] for index in fallback_indices])
        except Exception as exc:  # noqa: BLE001 - retain the source if fallback is unavailable
            replies = [{"error": f"MMS fallback failed: {exc}"} for _ in fallback_indices]
        for index, reply in zip(fallback_indices, replies, strict=True):
            if "row" in reply:
                reply["row"]["alignment_fallback_reason"] = reasons[index]
                outcomes[index] = reply
            else:
                updated = original_snapshot(rows[index][1])
                updated["alignment_backend"] = "original"
                updated["alignment_fallback_reason"] = reasons[index]
                updated["alignment_error"] = reply["error"]
                outcomes[index] = {"row": updated, "error": reply["error"]}
    return outcomes


def worker_main(
    device: str, worker_id: str, args: argparse.Namespace, tasks: Any, results: Any
) -> None:
    global _VITER, _VITER_WORDS
    fallback = MMSFallback(args, device)
    # Let finally close the MMS child when the coordinator terminates this worker.
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(1))
    try:
        if args.aligner == "mms":
            initialize_worker(device, args.model, args.dtype)
        else:
            configure_viter_gpu(device)
            os.environ["RAYON_NUM_THREADS"] = str(args.viter_threads)
            import viter

            _VITER = viter.Model(
                str(args.viter_model), dict=str(args.merged_dictionary), cpu=device == "cpu"
            )
            allowed = set("אבגדהוזחטיכלמנסעפצקרשת") | {"ג'", "ז'", "צ'", "ת'", "sil", "spn"}
            if not set(_VITER.phones) <= allowed or not set(_VITER.phones) & set(
                "אבגדהוזחטיכלמנסעפצקרשת"
            ):
                raise ValueError("--viter-model must be a Hebrew grapheme model")
            with args.merged_dictionary.open(encoding="utf-8") as handle:
                _VITER_WORDS = {line.split("\t", 1)[0] for line in handle}
        results.put(("ready", worker_id))
        while True:
            task = tasks.get()
            if task is None:
                return
            batch_id, rows = task
            started = time.monotonic()
            try:
                outcomes = (
                    align_batch(rows, args.decode_workers)
                    if args.aligner == "mms"
                    else align_viter_batch(rows, args, fallback)
                )
                results.put(("batch", batch_id, rows[-1][0], outcomes, time.monotonic() - started))
            except Exception as exc:  # noqa: BLE001 - surface batch failures to the coordinator
                results.put(("fatal", worker_id, f"{type(exc).__name__}: {exc}"))
                return
    except Exception as exc:  # noqa: BLE001 - imports/model loading happen in child processes
        results.put(("fatal", worker_id, f"{type(exc).__name__}: {exc}"))
    finally:
        fallback.close()


def resolve_devices(spec: str, environment: Path) -> list[str]:
    if spec == "cpu":
        return ["cpu"]
    # Query CUDA in the MMS environment without importing its dependencies into Viter.
    count = int(
        subprocess.check_output(
            [
                str(dedicated_python(environment)),
                "-c",
                "import torch; print(torch.cuda.device_count() if torch.cuda.is_available() else 0)",
            ],
            text=True,
        ).strip()
    )
    if not count:
        raise RuntimeError("CUDA is unavailable in the dedicated MMS environment")
    indices = list(range(count)) if spec == "all" else [int(x) for x in spec.split(",")]
    if not indices:
        raise ValueError("--devices selected no GPUs")
    invalid = [index for index in indices if index < 0 or index >= count]
    if invalid:
        raise ValueError(f"invalid CUDA device indices: {invalid}")
    if len(set(indices)) != len(indices):
        raise ValueError("--devices contains duplicate GPUs")
    return [f"cuda:{index}" for index in indices]


def progress_paths(output: Path) -> tuple[Path, Path]:
    return output.with_suffix(".progress.json"), output.with_suffix(".failed.jsonl")


def save_progress(path: Path, state: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(state) + "\n", encoding="utf-8")
    temporary.replace(path)


def open_resumable_files(
    output: Path, overwrite: bool, configuration: dict[str, Any]
) -> tuple[Any, Any, int]:
    progress, failed = progress_paths(output)
    if overwrite:
        for path in (output, progress, failed):
            path.unlink(missing_ok=True)
    if output.exists() and not progress.exists():
        raise FileExistsError(
            f"Completed output already exists: {output}; use --overwrite to replace it"
        )

    output.parent.mkdir(parents=True, exist_ok=True)
    state: dict[str, Any] = {
        "input_lines": 0,
        "output_bytes": 0,
        "failure_bytes": 0,
        "configuration": configuration,
    }
    if progress.exists():
        saved = json.loads(progress.read_text(encoding="utf-8"))
        if saved.get("configuration") != configuration:
            raise ValueError(
                "Resume configuration differs (or predates backend selection); "
                "use --overwrite to restart with the requested aligner"
            )
        state.update(saved)
    output_handle = output.open("r+b" if output.exists() else "w+b")
    failed_handle = failed.open("r+b" if failed.exists() else "w+b")
    output_handle.truncate(state["output_bytes"])
    failed_handle.truncate(state["failure_bytes"])
    output_handle.seek(0, os.SEEK_END)
    failed_handle.seek(0, os.SEEK_END)
    if not progress.exists():
        save_progress(progress, state)
    return output_handle, failed_handle, state["input_lines"]


def iter_batches(manifest: Path, start_line: int, batch_size: int, limit: int | None) -> Any:
    selected = []
    processed = 0
    with manifest.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if line_number <= start_line or not line.strip():
                continue
            selected.append((line_number, json.loads(line)))
            processed += 1
            if len(selected) == batch_size:
                yield selected
                selected = []
            if limit is not None and processed >= limit:
                break
    if selected:
        yield selected


def wait_for_workers(processes: list[Any], results: Any) -> None:
    ready = set()
    while len(ready) < len(processes):
        try:
            message = results.get(timeout=5)
        except queue.Empty:
            dead = [process.name for process in processes if not process.is_alive()]
            if dead:
                raise RuntimeError(f"GPU workers exited during startup: {', '.join(dead)}")
            continue
        if message[0] == "fatal":
            raise RuntimeError(f"worker {message[1]} failed during startup: {message[2]}")
        ready.add(message[1])


def write_batch(output: Any, failed: Any, outcomes: list[dict[str, Any]], rows: list[Any]) -> int:
    failures = 0
    for outcome, (line_number, source_row) in zip(outcomes, rows, strict=True):
        if "row" in outcome:
            payload = json.dumps(outcome["row"], ensure_ascii=False) + "\n"
            output.write(payload.encode("utf-8"))
        if "error" in outcome:
            payload = {
                "input_line": line_number,
                "path": source_row.get("path"),
                "start": source_row.get("start"),
                "error": outcome["error"],
            }
            failed.write((json.dumps(payload, ensure_ascii=False) + "\n").encode("utf-8"))
            failures += 1
    return failures


def align_split(args: argparse.Namespace, split: str, devices: list[str]) -> None:
    source = source_path(args.folder, split, args.input_stage)
    output = args.folder / ALIGNED.format(split=split)
    print(f"{split}: aligning {source.name} into {output.name}", flush=True)
    progress, _ = progress_paths(output)
    configuration = {
        "aligner": args.aligner,
        "mms_model": args.model,
        "dtype": args.dtype,
        "viter_model": str(args.viter_model),
        "viter_dictionary": str(args.viter_dictionary),
        "beam": args.beam,
        "retry_beam": args.retry_beam,
    }
    try:
        output_handle, failed_handle, start_line = open_resumable_files(
            output, args.overwrite, configuration
        )
    except FileExistsError as exc:
        print(f"{split}: {exc}", flush=True)
        return

    context = mp.get_context("spawn")
    worker_count = len(devices) * args.workers_per_gpu
    tasks = context.Queue(maxsize=worker_count * 2)
    results = context.Queue(maxsize=worker_count * 2)
    processes = [
        context.Process(
            target=worker_main,
            args=(device, f"{args.aligner}-{device}-{worker_index}", args, tasks, results),
            name=f"{args.aligner}-{device}-{worker_index}",
        )
        for device in devices
        for worker_index in range(args.workers_per_gpu)
    ]
    for process in processes:
        process.start()

    started = time.monotonic()
    submitted: dict[int, list[Any]] = {}
    completed: dict[int, tuple[int, list[dict[str, Any]], float]] = {}
    next_submit = 0
    next_write = 0
    exhausted = False
    aligned = failures = source_rows = retained = 0
    batches = iter(iter_batches(source, start_line, args.batch_size, args.limit))
    checkpoint_batches = 0
    try:
        wait_for_workers(processes, results)
        print(
            f"{split}: {worker_count} {args.aligner} workers ready on {len(devices)} devices; "
            f"resuming after source line {start_line}",
            flush=True,
        )
        while not exhausted or submitted:
            while not exhausted and len(submitted) < worker_count * 2:
                try:
                    rows = next(batches)
                except StopIteration:
                    exhausted = True
                    break
                submitted[next_submit] = rows
                tasks.put((next_submit, rows))
                next_submit += 1

            if not submitted:
                break
            try:
                message = results.get(timeout=30)
            except queue.Empty:
                dead = [process.name for process in processes if not process.is_alive()]
                if dead:
                    raise RuntimeError(f"GPU workers exited unexpectedly: {', '.join(dead)}")
                continue
            if message[0] == "fatal":
                raise RuntimeError(f"worker {message[1]} failed: {message[2]}")
            _, batch_id, last_line, outcomes, batch_seconds = message
            completed[batch_id] = (last_line, outcomes, batch_seconds)
            while next_write in completed:
                last_line, outcomes, batch_seconds = completed.pop(next_write)
                rows = submitted.pop(next_write)
                failures += write_batch(output_handle, failed_handle, outcomes, rows)
                retained += sum(
                    outcome.get("row", {}).get("alignment_backend") == "original"
                    for outcome in outcomes
                )
                aligned += sum("row" in outcome and "error" not in outcome for outcome in outcomes)
                source_rows += len(rows)
                next_write += 1
                checkpoint_batches += 1
                if checkpoint_batches >= 32:
                    output_handle.flush()
                    failed_handle.flush()
                    os.fsync(output_handle.fileno())
                    os.fsync(failed_handle.fileno())
                    save_progress(
                        progress,
                        {
                            "input_lines": last_line,
                            "output_bytes": output_handle.tell(),
                            "failure_bytes": failed_handle.tell(),
                            "configuration": configuration,
                        },
                    )
                    checkpoint_batches = 0
                if next_write % 25 == 0:
                    elapsed = time.monotonic() - started
                    print(
                        f"{split}: {source_rows} rows, {source_rows / elapsed:.2f} rows/s "
                        f"({batch_seconds:.2f}s last batch), {failures} failed",
                        flush=True,
                    )

        output_handle.flush()
        failed_handle.flush()
        os.fsync(output_handle.fileno())
        os.fsync(failed_handle.fileno())
        if source_rows:
            save_progress(
                progress,
                {
                    "input_lines": last_line,
                    "output_bytes": output_handle.tell(),
                    "failure_bytes": failed_handle.tell(),
                    "configuration": configuration,
                },
            )
        if args.limit is None:
            progress.unlink(missing_ok=True)
        elapsed = time.monotonic() - started
        print(
            f"{split}: aligned {aligned}, retained original {retained}, failed {failures}, elapsed {elapsed:.1f}s "
            f"({source_rows / elapsed:.2f} rows/s)",
            flush=True,
        )
    except BaseException:
        for process in processes:
            process.terminate()
        raise
    finally:
        output_handle.close()
        failed_handle.close()
        for _ in processes:
            try:
                tasks.put_nowait(None)
            except queue.Full:
                break
        for process in processes:
            process.join(timeout=10)
            if process.is_alive():
                process.terminate()


def main() -> None:
    args = parse_args()
    if args.mms_worker:
        # stdout is exclusively the protocol; dependency/model messages go to stderr.
        with contextlib.redirect_stdout(sys.stderr):
            initialize_worker(args.mms_worker, args.model, args.dtype)
        for line in sys.stdin:
            rows = json.loads(line)
            try:
                with contextlib.redirect_stdout(sys.stderr):
                    outcomes = align_batch(rows, args.decode_workers)
            except Exception as exc:  # noqa: BLE001
                outcomes = [{"error": f"MMS batch failed: {exc}"} for _ in rows]
            print(json.dumps(outcomes, ensure_ascii=False), flush=True)
        return
    reexec_in_dedicated_environment(args.environment)
    args.folder = args.folder.expanduser().resolve()
    devices = resolve_devices(args.devices, args.mms_environment)
    with tempfile.TemporaryDirectory(prefix="pocket-viter-") as scratch:
        if args.aligner == "viter":
            if not args.viter_model.is_file():
                raise FileNotFoundError(
                    f"Viter model missing: {args.viter_model}; see hebrew_training/ALIGNMENT.md"
                )
            args.merged_dictionary = Path(scratch) / "dictionary.tsv"
            prepare_viter_dictionary(args, args.merged_dictionary)
        for split in args.splits:
            align_split(args, split, devices)


if __name__ == "__main__":
    main()
