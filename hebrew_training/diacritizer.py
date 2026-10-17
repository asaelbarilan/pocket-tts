"""The diacritizer: DictaBERT for real runs, and a fake one for tests and machines without a GPU.

`load_diacritizer(spec)` returns `predict(lines) -> marked lines`, which is all the tokenizer
corpus builder needs. The manifest diacritize pass uses `open_diacritizer(args)`, which also
finds the GPUs: it detects how many CUDA devices there are and which kind, uses all of them
unless `--gpus 1,3,4` names some, and gives back one object that spreads batches over them.
With one device nothing is spawned; with several, one worker process per GPU loads the model on
its own device and takes batches from a shared queue.

The fake only exercises the mechanics. It puts one vowel mark on every Hebrew letter, which is
not how Hebrew is vocalized, so its output says nothing about real data. The diacritize pass
records in its summary that the fake produced a file.
"""

from __future__ import annotations

import argparse
import contextlib
import multiprocessing
import os
import queue
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass, replace

from hebrew_training.tokenizer_charset import HEBREW_LETTERS

DICTA = "dicta"
FAKE = "fake"
BACKENDS = (DICTA, FAKE)


@dataclass
class DiacritizerSpec:
    # A Hugging Face model id or a local directory. No default on purpose: loading one by id
    # downloads it, and that should be written down in the config, not implied.
    model: str | None = None
    device: str = "cuda"
    batch_size: int = 32
    revision: str | None = None
    dtype: str = "float32"
    preserve_matres: bool = True


class PythonOffsetsTokenizer:
    """Avoid thousands of scalar tensor operations in Dicta's Python reconstruction loop.

    The model's predict() only iterates/compares offsets; Python integers give identical
    results without the overhead of indexing zero-dimensional CPU tensors per character.
    """

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __getattr__(self, name):
        return getattr(self.tokenizer, name)

    def __call__(self, *args, **kwargs):
        encoded = self.tokenizer(*args, **kwargs)
        if "offset_mapping" in encoded:
            offsets = encoded["offset_mapping"].tolist()
            for row in offsets:
                seen = set()
                for span in row:
                    key = tuple(span)
                    # NFKC expands e.g. ellipsis into three tokens with the same original
                    # one-character span. Dicta otherwise emits the original char three times.
                    if span[1] - span[0] == 1:
                        if key in seen:
                            span[1] = span[0]
                        seen.add(key)
            encoded["offset_mapping"] = offsets
        return encoded


def load_dicta(spec: DiacritizerSpec) -> Callable[[list[str]], list[str]]:
    import torch
    from transformers import AutoModel, PreTrainedTokenizerFast

    # Not AutoTokenizer. This is a *character* model whose char-level pre-tokenizer lives only
    # in tokenizer.json; under transformers 5.x AutoTokenizer builds a word-level BertTokenizer
    # from vocab.txt instead, tokenizing every Hebrew word as [UNK]. predict() then returns
    # repeated prefixes with no nikud at all -- no error, just quiet rubbish.
    tokenizer = PythonOffsetsTokenizer(
        PreTrainedTokenizerFast.from_pretrained(spec.model, revision=spec.revision)
    )
    model = (
        AutoModel.from_pretrained(spec.model, revision=spec.revision, trust_remote_code=True)
        .eval()
        .to(device=spec.device, dtype=getattr(torch, spec.dtype))
    )
    kwargs = {"mark_matres_lectionis": ""} if spec.preserve_matres else {}

    def predict(lines: list[str]) -> list[str]:
        with torch.inference_mode():
            return model.predict(lines, tokenizer, **kwargs)

    return predict


# Marks in canonical (NFC) order for the one letter that takes two: a dagesh or shin dot is
# never combined with the vowel by the fake, so every output is already NFC-stable.
_FAKE_VOWELS = "\u05b8\u05b7\u05b4\u05b6\u05b9\u05bb\u05b0"


def fake_diacritizer(lines: list[str]) -> list[str]:
    """Deterministic and invertible: `strip_marks(fake(x)) == x`. Adds one mark per letter."""
    out = []
    for line in lines:
        marked = []
        for position, char in enumerate(line):
            marked.append(char)
            if char in HEBREW_LETTERS:
                marked.append(_FAKE_VOWELS[(ord(char) + position) % len(_FAKE_VOWELS)])
        out.append("".join(marked))
    return out


def load_diacritizer(
    spec: DiacritizerSpec, backend: str = DICTA
) -> Callable[[list[str]], list[str]]:
    """`backend` is not a field of the spec: the spec is hashed into the tokenizer corpus
    cache provenance, and adding a field would invalidate every existing cache."""
    if backend == FAKE:
        return fake_diacritizer
    if backend != DICTA:
        raise ValueError(f"unknown diacritizer backend {backend!r}; use one of {BACKENDS}")
    if not spec.model:
        raise SystemExit(
            "the dicta diacritizer needs --model (a Hugging Face id or a local directory); "
            "or pass --diacritizer fake for a local mechanics test"
        )
    return load_dicta(spec)


@dataclass(frozen=True)
class Gpu:
    index: int  # as torch numbers it, i.e. after CUDA_VISIBLE_DEVICES
    name: str
    memory_gib: float

    def __str__(self) -> str:
        return f"{self.index}: {self.name} ({self.memory_gib:.1f} GiB)"


def detect_gpus() -> list[Gpu]:
    """The CUDA devices torch can see, with their model names. Empty without torch or CUDA."""
    try:
        import torch
    except ImportError:
        return []
    if not torch.cuda.is_available():
        return []
    gpus = []
    for index in range(torch.cuda.device_count()):
        properties = torch.cuda.get_device_properties(index)
        gpus.append(Gpu(index, properties.name, properties.total_memory / 2**30))
    return gpus


def resolve_devices(gpus: str, backend: str) -> tuple[list[str], str]:
    """`(devices, what to tell the user)` for `--gpus`: `all` (default), `1,3,4`, `cpu`, `mps`."""
    choice = gpus.strip().lower()
    if choice in ("cpu", "mps"):
        return [choice], f"running on {choice}"
    available = detect_gpus()
    if not available:
        if backend == FAKE:
            return ["cpu"], "no CUDA GPU found; the fake diacritizer runs on cpu"
        raise SystemExit(
            "no CUDA GPU found. Pass --gpus cpu to run DictaBERT on the CPU (very slow), or "
            "--gpus mps on a Mac."
        )
    listing = "; ".join(str(g) for g in available)
    if choice == "all":
        chosen = list(available)
    else:
        try:
            wanted = [int(part) for part in choice.split(",")]
        except ValueError:
            raise SystemExit(
                f"--gpus must be 'all', 'cpu', 'mps' or indices like 1,3,4, not {gpus!r}"
            ) from None
        known = {g.index: g for g in available}
        missing = [i for i in wanted if i not in known]
        if missing or len(set(wanted)) != len(wanted):
            raise SystemExit(
                f"--gpus {gpus}: {'no such GPU ' + str(missing) if missing else 'duplicates'}. "
                f"Found {len(available)}: {listing}"
            )
        chosen = [known[i] for i in wanted]
    kinds = sorted({g.name for g in chosen})
    return (
        [f"cuda:{g.index}" for g in chosen],
        (
            f"found {len(available)} GPU(s): {listing}. Using {len(chosen)}: "
            f"{','.join(str(g.index) for g in chosen)} ({' / '.join(kinds)})"
        ),
    )


class Diacritizer:
    """`predict(lines)` plus `map_batches(batches)`, which is where several devices pay off."""

    devices: list[str]

    def __call__(self, lines: list[str]) -> list[str]:
        return self.map_batches([lines])[0]

    def map_batches(self, batches: Sequence[list[str]]) -> list[list[str]]:
        raise NotImplementedError

    def close(self) -> None:
        pass


class LocalDiacritizer(Diacritizer):
    def __init__(self, predict: Callable[[list[str]], list[str]], devices: list[str]):
        self.predict = predict
        self.devices = devices

    def map_batches(self, batches: Sequence[list[str]]) -> list[list[str]]:
        return [self.predict(batch) for batch in batches]


def _physical_gpu(logical: int) -> str:
    """The id a worker must put in CUDA_VISIBLE_DEVICES to see what the parent calls `logical`."""
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    return visible.split(",")[logical].strip() if visible else str(logical)


def _worker(spec, backend, device, worker_id, tasks, results) -> None:
    try:
        if device.startswith("cuda:"):
            # Before torch is imported in this process: the worker sees one GPU, as cuda:0,
            # so nothing in the model code can reach for a different device by accident.
            os.environ["CUDA_VISIBLE_DEVICES"] = _physical_gpu(int(device.split(":")[1]))
            device = "cuda:0"
        predict = load_diacritizer(replace(spec, device=device), backend)
        results.put(("ready", worker_id))
        while (task := tasks.get()) is not None:
            index, lines = task
            results.put(("done", index, predict(lines)))
    except (Exception, SystemExit) as error:  # noqa: BLE001 - the parent reports it
        results.put(("fatal", worker_id, f"{type(error).__name__}: {error}"))


class MultiDeviceDiacritizer(Diacritizer):
    """One worker process per device, all taking batches from one queue."""

    def __init__(self, spec: DiacritizerSpec, backend: str, devices: list[str]):
        context = multiprocessing.get_context("spawn")
        self.devices = devices
        self.tasks = context.Queue()
        self.results = context.Queue()
        self.processes = [
            context.Process(
                target=_worker,
                args=(spec, backend, device, f"{device}#{n}", self.tasks, self.results),
                name=f"diacritizer-{device}",
                daemon=True,
            )
            for n, device in enumerate(devices)
        ]
        for process in self.processes:
            process.start()
        ready = 0
        while ready < len(self.processes):
            message = self._next("starting")
            ready += message[0] == "ready"

    def _next(self, doing: str):
        while True:
            try:
                message = self.results.get(timeout=5)
            except queue.Empty:
                dead = [p.name for p in self.processes if not p.is_alive()]
                if dead:
                    self.close()
                    raise RuntimeError(f"diacritizer workers died while {doing}: {dead}") from None
                continue
            if message[0] == "fatal":
                self.close()
                raise RuntimeError(f"diacritizer worker {message[1]} failed: {message[2]}")
            return message

    def map_batches(self, batches: Sequence[list[str]]) -> list[list[str]]:
        for index, batch in enumerate(batches):
            self.tasks.put((index, batch))
        out: list[list[str] | None] = [None] * len(batches)
        for _ in batches:
            _, index, lines = self._next("vocalizing")
            out[index] = lines
        return out  # type: ignore[return-value]

    def close(self) -> None:
        for _ in self.processes:
            self.tasks.put(None)
        for process in self.processes:
            process.join(timeout=10)
            if process.is_alive():
                process.terminate()
        self.processes = []


@contextlib.contextmanager
def open_diacritizer(args: argparse.Namespace) -> Iterator[Diacritizer]:
    """The diacritizer the CLI flags ask for, on every GPU they allow, closed on exit."""
    spec = spec_from_args(args)
    if args.diacritizer == DICTA and not spec.model:
        raise SystemExit(
            "the dicta diacritizer needs --model (a Hugging Face id or a local directory); "
            "or pass --diacritizer fake for a local mechanics test"
        )
    devices, description = resolve_devices(args.gpus, args.diacritizer)
    print(f"diacritizer: {description}", flush=True)
    if len(devices) == 1:
        predict = load_diacritizer(replace(spec, device=devices[0]), args.diacritizer)
        yield LocalDiacritizer(predict, devices)
        return
    pool = MultiDeviceDiacritizer(spec, args.diacritizer, devices)
    try:
        yield pool
    finally:
        pool.close()


def add_diacritizer_args(parser: argparse.ArgumentParser) -> None:
    group = parser.add_argument_group("diacritizer")
    group.add_argument("--diacritizer", choices=BACKENDS, default=DICTA)
    group.add_argument(
        "--model",
        help="Dicta model: a Hugging Face id (downloaded on first use, and it runs the model "
        "repo's own code) or a local directory. See docs/TOKENIZER_EXPERIMENT.md for the "
        "reviewed revision.",
    )
    group.add_argument("--revision", help="Model revision; pin it.")
    group.add_argument(
        "--gpus",
        default="all",
        help="Which GPUs to run on: all (default), a list of indices like 1,3,4, or cpu / mps. "
        "Indices are torch's, so they follow CUDA_VISIBLE_DEVICES. Every listed GPU gets "
        "its own copy of the model.",
    )
    group.add_argument("--dtype", default="float32", choices=("float32", "float16", "bfloat16"))
    group.add_argument(
        "--batch-size", type=int, default=32, help="Lines per Dicta batch, on each GPU."
    )


def spec_from_args(args: argparse.Namespace) -> DiacritizerSpec:
    return DiacritizerSpec(
        model=args.model, batch_size=args.batch_size, revision=args.revision, dtype=args.dtype
    )
