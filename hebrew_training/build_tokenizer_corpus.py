"""Build the mixed text corpus a Hebrew tokenizer is trained on, then optionally train it.

    python -m hebrew_training.build_tokenizer_corpus \
        --config hebrew_training/configs/tokenizer_hebrew.yaml --train

`train_tokenizer.py` takes text. A tokenizer fitted on the training manifest alone cannot
spell anything the manifest lacks: measured on one corpus, vocalized Hebrew cost it 0.94
tokens per character and English 0.98, which is byte fallback. This assembles a mix instead
(docs/hebrew-english-data-mixing.md has the measurements behind the default shares).

Four kinds of text, each optional, in shares of **characters** (not lines or documents):

    manifest      the training manifest's own text -- what the model actually trains on.
                  Rebuild whenever the training corpus changes.
    hebrew        general modern Hebrew (FineWeb-2 `heb_Hebr` or any Hebrew text).
    hebrew_nikud  vocalized Hebrew. Either supplied text, or extra sampled manifest/general
                  Hebrew diacritized with a DictaBERT model (`diacritize: true`).
    english       general English (FineWeb or any English text).

A source with `pct: 0` is not used at all and needs nothing configured. The shares of the
rest must add to 100. Every source is a **local path** (file or folder; .txt, .jsonl,
.parquet, optionally .gz -- see tokenizer_text.py). Nothing is downloaded unless asked to:
`--download hebrew` / `--download english` (or `download: true` in the config) streams
FineWeb-2 / FineWeb into `cache_dir` instead of reading a path. A source that has neither a
path nor a download flag stops the run before any work is done.

Every line of every source passes the charset safety filter (tokenizer_charset.py) -- Latin,
Hebrew, niqqud, digits and punctuation only -- so a stray script in a crawl cannot reach the
tokenizer.

Needs: pyyaml, sentencepiece; `datasets` only for --download; `transformers` and `torch`
only for diacritize.
"""

from __future__ import annotations

import argparse
import json
import shlex
from collections.abc import Callable, Iterable
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path

import yaml

from hebrew_training.diacritizer import DiacritizerSpec, load_dicta
from hebrew_training.tokenizer_charset import (
    CANTILLATION,
    CATEGORIES,
    NIQQUD,
    CharsetFilter,
    add_charset_args,
    has_hebrew,
    has_latin,
    niqqud_rate,
)
from hebrew_training.tokenizer_sampling import SamplingSpec, digest, sample_source
from hebrew_training.tokenizer_text import find_files, iter_documents, lines_of

SOURCE_NAMES = ("manifest", "hebrew", "hebrew_nikud", "english")
# Where --download reads from. Only these two sources can be downloaded.
DOWNLOADS = {
    "hebrew": ("HuggingFaceFW/fineweb-2", "heb_Hebr"),
    "english": ("HuggingFaceFW/fineweb", "sample-10BT"),
}
_NIKUD_FLOOR = 0.05  # vocalized Hebrew is ~25% marks; below this the text is not vocalized


@dataclass
class SourceSpec:
    pct: float = 0.0
    path: str | None = None
    download: bool = False
    # hebrew_nikud only: generate the text by diacritizing the `hebrew` source.
    diacritize: bool = False
    diacritize_sources: list[str] = field(default_factory=lambda: ["hebrew"])


@dataclass
class CharsetSpec:
    allow: list[str] = field(default_factory=lambda: list(CATEGORIES))
    on_foreign: str = "drop-line"
    fold: bool = True


@dataclass
class TokenizerConfig:
    vocab_size: int = 8000
    required_chars: str = ""
    target_mb: float = 16.0
    output_dir: Path = Path("artifacts/tokenizer")
    cache_dir: Path = Path("artifacts/tokenizer/cache")
    vocalization_seed_dir: Path | None = None
    charset: CharsetSpec = field(default_factory=CharsetSpec)
    diacritizer: DiacritizerSpec = field(default_factory=DiacritizerSpec)
    sampling: SamplingSpec = field(default_factory=SamplingSpec)
    sources: dict[str, SourceSpec] = field(
        default_factory=lambda: {name: SourceSpec() for name in SOURCE_NAMES}
    )


def _build(cls, data: dict, where: str):
    known = {f.name for f in fields(cls)}
    unknown = set(data) - known
    if unknown:
        raise ValueError(f"{where}: unknown keys {sorted(unknown)}; expected {sorted(known)}")
    return cls(**data)


def load_config(path: Path | None) -> TokenizerConfig:
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) if path else {}
    raw = dict(raw or {})
    sources = raw.pop("sources", {}) or {}
    unknown = set(sources) - set(SOURCE_NAMES)
    if unknown:
        raise ValueError(f"sources: unknown {sorted(unknown)}; expected {list(SOURCE_NAMES)}")
    nested = {"charset": CharsetSpec, "diacritizer": DiacritizerSpec, "sampling": SamplingSpec}
    for key, cls in nested.items():
        raw[key] = _build(cls, raw.get(key) or {}, key)
    for key in ("output_dir", "cache_dir", "vocalization_seed_dir"):
        if key in raw and raw[key] is not None:
            raw[key] = Path(raw[key])
    config = _build(TokenizerConfig, raw, "config")
    config.sources = {
        name: _build(SourceSpec, sources.get(name) or {}, f"sources.{name}")
        for name in SOURCE_NAMES
    }
    return config


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--config", type=Path, help="Tokenizer config yaml (see configs/).")
    p.add_argument("--output-dir", type=Path, help="Overrides output_dir.")
    p.add_argument("--cache-dir", type=Path, help="Overrides cache_dir.")
    p.add_argument("--vocab-size", type=int, help="Overrides vocab_size.")
    p.add_argument("--target-mb", type=float, help="Overrides target_mb.")
    p.add_argument(
        "--pct",
        action="append",
        default=[],
        metavar="SOURCE=PCT",
        help=f"Override a source's share. SOURCE is one of {', '.join(SOURCE_NAMES)}. "
        "0 leaves it out. Repeatable.",
    )
    p.add_argument(
        "--path",
        action="append",
        default=[],
        metavar="SOURCE=PATH",
        help="Read a source from this local file or folder, replacing whatever the config says "
        "(a path, download or diacritize). Repeatable.",
    )
    p.add_argument(
        "--download",
        action="append",
        default=[],
        choices=sorted(DOWNLOADS),
        help="Opt in to streaming this source from Hugging Face instead of reading a path. "
        "Repeatable. Without it a source with no path is an error.",
    )
    p.add_argument("--device", help="Overrides diacritizer.device (cuda, cpu).")
    p.add_argument("--train", action="store_true", help="Train the tokenizer on the result.")
    add_charset_args(p, allow_disable=False)
    # Unset means "use the config"; the helper's own defaults would hide the config's values.
    p.set_defaults(allow=None, on_foreign=None)
    return p.parse_args(argv)


def apply_overrides(config: TokenizerConfig, args: argparse.Namespace) -> None:
    for key in ("output_dir", "cache_dir", "vocab_size", "target_mb"):
        if getattr(args, key) is not None:
            setattr(config, key, getattr(args, key))
    for item in args.pct:
        name, value = _pair(item, "--pct")
        config.sources[name].pct = float(value)
    for item in args.path:
        name, value = _pair(item, "--path")
        # A path given on the command line replaces however the config said to get this
        # source, rather than conflicting with it.
        config.sources[name].path = value
        config.sources[name].download = False
        config.sources[name].diacritize = False
    for name in args.download:
        config.sources[name].download = True
        config.sources[name].path = None
    if args.device:
        config.diacritizer.device = args.device
    if args.allow is not None:
        config.charset.allow = [c.strip() for c in args.allow.split(",") if c.strip()]
    if args.on_foreign is not None:
        config.charset.on_foreign = args.on_foreign
    if args.no_fold:
        config.charset.fold = False


def _pair(item: str, flag: str) -> tuple[str, str]:
    name, sep, value = item.partition("=")
    if not sep or name not in SOURCE_NAMES:
        raise SystemExit(f"{flag} expects SOURCE=VALUE with SOURCE in {SOURCE_NAMES}, got {item!r}")
    return name, value


def validate(config: TokenizerConfig) -> dict[str, SourceSpec]:
    """The sources in use, after checking everything that can be checked without work."""
    if config.vocab_size < 1:
        raise SystemExit(f"vocab_size must be positive, got {config.vocab_size}")
    if config.target_mb <= 0:
        raise SystemExit(f"target_mb must be positive, got {config.target_mb}")
    if config.sampling.mode not in ("prefix", "bounded"):
        raise SystemExit("sampling.mode must be prefix or bounded")
    if config.diacritizer.batch_size < 1 or config.diacritizer.dtype not in (
        "float32",
        "float16",
        "bfloat16",
    ):
        raise SystemExit("invalid diacritizer batch_size or dtype")
    active = {n: s for n, s in config.sources.items() if s.pct > 0}
    if any(s.pct < 0 for s in config.sources.values()):
        raise SystemExit("source shares cannot be negative")
    total = sum(s.pct for s in active.values())
    if abs(total - 100.0) > 0.01:
        shares = ", ".join(f"{n} {s.pct:g}" for n, s in config.sources.items())
        raise SystemExit(f"source shares must add to 100, they add to {total:g} ({shares})")

    for name, spec in active.items():
        if spec.download and name not in DOWNLOADS:
            raise SystemExit(f"source '{name}' cannot be downloaded; only {sorted(DOWNLOADS)} can")
        if spec.download and spec.path:
            raise SystemExit(f"source '{name}' has both a path and download: pick one")
        if name == "hebrew_nikud" and spec.diacritize:
            if spec.path:
                raise SystemExit("hebrew_nikud has both a path and diacritize: pick one")
            if not config.diacritizer.model:
                raise SystemExit(
                    "hebrew_nikud.diacritize needs diacritizer.model (a model id or local "
                    "directory, e.g. dicta-il/dictabert-large-char-menaked)"
                )
            continue
        if not spec.path and not spec.download:
            raise SystemExit(
                f"source '{name}' has {spec.pct:g}% of the corpus but nowhere to read from.\n"
                f"  give it a local folder: --path {name}=DIR (or sources.{name}.path)"
                + (
                    f"\n  or opt in to downloading it: --download {name}"
                    if name in DOWNLOADS
                    else ""
                )
                + f"\n  or leave it out: --pct {name}=0"
            )
    used = dict(active)
    if "hebrew_nikud" in active and active["hebrew_nikud"].diacritize:
        names = active["hebrew_nikud"].diacritize_sources
        if not names or len(names) != len(set(names)) or set(names) - {"manifest", "hebrew"}:
            raise SystemExit(
                "diacritize_sources must list manifest and/or hebrew without duplicates"
            )
        for name in names:
            source = config.sources[name]
            if not source.path and not source.download:
                raise SystemExit(
                    f"hebrew_nikud.diacritize needs the `{name}` source path or download"
                )
            used[name] = source
    # Fail on a missing folder now, not after an hour of diacritizing.
    for name, spec in used.items():
        if name == "hebrew_nikud" and spec.diacritize:
            continue
        if spec.path:
            files = find_files(Path(spec.path))
            if config.sampling.mode == "bounded" and any(
                p.suffix.lower() not in (".txt", ".jsonl") for p in files
            ):
                raise SystemExit("bounded sampling needs uncompressed .txt/.jsonl sources")
        elif config.sampling.mode == "bounded":
            raise SystemExit("bounded sampling needs local source paths, not downloads")
    charset = CharsetFilter(config.charset.allow, config.charset.on_foreign, config.charset.fold)
    if any(charset._map(c) != c for c in config.required_chars):
        raise SystemExit("required_chars must be allowed by the charset without folding/removal")
    return active


def collect(
    documents: Iterable[str], want_chars: int, charset: CharsetFilter, keep: Callable[[str], bool]
) -> list[str]:
    """Filtered sentence-ish lines from `documents` until they total `want_chars`."""
    out, have = [], 0
    for document in documents:
        for line in lines_of(document):
            line = charset.clean(line)
            if not line or not keep(line):
                continue
            out.append(line)
            have += len(line)
        if have >= want_chars:
            break
    return out


def _stream(repo: str, config: str) -> Iterable[str]:
    try:
        from datasets import load_dataset
    except ImportError as error:
        raise SystemExit(
            "--download needs the `datasets` package: `uv sync --extra training`"
        ) from error
    for row in load_dataset(repo, name=config, split="train", streaming=True):
        yield row.get("text") or ""


def _keep_for(name: str) -> Callable[[str], bool]:
    if name == "english":
        # FineWeb-2's Hebrew split still carries Latin pages, and English pages carry the
        # odd Hebrew word; either would quietly move a share into the wrong slice.
        return lambda line: has_latin(line) and not has_hebrew(line)
    if name in ("hebrew", "hebrew_nikud"):
        return has_hebrew
    return lambda line: True


def load_source(
    name: str, spec: SourceSpec, want_chars: int, config: TokenizerConfig, charset: CharsetFilter
) -> list[str]:
    keep = _keep_for(name)
    if spec.path:
        if config.sampling.mode == "bounded":
            lines = sample_source(
                spec.path,
                want_chars,
                charset,
                keep,
                config.sampling,
                config.cache_dir / "samples" / name,
            )
        else:
            lines = collect(iter_documents(Path(spec.path)), want_chars, charset, keep)
        print(f"  {name}: {len(lines)} lines from {spec.path}")
        return lines

    cached = config.cache_dir / f"{name}.txt"
    if cached.exists():
        lines = [
            c
            for line in cached.read_text(encoding="utf-8").splitlines()
            if (c := charset.clean(line))
        ]
        if sum(map(len, lines)) >= want_chars:
            print(f"  {name}: reusing {len(lines)} cached lines from {cached}")
            return lines
    repo, subset = DOWNLOADS[name]
    print(f"  {name}: streaming {repo} ({subset}) for {want_chars / 1e6:.1f}M chars", flush=True)
    lines = collect(_stream(repo, subset), want_chars, charset, keep)
    cached.parent.mkdir(parents=True, exist_ok=True)
    cached.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    return lines


def _without_hebrew_marks(text: str) -> str:
    return "".join(c for c in text if c not in NIQQUD and c not in CANTILLATION)


def diacritize(
    cached: Path, lines: list[str], want_chars: int, spec: DiacritizerSpec, charset: CharsetFilter
) -> list[str]:
    """Vocalize plain Hebrew with a provenance-checked, input-cursor-based resume cache."""
    import re

    provenance = {"inputs": digest(lines), "model": asdict(spec), "charset": charset.describe()}
    progress_path = cached.with_suffix(".progress.json")
    progress = {"provenance": provenance, "next_input": 0, "deduplicated_offsets": True}
    if progress_path.exists():
        progress = json.loads(progress_path.read_text())
        if progress["provenance"] != provenance:
            raise ValueError(f"stale diacritizer cache {cached}; use a new cache directory")
    elif cached.exists():
        raise ValueError(f"unversioned diacritizer cache {cached}; use a new cache directory")
    done = cached.read_text(encoding="utf-8").splitlines() if cached.exists() else []
    if len(done) != progress.get("output_lines", len(done)):
        raise ValueError(f"incomplete diacritizer cache {cached}; use a new cache directory")
    if not progress.get("deduplicated_offsets"):
        # Strict migration for caches from before the offset fix. Only the proven ellipsis
        # triplication is repaired; any other content change requires a fresh cache.
        if progress["next_input"] != len(done):
            raise ValueError(f"legacy cache {cached} has rejected lines; use a new cache directory")
        repaired = []
        for plain, marked in zip(lines, done):
            if _without_hebrew_marks(marked) != _without_hebrew_marks(plain):
                if _without_hebrew_marks(plain).replace("…", "………") != _without_hebrew_marks(
                    marked
                ):
                    raise ValueError(f"unexpected content change in {cached}; use a new cache")
                marked = marked.replace("………", "…")
                if _without_hebrew_marks(marked) != _without_hebrew_marks(plain):
                    raise ValueError(f"cannot safely repair {cached}; use a new cache")
            repaired.append(marked)
        done = repaired
        temporary = cached.with_suffix(".tmp")
        temporary.write_text("\n".join(done) + "\n", encoding="utf-8")
        temporary.replace(cached)
        progress["deduplicated_offsets"] = True
        progress_path.write_text(json.dumps(progress) + "\n")
    if sum(map(len, done)) >= want_chars or progress["next_input"] >= len(lines):
        print(f"  hebrew_nikud: reusing {len(done)} cached lines")
        return done

    print(f"  hebrew_nikud: diacritizing with {spec.model} on {spec.device}, from line {len(done)}")
    predict = load_dicta(spec)
    cached.parent.mkdir(parents=True, exist_ok=True)
    have = sum(map(len, done))
    ws = re.compile(r"\s+")
    start_input = progress["next_input"]
    with cached.open("a", encoding="utf-8", newline="\n") as handle:
        for i in range(progress["next_input"], len(lines), spec.batch_size):
            batch = lines[i : i + spec.batch_size]
            marked = predict(batch)
            if len(marked) != len(batch) or (
                spec.preserve_matres
                and any(
                    _without_hebrew_marks(a) != _without_hebrew_marks(b)
                    for a, b in zip(batch, marked)
                )
            ):
                raise ValueError(
                    "diacritizer changed non-niqqud content or returned the wrong batch size"
                )
            if i == start_input:
                rate = niqqud_rate("".join(marked))
                if rate < _NIKUD_FLOOR:
                    raise SystemExit(
                        f"diacritizer produced {rate:.1%} niqqud, expected ~25%: it is not "
                        f"vocalizing. First line back:\n  {marked[0][:120]}"
                    )
            for line in marked:
                line = charset.clean(ws.sub(" ", line))
                if line:
                    handle.write(line + "\n")
                    done.append(line)
                    have += len(line)
            handle.flush()
            progress.update(next_input=min(i + spec.batch_size, len(lines)), output_lines=len(done))
            temporary = progress_path.with_suffix(".tmp")
            temporary.write_text(json.dumps(progress) + "\n")
            temporary.replace(progress_path)
            if (i // spec.batch_size) % 100 == 0:
                print(f"  vocalized {have:,}/{want_chars:,} chars", flush=True)
            if have >= want_chars:
                break
    print(f"  hebrew_nikud: {len(done)} lines, {have / 1e6:.1f}M chars")
    return done


def clip(lines: list[str], want_chars: int) -> list[str]:
    out, have = [], 0
    for line in lines:
        if have >= want_chars:
            break
        out.append(line)
        have += len(line)
    return out


def _vocalization_seed(config: TokenizerConfig, charset: CharsetFilter):
    """A validated prefix from a smaller pool, so extending a sweep does not redo inference."""
    if config.vocalization_seed_dir is None:
        return [], []
    seed = config.vocalization_seed_dir
    progress = json.loads((seed / "hebrew_nikud.progress.json").read_text())
    inputs = (seed / "vocalization_input.txt").read_text(encoding="utf-8").splitlines()
    outputs = (seed / "hebrew_nikud.txt").read_text(encoding="utf-8").splitlines()
    if progress["provenance"] != {
        "inputs": digest(inputs),
        "model": asdict(config.diacritizer),
        "charset": charset.describe(),
    } or not progress.get("deduplicated_offsets"):
        raise ValueError("vocalization seed provenance does not match the current model/charset")
    if progress["next_input"] != len(outputs) or progress["output_lines"] != len(outputs):
        raise ValueError("vocalization seed must have one accepted output per processed input")
    prefix = inputs[: progress["next_input"]]
    if any(_without_hebrew_marks(a) != _without_hebrew_marks(b) for a, b in zip(prefix, outputs)):
        raise ValueError("vocalization seed changed non-niqqud content")
    return prefix, outputs


def build(config: TokenizerConfig, active: dict[str, SourceSpec]) -> Path:
    charset = CharsetFilter(config.charset.allow, config.charset.on_foreign, config.charset.fold)
    budget = {n: int(config.target_mb * 1e6 * s.pct / 100) for n, s in active.items()}
    print(
        f"building {config.target_mb:g}M chars: "
        + ", ".join(f"{n} {s.pct:g}%" for n, s in active.items())
    )
    print(charset.describe())

    generated = "hebrew_nikud" in active and active["hebrew_nikud"].diacritize
    vocal_sources = active["hebrew_nikud"].diacritize_sources if generated else []
    seed_inputs, seed_outputs = _vocalization_seed(config, charset) if generated else ([], [])
    seed_chars = sum(map(len, seed_outputs))
    extra = max(0, budget["hebrew_nikud"] - seed_chars) // len(vocal_sources) if generated else 0
    pools: dict[str, list[str]] = {}
    for name in dict.fromkeys([*active, *vocal_sources]):
        spec = config.sources[name]
        if name == "hebrew_nikud" and generated:
            continue
        want = budget.get(name, 0) + (extra if name in vocal_sources else 0)
        pools[name] = load_source(name, spec, want, config, charset)
    if generated:
        # Equal-character slices of fresh sampled text from each requested domain. Interleave
        # them so the diacritizer's early stop cannot consume just the first domain.
        candidates = []
        for name in vocal_sources:
            plain_count = len(clip(pools[name], budget.get(name, 0)))
            candidates.append(clip(pools[name][plain_count:], extra))
        # Round-robin by characters, not lines: committee lines tend to be much shorter.
        positions = [0] * len(candidates)
        consumed = [0] * len(candidates)
        vocal_input = list(seed_inputs)
        while any(positions[j] < len(c) for j, c in enumerate(candidates)):
            j = min(
                (j for j, c in enumerate(candidates) if positions[j] < len(c)),
                key=lambda j: consumed[j],
            )
            line = candidates[j][positions[j]]
            vocal_input.append(line)
            positions[j] += 1
            consumed[j] += len(line)
        config.cache_dir.mkdir(parents=True, exist_ok=True)
        (config.cache_dir / "vocalization_input.txt").write_text(
            "\n".join(vocal_input) + "\n", encoding="utf-8"
        )
        cached_vocal = config.cache_dir / "hebrew_nikud.txt"
        if seed_outputs and not cached_vocal.exists():
            cached_vocal.write_text("\n".join(seed_outputs) + "\n", encoding="utf-8")
            cached_vocal.with_suffix(".progress.json").write_text(
                json.dumps(
                    {
                        "provenance": {
                            "inputs": digest(vocal_input),
                            "model": asdict(config.diacritizer),
                            "charset": charset.describe(),
                        },
                        "next_input": len(seed_inputs),
                        "output_lines": len(seed_outputs),
                        "deduplicated_offsets": True,
                        "seed_dir": str(config.vocalization_seed_dir),
                    }
                )
                + "\n"
            )
        pools["hebrew_nikud"] = diacritize(
            cached_vocal, vocal_input, budget["hebrew_nikud"], config.diacritizer, charset
        )
    elif "hebrew_nikud" in active:
        rate = niqqud_rate("".join(pools["hebrew_nikud"]))
        if rate < _NIKUD_FLOOR:
            raise SystemExit(
                f"hebrew_nikud text is {rate:.1%} niqqud, expected ~25%. "
                f"{active['hebrew_nikud'].path} does not look vocalized."
            )

    slices = {n: clip(pools[n], budget[n]) for n in active}
    out = config.output_dir / "corpus.txt"
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8", newline="\n") as handle:
        for name in active:
            handle.write("\n".join(slices[name]) + "\n")
    slice_dir = config.output_dir / "source_slices"
    slice_dir.mkdir(exist_ok=True)
    for name, lines in slices.items():
        (slice_dir / f"{name}.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")

    got = {n: sum(map(len, v)) for n, v in slices.items()}
    whole = sum(got.values()) or 1
    print(f"\n-> {out}  ({whole / 1e6:.2f}M chars)")
    for name, spec in active.items():
        print(
            f"   {name:13s} {got[name] / 1e6:6.2f}M chars  {100 * got[name] / whole:5.1f}%  "
            f"(asked {spec.pct:g}%)  {len(slices[name])} lines"
        )
    short = [n for n in active if got[n] < 0.95 * budget[n]]
    if short:
        print(f"   SHORT: {', '.join(short)} -- that source ran out, shares are off")
    print(charset.stats.report())
    summary = {
        "vocab_size": config.vocab_size,
        "target_mb": config.target_mb,
        "charset": charset.describe(),
        "asked_pct": {n: s.pct for n, s in active.items()},
        "got_pct": {n: round(100 * got[n] / whole, 2) for n in active},
        "chars": got,
        "short": short,
        "vocalization_sources": vocal_sources,
        "vocalization_input_chars": dict(zip(vocal_sources, consumed)) if generated else {},
        "vocalization_seed_dir": str(config.vocalization_seed_dir) if generated else None,
        "vocalization_seed_chars": seed_chars,
    }
    (config.output_dir / "corpus_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return out


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    config = load_config(args.config)
    apply_overrides(config, args)
    active = validate(config)
    corpus = build(config, active)
    if args.train:
        from hebrew_training import train_tokenizer

        # corpus.txt was just written through the charset filter, so it is not applied twice.
        model = train_tokenizer.train(
            [corpus], config.output_dir, config.vocab_size, required_chars=config.required_chars
        )
        train_tokenizer.report(model)
        print(f"\nSet flow_lm.lookup_table.tokenizer_path to {model}; n_bins follows it.")
    else:
        required = (
            f" --required-chars {shlex.quote(config.required_chars)}"
            if config.required_chars
            else ""
        )
        print(
            f"\nNext:\n  python -m hebrew_training.train_tokenizer --input {corpus} "
            f"--output-dir {config.output_dir} --vocab-size {config.vocab_size}{required}"
        )


if __name__ == "__main__":
    main()
