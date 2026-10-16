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
    hebrew_nikud  vocalized Hebrew. Either text you already have, or the `hebrew` text
                  diacritized with a DictaBERT model (`diacritize: true`).
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
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field, fields
from pathlib import Path

import yaml

from hebrew_training.tokenizer_charset import (
    CATEGORIES,
    CharsetFilter,
    add_charset_args,
    has_hebrew,
    has_latin,
    niqqud_rate,
)
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


@dataclass
class CharsetSpec:
    allow: list[str] = field(default_factory=lambda: list(CATEGORIES))
    on_foreign: str = "drop-line"
    fold: bool = True


@dataclass
class DiacritizerSpec:
    # A Hugging Face model id or a local directory. No default on purpose: loading one by id
    # downloads it, and that should be written down in the config, not implied.
    model: str | None = None
    device: str = "cuda"
    batch_size: int = 32


@dataclass
class TokenizerConfig:
    vocab_size: int = 8000
    target_mb: float = 16.0
    output_dir: Path = Path("artifacts/tokenizer")
    cache_dir: Path = Path("artifacts/tokenizer/cache")
    charset: CharsetSpec = field(default_factory=CharsetSpec)
    diacritizer: DiacritizerSpec = field(default_factory=DiacritizerSpec)
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
    nested = {"charset": CharsetSpec, "diacritizer": DiacritizerSpec}
    for key, cls in nested.items():
        raw[key] = _build(cls, raw.get(key) or {}, key)
    for key in ("output_dir", "cache_dir"):
        if key in raw:
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
        hebrew = config.sources["hebrew"]
        if not hebrew.path and not hebrew.download:
            raise SystemExit(
                "hebrew_nikud.diacritize works from the `hebrew` source, which has no path "
                "and no download. Point sources.hebrew.path at plain Hebrew text (its pct may "
                "stay 0), or give hebrew_nikud its own path of vocalized text."
            )
        used["hebrew"] = hebrew
    # Fail on a missing folder now, not after an hour of diacritizing.
    for spec in used.values():
        if spec.path:
            find_files(Path(spec.path))
    CharsetFilter(config.charset.allow, config.charset.on_foreign, config.charset.fold)
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
    print(f"  {name}: streaming {repo} ({subset}) for {want_chars / 1e6:.1f} MB", flush=True)
    lines = collect(_stream(repo, subset), want_chars, charset, keep)
    cached.parent.mkdir(parents=True, exist_ok=True)
    cached.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    return lines


def diacritize(
    cached: Path, lines: list[str], want_chars: int, spec: DiacritizerSpec, charset: CharsetFilter
) -> list[str]:
    """Nikud onto plain Hebrew, cached because it is the slow step. Resumes where it stopped.

    Deliberately the *same* text as the `hebrew` slice rather than fresh documents: it teaches
    the tokenizer both spellings of one vocabulary instead of two disjoint ones.
    """
    import re

    done = cached.read_text(encoding="utf-8").splitlines() if cached.exists() else []
    if sum(map(len, done)) >= want_chars:
        print(f"  hebrew_nikud: reusing {len(done)} cached lines")
        return done

    import torch
    from transformers import AutoModel, PreTrainedTokenizerFast

    print(f"  hebrew_nikud: diacritizing with {spec.model} on {spec.device}, from line {len(done)}")
    # Not AutoTokenizer. This is a *character* model whose char-level pre-tokenizer lives only
    # in tokenizer.json; under transformers 5.x AutoTokenizer builds a word-level BertTokenizer
    # from vocab.txt instead, tokenizing every Hebrew word as [UNK]. predict() then returns
    # repeated prefixes with no nikud at all -- no error, just quiet rubbish.
    tokenizer = PreTrainedTokenizerFast.from_pretrained(spec.model)
    model = AutoModel.from_pretrained(spec.model, trust_remote_code=True).eval().to(spec.device)
    cached.parent.mkdir(parents=True, exist_ok=True)
    have = sum(map(len, done))
    ws = re.compile(r"\s+")
    with cached.open("a", encoding="utf-8", newline="\n") as handle:
        for i in range(len(done), len(lines), spec.batch_size):
            with torch.inference_mode():
                marked = model.predict(lines[i : i + spec.batch_size], tokenizer)
            if i == len(done):
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
            if have >= want_chars:
                break
    print(f"  hebrew_nikud: {len(done)} lines, {have / 1e6:.1f} MB")
    return done


def clip(lines: list[str], want_chars: int) -> list[str]:
    out, have = [], 0
    for line in lines:
        if have >= want_chars:
            break
        out.append(line)
        have += len(line)
    return out


def build(config: TokenizerConfig, active: dict[str, SourceSpec]) -> Path:
    charset = CharsetFilter(config.charset.allow, config.charset.on_foreign, config.charset.fold)
    budget = {n: int(config.target_mb * 1e6 * s.pct / 100) for n, s in active.items()}
    print(
        f"building {config.target_mb:g} MB: "
        + ", ".join(f"{n} {s.pct:g}%" for n, s in active.items())
    )
    print(charset.describe())

    generated = "hebrew_nikud" in active and active["hebrew_nikud"].diacritize
    pools: dict[str, list[str]] = {}
    for name, spec in active.items():
        if name == "hebrew_nikud" and generated:
            continue
        # The vocalized slice is cut from the same pool, so that pool has to cover both.
        want = budget[name]
        if generated and name == "hebrew":
            want = max(want, budget["hebrew_nikud"])
        pools[name] = load_source(name, spec, want, config, charset)
    if generated:
        if "hebrew" not in pools:  # pct 0: still the text the vocalized slice is made from
            pools["hebrew"] = load_source(
                "hebrew", config.sources["hebrew"], budget["hebrew_nikud"], config, charset
            )
        pools["hebrew_nikud"] = diacritize(
            config.cache_dir / "hebrew_nikud.txt",
            pools["hebrew"],
            budget["hebrew_nikud"],
            config.diacritizer,
            charset,
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

    got = {n: sum(map(len, v)) for n, v in slices.items()}
    whole = sum(got.values()) or 1
    print(f"\n-> {out}  ({whole / 1e6:.2f} MB)")
    for name, spec in active.items():
        print(
            f"   {name:13s} {got[name] / 1e6:6.2f} MB  {100 * got[name] / whole:5.1f}%  "
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
        model = train_tokenizer.train([corpus], config.output_dir, config.vocab_size)
        train_tokenizer.report(model)
        print(f"\nSet flow_lm.lookup_table.tokenizer_path to {model}; n_bins follows it.")
    else:
        print(
            f"\nNext:\n  python -m hebrew_training.train_tokenizer --input {corpus} "
            f"--output-dir {config.output_dir} --vocab-size {config.vocab_size}"
        )


if __name__ == "__main__":
    main()
