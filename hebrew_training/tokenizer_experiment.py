"""Build bounded tokenizer pools, vocalize, sweep nested corpus sizes and audit held-out text.

uv run --no-sync python -m hebrew_training.tokenizer_experiment \
    --config hebrew_training/configs/tokenizer_he_en_niqqud.yaml

Selection is the smallest tested corpus within 1% of the largest on every held-out set.
This is an operational size/quality tradeoff, not a claim of a universal optimum.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import shutil
import time
import unicodedata
from dataclasses import asdict
from pathlib import Path

import sentencepiece as spm
import yaml

from hebrew_training import build_tokenizer_corpus as builder
from hebrew_training import train_tokenizer
from hebrew_training.tokenizer_charset import (
    CANTILLATION,
    DIGITS,
    HEBREW_LETTERS,
    LATIN_LETTERS,
    NIQQUD,
    CharsetFilter,
)
from hebrew_training.tokenizer_sampling import digest


def unmarked(line: str) -> str:
    return "".join(c for c in line if c not in NIQQUD and c not in CANTILLATION)


def evaluate(model: Path, sets: dict[str, list[str]]) -> dict:
    tokenizer = spm.SentencePieceProcessor(model_file=str(model))
    report = {}
    for name, lines in sets.items():
        ids = tokenizer.encode(lines, out_type=int)
        chars = sum(map(len, lines))
        tokens = sum(map(len, ids))
        byte_tokens = sum(tokenizer.is_byte(i) for row in ids for i in row)
        unknowns = sum(i == tokenizer.unk_id() for row in ids for i in row)
        failures = [
            {"text": line, "decoded": decoded}
            for line, decoded in zip(lines, tokenizer.decode(ids))
            if line != decoded
        ]
        report[name] = {
            "lines": len(lines),
            "chars": chars,
            "tokens": tokens,
            "tokens_per_char": tokens / chars,
            "byte_tokens": byte_tokens,
            "byte_token_pct": 100 * byte_tokens / tokens,
            "unknown_tokens": unknowns,
            "roundtrip_failures": len(failures),
            "failure_examples": failures[:3],
            "line_tokens_p95": sorted(map(len, ids))[int(0.95 * (len(ids) - 1))],
        }
    return report


def choose_size(results: dict, tolerance: float = 0.01) -> float:
    largest = results[max(results)]
    for size, scores in sorted(results.items()):
        if all(
            scores[name]["tokens_per_char"] <= metrics["tokens_per_char"] * (1 + tolerance)
            and not scores[name]["roundtrip_failures"]
            and not scores[name]["unknown_tokens"]
            for name, metrics in largest.items()
        ):
            return size
    raise RuntimeError("no tokenizer passed all round-trip and unknown-token checks")


def artifact_audit(config: builder.TokenizerConfig, selected: Path) -> dict:
    tokenizer = spm.SentencePieceProcessor(model_file=str(selected / "tokenizer.model"))
    coverage = {}
    for name, chars in (
        ("latin_letters", LATIN_LETTERS),
        ("digits", DIGITS),
        ("hebrew_letters", HEBREW_LETTERS),
        ("niqqud", NIQQUD),
    ):
        fallback = []
        for char in sorted(chars):
            if any(tokenizer.is_byte(i) for i in tokenizer.encode("א" + char)):
                fallback.append(f"U+{ord(char):04X} {unicodedata.name(char)}")
        coverage[name] = {"tested": len(chars), "requiring_byte_fallback": fallback}
    source_stats = {}
    for source in ("manifest", "hebrew", "english"):
        report = json.loads((config.cache_dir / "samples" / source / "sampling.json").read_text())
        source_stats[source] = {
            key: report[key]
            for key in (
                "read_bytes",
                "train_chars",
                "heldout_chars",
                "train_groups",
                "heldout_groups",
                "group_overlap",
                "duplicates_skipped",
            )
        }
    metadata = Path(config.diacritizer.model) / ".cache/huggingface/download/config.json.metadata"
    revision = (
        metadata.read_text().splitlines()[0] if metadata.exists() else config.diacritizer.revision
    )
    prior_read_bytes = 0
    seed_dir = config.vocalization_seed_dir
    visited = set()
    while seed_dir is not None:
        if seed_dir in visited:
            raise ValueError("cycle in vocalization seed provenance")
        visited.add(seed_dir)
        for source in ("manifest", "hebrew", "english"):
            report = json.loads((seed_dir / "samples" / source / "sampling.json").read_text())
            prior_read_bytes += report["read_bytes"]
        progress = json.loads((seed_dir / "hebrew_nikud.progress.json").read_text())
        seed_dir = Path(progress["seed_dir"]) if progress.get("seed_dir") else None
    active_read_bytes = sum(s["read_bytes"] for s in source_stats.values())
    return {
        "source_sampling": source_stats,
        "active_pool_source_bytes_read": active_read_bytes,
        "prior_pool_source_bytes_read": prior_read_bytes,
        "total_source_bytes_read": active_read_bytes + prior_read_bytes,
        "dicta_revision": revision,
        "character_coverage": coverage,
        "selected_corpus_utf8_bytes": (selected / "corpus.txt").stat().st_size,
        "sha256": {
            name: hashlib.sha256((selected / name).read_bytes()).hexdigest()
            for name in ("tokenizer.model", "corpus.txt")
        },
    }


def prepare_evaluation(config: builder.TokenizerConfig, root: Path) -> dict[str, list[str]]:
    # Compare unmarked forms as well, so vocalizing a training sentence cannot leak it into
    # evaluation. Exclude exact repeats across domains, not only within each sampled source.
    sampled = config.cache_dir / "samples"
    train_keys = set()
    for name in ("manifest", "hebrew", "english"):
        for line in (sampled / name / "train.txt").read_text(encoding="utf-8").splitlines():
            train_keys.add(digest(unmarked(line)))
    for line in (
        (config.cache_dir / "vocalization_input.txt").read_text(encoding="utf-8").splitlines()
    ):
        train_keys.add(digest(unmarked(line)))
    names = {"manifest": "committee", "hebrew": "general_hebrew", "english": "english"}
    sets, excluded = {}, {}
    for source, name in names.items():
        lines = (sampled / source / "heldout.txt").read_text(encoding="utf-8").splitlines()
        sets[name] = [line for line in lines if digest(unmarked(line)) not in train_keys]
        excluded[name] = len(lines) - len(sets[name])
        if not sets[name]:
            raise RuntimeError(f"no held-out text for {name}")
    evaluation_dir = root / f"evaluation{config.target_mb:g}m"
    evaluation_dir.mkdir(exist_ok=True)
    for name in ("committee", "general_hebrew"):
        plain = sets[name]
        marked = builder.diacritize(
            evaluation_dir / f"{name}_vocalized.txt",
            plain,
            sum(map(len, plain)) * 3,
            config.diacritizer,
            CharsetFilter(config.charset.allow, config.charset.on_foreign, config.charset.fold),
        )
        if len(marked) != len(plain) or any(
            unmarked(a) != unmarked(b) for a, b in zip(plain, marked)
        ):
            raise RuntimeError(f"diacritizer changed non-niqqud content in {name}")
        sets[f"{name}_vocalized"] = marked
    for name, lines in sets.items():
        (evaluation_dir / f"{name}.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (evaluation_dir / "audit.json").write_text(
        json.dumps(
            {
                "exact_unmarked_train_overlap_removed": excluded,
                "split": "committee recording / text-record hash splits; exact unmarked cross-source repeats removed",
                "limitation": "plain-text sources have no original web-page IDs; near-duplicate pages are not detected",
                "code_switch_set": "not available; no synthetic substitute added",
            },
            indent=2,
        )
        + "\n"
    )
    return sets


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--sizes", type=float, nargs="+")
    parser.add_argument(
        "--extend-to", type=float, help="Extend the previous pool, reusing vocalization."
    )
    parser.add_argument("--reuse-pool", action="store_true")
    args = parser.parse_args(argv)
    config = builder.load_config(args.config)
    root = config.output_dir.parent
    if args.extend_to is not None:
        previous = json.loads((root / "results.json").read_text())
        if args.extend_to <= previous["config"]["target_mb"]:
            parser.error("--extend-to must exceed the previous maximum pool size")
        config.vocalization_seed_dir = Path(previous["config"]["cache_dir"])
        config.target_mb = args.extend_to
        config.output_dir = root / f"pool{args.extend_to:g}"
        config.cache_dir = root / f"cache{args.extend_to:g}"
        args.sizes = args.sizes or [*map(float, previous["results"]), args.extend_to]
    if args.sizes is None:
        args.sizes = []
        size = 4.0
        while size < config.target_mb:
            args.sizes.append(size)
            size *= 2
        args.sizes.append(config.target_mb)
    if max(args.sizes) > config.target_mb or min(args.sizes) <= 0:
        parser.error("sizes must be positive and no larger than config.target_mb")
    if config.sampling.mode != "bounded":
        parser.error("experiment requires sampling.mode: bounded")
    root.mkdir(parents=True, exist_ok=True)
    (root / "experiment_config.yaml").write_text(
        yaml.safe_dump(json.loads(json.dumps(asdict(config), default=str)), sort_keys=False)
    )
    if not args.reuse_pool:
        builder.build(config, builder.validate(config))
    sets = prepare_evaluation(config, root)
    results, timings = {}, {}
    for size in sorted(set(args.sizes)):
        name = f"size{size:g}m"
        variant = copy.deepcopy(config)
        variant.output_dir = root / name
        variant.target_mb = size
        variant.sampling.mode = "prefix"
        for source, spec in variant.sources.items():
            spec.path = str(config.output_dir / "source_slices" / f"{source}.txt")
            spec.download = spec.diacritize = False
        corpus = builder.build(variant, builder.validate(variant))
        model = variant.output_dir / "tokenizer.model"
        stamp = variant.output_dir / "training_provenance.json"
        fingerprint = digest(
            {
                "corpus_sha": digest(corpus.read_text(encoding="utf-8")),
                "vocab_size": variant.vocab_size,
                "required_chars": variant.required_chars,
            }
        )
        start = time.perf_counter()
        if (
            not model.exists()
            or not stamp.exists()
            or json.loads(stamp.read_text())["hash"] != fingerprint
        ):
            model = train_tokenizer.train(
                [corpus],
                variant.output_dir,
                variant.vocab_size,
                required_chars=variant.required_chars,
            )
            stamp.write_text(
                json.dumps({"hash": fingerprint, "seconds": time.perf_counter() - start}) + "\n"
            )
        timings[size] = json.loads(stamp.read_text())["seconds"]
        results[size] = evaluate(model, sets)
        (variant.output_dir / "evaluation.json").write_text(
            json.dumps(results[size], indent=2) + "\n"
        )
        print(f"SIZE {size:g}M RESULTS: {json.dumps(results[size])}", flush=True)
    selected_size = choose_size(results)
    selected = root / "selected"
    selected.mkdir(exist_ok=True)
    source = root / f"size{selected_size:g}m"
    for name in (
        "tokenizer.model",
        "tokenizer.vocab",
        "corpus.txt",
        "corpus_summary.json",
        "evaluation.json",
    ):
        shutil.copyfile(source / name, selected / name)
    summary = {
        "selected_million_chars": selected_size,
        "selection_rule": "smallest within 1% of largest tested on every held-out set; no unknowns/roundtrip failures",
        "boundary_selected": selected_size == max(results),
        "training_seconds": timings,
        "results": results,
        "config": json.loads(json.dumps(asdict(config), default=str)),
        "audit": artifact_audit(config, selected),
    }
    (root / "results.json").write_text(json.dumps(summary, indent=2) + "\n")
    languages = list(next(iter(results.values())))
    rows = [
        "# Hebrew / English / niqqud tokenizer experiment",
        "",
        f"Selected **{selected_size:g}M characters**, **{config.vocab_size} pieces**.",
        "Mix: committee 20%, general Hebrew 40%, vocalized Hebrew 20%, English 20%.",
        "",
        "Tokens per character (lower is better):",
        "",
        "| Million chars | " + " | ".join(languages) + " | Training seconds |",
        "|---|" + "---|" * (len(languages) + 1),
    ]
    for size, scores in results.items():
        rows.append(
            f"| {size:g} | "
            + " | ".join(f"{scores[n]['tokens_per_char']:.4f}" for n in languages)
            + f" | {timings[size]:.1f} |"
        )
    rows += [
        "",
        "Selected model: `selected/tokenizer.model`; input: `selected/corpus.txt`.",
        "Selection: smallest within 1% of largest tested on every held-out set.",
        "No code-switch set was available. This evaluates encoding, not speech quality.",
        "Sampling is byte-weighted with per-document caps, not uniform over documents.",
        "Plain-text holdouts are record-hash split; original web-page IDs are unavailable.",
        "",
        f"Raw source reads (successful sampling passes): {summary['audit']['total_source_bytes_read'] / 1e6:.1f} MB.",
        f"Dicta revision: `{summary['audit']['dicta_revision']}`.",
        "",
        "Selected tokenizer byte-fallback percentages (of tokens):",
    ]
    for name, metrics in results[selected_size].items():
        rows.append(
            f"- {name}: {metrics['byte_token_pct']:.4f}%; "
            f"{metrics['roundtrip_failures']} round-trip failures, "
            f"{metrics['unknown_tokens']} unknown tokens."
        )
    rows += ["", "Character audit:"]
    for name, coverage in summary["audit"]["character_coverage"].items():
        missing = coverage["requiring_byte_fallback"]
        rows.append(
            f"- {name}: {coverage['tested'] - len(missing)}/{coverage['tested']} without byte fallback."
        )
    if summary["boundary_selected"]:
        rows += ["", "The largest size won: saturation is not established; extend the sweep."]
    (root / "REPORT.md").write_text("\n".join(rows) + "\n")
    print(f"SELECTED: {selected / 'tokenizer.model'} ({selected_size:g}M chars)", flush=True)


if __name__ == "__main__":
    main()
