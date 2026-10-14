"""Build the four-way text corpus the Hebrew tokenizer is trained on.

    python -m hebrew_training.build_tokenizer_corpus \
        --manifest-text artifacts/hebrew_server_8s/tokenizer_corpus.txt \
        --out artifacts/hebrew_server_8s/tokenizer_corpus_mixed.txt

`train_tokenizer.py` takes one text file. Today that file is the training manifest's own
text, which is Knesset and nothing else, and a tokenizer built from it reads anything else
badly: vocalized Hebrew costs it 0.936 tokens per character and English 0.977 -- both are
byte fallback, which is to say it cannot spell them at all.

This writes a file with four kinds of text in it instead, in the shares the measurements in
docs/hebrew-english-data-mixing.md settled on:

    manifest    15%   the training manifest's own text (--manifest-text), normalized.
                      Whatever the model trains on -- CrowdRecital today, Knesset later.
                      Rebuild this slice whenever the training corpus changes.
    general he  40%   HuggingFaceFW/fineweb-2, config heb_Hebr
    gen nikud   15%   that same FineWeb-2 text, diacritized with DictaBERT
    english     30%   HuggingFaceFW/fineweb, sample-10BT

Shares are of **characters**, not lines or documents: what matters is how much of each kind
the tokenizer actually sees.

Two things worth knowing before running it:

- **The vocalized slice has to be generated.** Only 0.11% of FineWeb-2 Hebrew carries nikud,
  so it cannot be collected. `dicta-il/dictabert-large-char-menaked` writes it. That is the
  slow step and the only one that wants a GPU, so its output is cached under --cache and a
  re-run picks up where it stopped.
- **A tokenizer saturates on a few MB.** --target-mb defaults to 16, already generous for
  8,000 pieces. There is no reason to stream hundreds of GB of FineWeb.

Needs: datasets, transformers, torch, sentencepiece.
"""

from __future__ import annotations

import argparse
import re
import unicodedata
from pathlib import Path

SOURCES = {
    "general": ("HuggingFaceFW/fineweb-2", "heb_Hebr"),
    "english": ("HuggingFaceFW/fineweb", "sample-10BT"),
}
DIACRITIZER = "dicta-il/dictabert-large-char-menaked"
# A SentencePiece line should be a sentence, not a web page. Longer than this and DictaBERT
# also starts truncating, which would silently drop the tail of every long paragraph.
MAX_LINE = 400
_WS = re.compile(r"\s+")
_HEBREW = re.compile(r"[֐-׿]")
_NIKUD = re.compile(r"[֑-ׇ]")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--manifest-text", type=Path, required=True,
                   help="The existing tokenizer_corpus.txt that prepare_data_v2.py writes.")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--cache", type=Path, default=Path("artifacts/tokenizer_corpus_cache"),
                   help="Where the downloaded and diacritized slices are kept between runs.")
    p.add_argument("--target-mb", type=float, default=16.0,
                   help="Total size of the mixed corpus (default: 16).")
    p.add_argument("--manifest", type=float, default=15.0)
    p.add_argument("--general", type=float, default=40.0)
    p.add_argument("--gen-nikud", type=float, default=15.0)
    p.add_argument("--english", type=float, default=30.0)
    p.add_argument("--device", default="cuda", help="Device for DictaBERT (cuda, cpu).")
    p.add_argument("--batch-size", type=int, default=32, help="Diacritizer batch size.")
    p.add_argument("--train", action="store_true",
                   help="Also run train_tokenizer.py on the result, at --vocab-size.")
    p.add_argument("--vocab-size", type=int, default=8000)
    return p.parse_args()


def lines_of(text: str) -> list[str]:
    """One sentence-ish line per entry, normalized, nothing empty or enormous."""
    out = []
    for raw in unicodedata.normalize("NFC", text).splitlines():
        line = _WS.sub(" ", raw).strip()
        while len(line) > MAX_LINE:
            cut = line.rfind(" ", 0, MAX_LINE)
            cut = cut if cut > MAX_LINE // 2 else MAX_LINE
            out.append(line[:cut].strip())
            line = line[cut:].strip()
        if len(line) > 1:
            out.append(line)
    return out


def take(path: Path, repo: str, config: str, want_chars: int, hebrew_only: bool) -> list[str]:
    """`want_chars` of a streamed dataset, cached as a text file."""
    if path.exists():
        kept = path.read_text(encoding="utf-8").splitlines()
        if sum(len(x) for x in kept) >= want_chars:
            print(f"  {path.name}: reusing {len(kept)} cached lines")
            return kept
    from datasets import load_dataset

    print(f"  {path.name}: streaming {repo} ({config}) for {want_chars / 1e6:.1f} MB", flush=True)
    kept, have = [], 0
    for row in load_dataset(repo, name=config, split="train", streaming=True):
        for line in lines_of(row.get("text") or ""):
            # FineWeb-2's Hebrew split still carries some fully Latin pages; they would
            # quietly inflate the English share and shrink the Hebrew one.
            if hebrew_only and not _HEBREW.search(line):
                continue
            kept.append(line)
            have += len(line)
        if have >= want_chars:
            break
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(kept) + "\n", encoding="utf-8", newline="\n")
    print(f"  {path.name}: {len(kept)} lines, {have / 1e6:.1f} MB")
    return kept


def diacritize(path: Path, lines: list[str], want_chars: int, device: str, batch: int) -> list[str]:
    """Nikud onto plain Hebrew, cached -- this is the slow step.

    Deliberately the *same* text as the general-Hebrew slice rather than fresh documents:
    that is what was measured, and it teaches the tokenizer both spellings of one vocabulary
    instead of two disjoint ones.
    """
    done = path.read_text(encoding="utf-8").splitlines() if path.exists() else []
    if sum(len(x) for x in done) >= want_chars:
        print(f"  {path.name}: reusing {len(done)} cached lines")
        return done

    import torch
    from transformers import AutoModel, PreTrainedTokenizerFast

    print(f"  {path.name}: diacritizing on {device} from line {len(done)}", flush=True)
    # Not AutoTokenizer. This is a *character* model, and its char-level pre-tokenizer lives
    # only in tokenizer.json; under transformers 5.x AutoTokenizer builds a word-level
    # BertTokenizer from vocab.txt instead, which tokenizes every Hebrew word as [UNK] with
    # word-sized offsets. predict() walks those offsets, skips every token that is not one
    # character, and returns a string of repeated prefixes with no nikud in it at all -- no
    # error, just quiet rubbish. PreTrainedTokenizerFast loads tokenizer.json.
    tok = PreTrainedTokenizerFast.from_pretrained(DIACRITIZER)
    model = AutoModel.from_pretrained(DIACRITIZER, trust_remote_code=True).eval().to(device)
    path.parent.mkdir(parents=True, exist_ok=True)
    have = sum(len(x) for x in done)
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        for i in range(len(done), len(lines), batch):
            chunk = lines[i:i + batch]
            with torch.inference_mode():
                marked = model.predict(chunk, tok)
            if i == len(done):
                # The whole point of this slice is the nikud. If the diacritizer is quietly
                # doing nothing -- a wrong tokenizer does exactly that -- stop here rather
                # than spend an hour writing plain text into a file labelled vocalized.
                rate = sum(len(_NIKUD.findall(x)) for x in marked) / max(
                    1, sum(len(x) for x in marked))
                if rate < 0.05:
                    raise SystemExit(
                        f"diacritizer produced {rate:.1%} nikud characters, expected ~25%. "
                        f"It is not vocalizing. First line back:\n  {marked[0][:120]}")
            for line in marked:
                line = _WS.sub(" ", line).strip()
                if line:
                    handle.write(line + "\n")
                    done.append(line)
                    have += len(line)
            if have >= want_chars:
                break
            if (i // batch) % 20 == 0:
                print(f"    {have / 1e6:.1f} / {want_chars / 1e6:.1f} MB", flush=True)
    print(f"  {path.name}: {len(done)} lines, {have / 1e6:.1f} MB")
    return done


def clip(lines: list[str], want_chars: int) -> list[str]:
    out, have = [], 0
    for line in lines:
        if have >= want_chars:
            break
        out.append(line)
        have += len(line)
    return out


def main() -> None:
    args = parse_args()
    shares = {"manifest": args.manifest, "general": args.general,
              "gen_nikud": args.gen_nikud, "english": args.english}
    total = sum(shares.values())
    if abs(total - 100.0) > 0.01:
        raise SystemExit(f"shares must add to 100, they add to {total}")
    budget = {k: int(args.target_mb * 1e6 * v / 100) for k, v in shares.items()}

    print(f"building {args.target_mb:.0f} MB: " +
          ", ".join(f"{k} {v:.0f}%" for k, v in shares.items()))

    manifest = lines_of(args.manifest_text.read_text(encoding="utf-8"))
    print(f"  manifest: {len(manifest)} lines, "
          f"{sum(len(x) for x in manifest) / 1e6:.1f} MB available")

    # The vocalized slice is cut from the general-Hebrew pool, so that pool covers both.
    repo, config = SOURCES["general"]
    general = take(args.cache / "general_he.txt", repo, config,
                   budget["general"] + budget["gen_nikud"], hebrew_only=True)
    repo, config = SOURCES["english"]
    english = take(args.cache / "english.txt", repo, config, budget["english"], hebrew_only=False)
    nikud = diacritize(args.cache / "gen_nikud.txt", general, budget["gen_nikud"],
                       args.device, args.batch_size)

    slices = {
        "manifest": clip(manifest, budget["manifest"]),
        "general": clip(general, budget["general"]),
        "gen_nikud": clip(nikud, budget["gen_nikud"]),
        "english": clip(english, budget["english"]),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8", newline="\n") as handle:
        for name in ("manifest", "general", "gen_nikud", "english"):
            handle.write("\n".join(slices[name]) + "\n")

    # What actually landed, which is not what was asked for whenever a source ran out.
    got = {k: sum(len(x) for x in v) for k, v in slices.items()}
    whole = sum(got.values())
    print(f"\n-> {args.out}  ({whole / 1e6:.1f} MB)")
    for name in ("manifest", "general", "gen_nikud", "english"):
        print(f"   {name:10s} {got[name] / 1e6:6.2f} MB  {100 * got[name] / whole:5.1f}%  "
              f"(asked {shares[name]:.0f}%)  {len(slices[name])} lines")
    short = [k for k in got if got[k] < 0.95 * budget[k]]
    if short:
        print(f"   SHORT: {', '.join(short)} -- that source ran out, shares are off")

    print(f"\nNext:\n  python -m hebrew_training.train_tokenizer --input {args.out} "
          f"--output-dir {args.out.parent} --vocab-size {args.vocab_size}")
    print(f"  then set flow_lm.lookup_table.n_bins: {args.vocab_size} in the config")
    if args.train:
        import subprocess
        import sys

        subprocess.run([sys.executable, "-m", "hebrew_training.train_tokenizer",
                        "--input", str(args.out), "--output-dir", str(args.out.parent),
                        "--vocab-size", str(args.vocab_size)], check=True)


if __name__ == "__main__":
    main()
