"""Train the Hebrew SentencePiece tokenizer.

    python -m hebrew_training.train_tokenizer \
        --input artifacts/tokenizer/corpus.txt --output-dir tokenizers --vocab-size 8000

`--input` is repeatable and takes text files, jsonl (the "text"/"transcript" field) and
parquet, or folders of them (see tokenizer_text.py). Use build_tokenizer_corpus.py to make a
mixed corpus first; a tokenizer fitted on one kind of text cannot spell any other.

Every input line goes through the charset filter (tokenizer_charset.py): Latin, Hebrew
letters, niqqud, digits and punctuation only, anything else dropped. `--no-charset-filter`
turns that off.

The vocabulary size is a parameter and is *not* repeated in the training config: the trainer
reads it back from the `.model` this writes (training/modules/builders.py sets
`flow_lm.lookup_table.n_bins` from the tokenizer), so changing `--vocab-size` here is the only
edit needed.
"""

from __future__ import annotations

import argparse
import itertools
import json
import re
from collections.abc import Iterator
from pathlib import Path

import sentencepiece as spm

from hebrew_training.tokenizer_charset import CharsetFilter, add_charset_args, filter_from_args
from hebrew_training.tokenizer_text import iter_documents, lines_of

DEFAULT_VOCAB_SIZE = 8000

# One sentence per kind of text the tokenizer is meant to carry.
PROBES = {
    "hebrew": "שלום, זהו מבחן של מודל דיבור בעברית.",
    "niqqud": "אֲנִי רוֹצָה לָלֶכֶת הַבַּיְתָה, תּוֹדָה רַבָּה.",
    "english": "He joined Microsoft before moving to the platform team.",
}
_BYTE_PIECE = re.compile(r"^<0x[0-9A-F]{2}>$")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "--input",
        type=Path,
        action="append",
        required=True,
        help="Text file, jsonl, parquet, or a folder of them. Repeatable.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--required-chars",
        default="",
        help="Reserve these characters even if absent from the corpus.",
    )
    parser.add_argument(
        "--vocab-size",
        type=int,
        default=DEFAULT_VOCAB_SIZE,
        help=f"Number of pieces (default: {DEFAULT_VOCAB_SIZE}). The training config picks it "
        "up from the tokenizer file; nothing else needs to change.",
    )
    add_charset_args(parser)
    return parser.parse_args(argv)


def iter_lines(inputs: list[Path], charset: CharsetFilter | None) -> Iterator[str]:
    """Every input as sentence-per-line text, charset-filtered, one line at a time."""
    for path in inputs:
        for document in iter_documents(path):
            for line in lines_of(document):
                if charset is not None:
                    line = charset.clean(line)
                if line:
                    yield line


def train(
    inputs: list[Path],
    output_dir: Path,
    vocab_size: int = DEFAULT_VOCAB_SIZE,
    charset: CharsetFilter | None = None,
    required_chars: str = "",
) -> Path:
    """Fit the tokenizer on `inputs` and write `output_dir/tokenizer.model`.

    Lines are fed to SentencePiece as they are read and filtered; no intermediate corpus file
    is written (SentencePiece keeps its own in-memory copy to shuffle, whatever the input).
    """
    if charset is not None and any(charset._map(c) != c for c in required_chars):
        raise ValueError("required_chars must be allowed by the charset without folding/removal")
    output_dir.mkdir(parents=True, exist_ok=True)
    prefix = output_dir / "tokenizer"
    lines = iter_lines(inputs, charset)
    first = next(lines, None)
    if first is None:
        if charset is not None:
            print(charset.stats.report())
        raise SystemExit("no text left to train on after reading and filtering the inputs")
    count = 0

    def counted() -> Iterator[str]:
        nonlocal count
        for line in itertools.chain([first], lines):
            count += 1
            yield line

    if charset is not None:
        print(charset.describe())
    print(f"training a {vocab_size}-piece tokenizer")
    spm.SentencePieceTrainer.train(
        sentence_iterator=counted(),
        model_prefix=str(prefix.resolve()),
        vocab_size=vocab_size,
        model_type="unigram",
        character_coverage=1.0,
        required_chars=required_chars,
        byte_fallback=True,
        split_digits=True,
        normalization_rule_name="identity",
        bos_id=1,
        eos_id=2,
        unk_id=0,
        pad_id=-1,
        hard_vocab_limit=True,
        input_sentence_size=0,
        shuffle_input_sentence=True,
    )
    print(f"trained on {count} lines")
    if charset is not None:
        print(charset.stats.report())
    model = prefix.with_suffix(".model")
    tokenizer = spm.SentencePieceProcessor(model_file=str(model))
    if tokenizer.vocab_size() != vocab_size:
        raise RuntimeError(f"Expected {vocab_size} pieces, tokenizer has {tokenizer.vocab_size()}")
    return model


def report(model: Path) -> None:
    tokenizer = spm.SentencePieceProcessor(model_file=str(model))
    print(f"Tokenizer: {model}")
    print(f"Vocabulary: {tokenizer.vocab_size()}  (flow_lm.lookup_table.n_bins follows this)")
    for name, probe in PROBES.items():
        pieces = tokenizer.encode(probe, out_type=str)
        bytes_used = sum(bool(_BYTE_PIECE.match(p)) for p in pieces)
        ok = tokenizer.decode(tokenizer.encode(probe, out_type=int)) == probe
        print(
            f"  {name:8s} {len(pieces) / len(probe):.2f} tokens/char, {bytes_used} byte pieces, "
            f"round trip {'ok' if ok else 'FAILED'}"
        )
    print(
        "Probe pieces: "
        + json.dumps(tokenizer.encode(PROBES["hebrew"], out_type=str), ensure_ascii=True)
    )


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    model = train(
        args.input, args.output_dir, args.vocab_size, filter_from_args(args), args.required_chars
    )
    report(model)


if __name__ == "__main__":
    main()
