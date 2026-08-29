from __future__ import annotations

import argparse
import json
from pathlib import Path

import sentencepiece as spm


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a 4,000-piece Hebrew tokenizer.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--vocab-size", type=int, default=4000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    prefix = args.output_dir / "tokenizer"
    spm.SentencePieceTrainer.train(
        input=str(args.input.resolve()),
        model_prefix=str(prefix.resolve()),
        vocab_size=args.vocab_size,
        model_type="unigram",
        character_coverage=1.0,
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
    tokenizer = spm.SentencePieceProcessor(model_file=str(prefix.with_suffix(".model")))
    if tokenizer.vocab_size() != args.vocab_size:
        raise RuntimeError(
            f"Expected {args.vocab_size} pieces, tokenizer has {tokenizer.vocab_size()}"
        )
    probe = "שלום, זהו מבחן של מודל דיבור בעברית."
    ids = tokenizer.encode(probe, out_type=int)
    decoded = tokenizer.decode(ids)
    print(f"Tokenizer: {prefix.with_suffix('.model')}")
    print(f"Vocabulary: {tokenizer.vocab_size()}")
    print("Probe pieces: " + json.dumps(tokenizer.encode(probe, out_type=str), ensure_ascii=True))
    print("Round trip: " + json.dumps(decoded, ensure_ascii=True))


if __name__ == "__main__":
    main()
