"""Retarget a released Pocket TTS checkpoint at a new tokenizer, offline.

Kyutai release 24-layer teachers alongside the 6-layer models --
`languages/english_2026-04_24l/model.safetensors` really does carry 24 transformer
layers and the flow head. So `start_from_pretrained: true` on the teacher config is
possible; the only thing that does not survive the language change is the text table,
`flow_lm.conditioner.embed.weight`, whose [n_bins + 1, dim] rows are one-per-piece of the
ENGLISH tokenizer. Point a Hebrew tokenizer of the same size at it and every row keeps its
shape while losing its meaning -- `load_state_dict(strict=True)` succeeds and says nothing.

This rewrites that one tensor ahead of time, so the trainer needs no patching: the output
is an ordinary checkpoint that loads strictly against a Hebrew config.

Rows are transplanted the way hebrew_training/model_utils.py:install_tokenizer did it for
the earlier run -- matched on the piece STRING, so a piece in both vocabularies keeps its
learned embedding. Punctuation, digits, spaces and byte-fallback pieces are shared between
any two SentencePiece models, so this is not a rounding error. Everything else is drawn
from the old table's mean and standard deviation rather than left at zero.

    python -m hebrew_training.warm_start_checkpoint \
        --tokenizer tokenizers/hebrew.model \
        --out weights/hebrew_24l_warmstart.safetensors

Then in your model config:

    weights_path: weights/hebrew_24l_warmstart.safetensors
    flow_lm.lookup_table.n_bins: <the new tokenizer's vocab size>

and set `start_from_pretrained: true` in lsd_scratch.yaml.

NOTE this is not a path Kyutai measured. Both their configs train from scratch, and their
published 2000 h result is from scratch. Treat it as an experiment worth running against a
from-scratch baseline, not as the recommended route.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

# The released teacher. Its 6-layer sibling is languages/english/model.safetensors.
DEFAULT_WEIGHTS = (
    "hf://kyutai/pocket-tts/languages/english_2026-04_24l/model.safetensors"
    "@492522650173a0653b7575cdc25ae09810e5d741"
)
DEFAULT_SOURCE_TOKENIZER = (
    "hf://kyutai/pocket-tts-without-voice-cloning/languages/english_2026-04_24l/tokenizer.model"
    "@e81d79e8194ad4c7ce879c87a4258ef20cbf2487"
)
EMBED_KEY = "flow_lm.conditioner.embed.weight"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "--tokenizer",
        type=Path,
        required=True,
        help="The new SentencePiece model, from train_tokenizer.py.",
    )
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--weights", default=DEFAULT_WEIGHTS)
    parser.add_argument("--source-tokenizer", default=DEFAULT_SOURCE_TOKENIZER)
    return parser.parse_args()


def resolve(reference: str) -> str:
    """Accept a local path or Kyutai's hf://repo/path@revision form."""
    if not reference.startswith("hf://"):
        return reference
    from huggingface_hub import hf_hub_download

    body = reference[len("hf://") :]
    revision = None
    if "@" in body:
        body, revision = body.rsplit("@", 1)
    owner, name, path = body.split("/", 2)
    return hf_hub_download(f"{owner}/{name}", path, revision=revision)


def main() -> None:
    args = parse_args()
    import safetensors.torch
    import sentencepiece
    import torch

    state = safetensors.torch.load_file(resolve(args.weights))
    if EMBED_KEY not in state:
        raise SystemExit(f"{EMBED_KEY} not in the checkpoint; is this a Pocket TTS model?")

    layers = {
        int(m.group(1))
        for k in state
        if (m := re.search(r"flow_lm\.transformer\.layers\.(\d+)\.", k))
    }
    old_embed = state[EMBED_KEY]
    print(f"source: {len(layers)} transformer layers, text table {tuple(old_embed.shape)}")

    source = sentencepiece.SentencePieceProcessor(resolve(args.source_tokenizer))
    target = sentencepiece.SentencePieceProcessor(str(args.tokenizer.resolve()))
    # The table carries one extra row past the vocabulary (padding / no-text).
    assert old_embed.shape[0] == source.vocab_size() + 1, (
        f"table has {old_embed.shape[0]} rows but the source tokenizer has "
        f"{source.vocab_size()} pieces -- wrong --source-tokenizer for these weights"
    )

    new_embed = torch.empty(target.vocab_size() + 1, old_embed.shape[1], dtype=old_embed.dtype)
    new_embed.normal_(mean=float(old_embed.float().mean()), std=float(old_embed.float().std()))

    by_piece = {source.id_to_piece(i): i for i in range(source.vocab_size())}
    copied = 0
    for new_id in range(target.vocab_size()):
        old_id = by_piece.get(target.id_to_piece(new_id))
        if old_id is not None:
            new_embed[new_id] = old_embed[old_id]
            copied += 1
    new_embed[-1] = old_embed[-1]  # the trailing row keeps its role

    state[EMBED_KEY] = new_embed
    args.out.parent.mkdir(parents=True, exist_ok=True)
    safetensors.torch.save_file(state, str(args.out))

    print(
        f"transplanted {copied}/{target.vocab_size()} pieces "
        f"({100 * copied / target.vocab_size():.1f}%); the rest are freshly initialized"
    )
    print(f"wrote {args.out}")
    print(
        f"set weights_path to it, n_bins to {target.vocab_size()}, and start_from_pretrained: true"
    )


if __name__ == "__main__":
    main()
