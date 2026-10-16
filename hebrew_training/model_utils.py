from __future__ import annotations

from pathlib import Path

import torch
from torch import nn

from pocket_tts.conditioners.text import SentencePieceTokenizer


def tokenizer_vocab_size(tokenizer_path: Path) -> int:
    """The tokenizer's own size. Never a literal: n_bins has to equal it, and a hardcoded
    4000 silently became wrong the day the vocabulary was made a parameter."""
    import sentencepiece

    return sentencepiece.SentencePieceProcessor(model_file=str(tokenizer_path)).vocab_size()


def require_voice_cloning_access(model) -> None:
    if model.has_voice_cloning:
        return
    raise RuntimeError(
        "Kyutai's gated voice-cloning weights are not available. Accept the terms at "
        "https://huggingface.co/kyutai/pocket-tts, run `hf auth login`, and retry. "
        "The non-cloning fallback must not be adapted from user voice prompts or "
        "re-exported as a voice-cloning model."
    )


def install_tokenizer(model, tokenizer_path: Path) -> int:
    """Install a Hebrew tokenizer and retain embeddings for pieces shared with the base model."""
    conditioner = model.flow_lm.conditioner
    old_tokenizer = conditioner.tokenizer
    vocab_size = tokenizer_vocab_size(tokenizer_path)
    new_tokenizer = SentencePieceTokenizer(vocab_size, str(tokenizer_path.resolve()))
    old_weight = conditioner.embed.weight.detach()

    new_embed = nn.Embedding(
        vocab_size + 1, conditioner.dim, device=old_weight.device, dtype=old_weight.dtype
    )
    with torch.no_grad():
        new_embed.weight.normal_(mean=float(old_weight.mean()), std=float(old_weight.std()))
        old_piece_to_id = {
            old_tokenizer.sp.id_to_piece(index): index
            for index in range(old_tokenizer.sp.vocab_size())
        }
        copied = 0
        for new_id in range(new_tokenizer.sp.vocab_size()):
            old_id = old_piece_to_id.get(new_tokenizer.sp.id_to_piece(new_id))
            if old_id is not None:
                new_embed.weight[new_id].copy_(old_weight[old_id])
                copied += 1
        new_embed.weight[vocab_size].copy_(old_weight[-1])

    conditioner.tokenizer = new_tokenizer
    conditioner.embed = new_embed
    return copied


def choose_device(requested: str) -> torch.device:
    if requested == "auto":
        requested = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but this PyTorch build cannot access CUDA")
    return device
