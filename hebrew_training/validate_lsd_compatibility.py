from __future__ import annotations

import argparse
import gc
from pathlib import Path

import torch

from hebrew_training.lsd import AdaptiveLossWeight
from hebrew_training.model_utils import (
    choose_device,
    install_tokenizer,
    require_voice_cloning_access,
)
from hebrew_training.train import LatentDataset, compute_loss
from pocket_tts import TTSModel


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run one FM/LSD forward/backward pass on real Pocket TTS checkpoints."
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument(
        "--languages",
        nargs="+",
        default=["english", "french_24l"],
        help="A 6-layer and 24-layer pair is recommended.",
    )
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = choose_device(args.device)
    item = LatentDataset(args.manifest)[0]
    # This is an interface/gradient probe, not a quality measurement. Keeping only four
    # target frames makes the 24-layer CPU check quick while still producing a 3/1 split.
    item["target"] = item["target"][:4]

    for language in args.languages:
        model = TTSModel.load_model(language=language)
        require_voice_cloning_access(model)
        copied_pieces = install_tokenizer(model, args.tokenizer)
        model.mimi.requires_grad_(False)
        model.mimi.eval()
        model.to(device)
        model.flow_lm.train()
        adaptive_weight = AdaptiveLossWeight(128).to(device)

        loss, flow_loss, lsd_loss, eos_loss = compute_loss(
            model,
            item,
            head_samples=4,
            eos_weight=0.1,
            device=device,
            head_batch_multiplier=1,
            lsd_fraction=0.25,
            adaptive_loss_weight=adaptive_weight,
        )
        loss.backward()
        flow_gradients = [
            parameter.grad
            for parameter in model.flow_lm.flow_net.parameters()
            if parameter.requires_grad
        ]
        if not flow_gradients or any(gradient is None for gradient in flow_gradients):
            raise RuntimeError(f"{language}: missing flow-head gradients")
        if not all(torch.isfinite(gradient).all() for gradient in flow_gradients):
            raise RuntimeError(f"{language}: non-finite flow-head gradients")
        if (
            adaptive_weight.weight.grad is None
            or not torch.isfinite(adaptive_weight.weight.grad).all()
        ):
            raise RuntimeError(f"{language}: invalid adaptive-weight gradient")

        trainable = sum(
            parameter.numel() for parameter in model.flow_lm.parameters() if parameter.requires_grad
        )
        print(
            f"PASS language={language} trainable={trainable:,} copied_pieces={copied_pieces} "
            f"loss={float(loss.detach()):.5f} flow={float(flow_loss):.5f} "
            f"lsd={float(lsd_loss):.5f} eos={float(eos_loss):.5f}"
        )

        del model, adaptive_weight, flow_gradients
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
