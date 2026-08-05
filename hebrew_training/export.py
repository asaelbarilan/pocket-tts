from __future__ import annotations

import argparse
from pathlib import Path

import safetensors.torch
import yaml

from hebrew_training.model_utils import install_tokenizer, require_voice_cloning_access
from pocket_tts import TTSModel
from pocket_tts.utils.config import CONFIGS_DIR


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export a Hebrew checkpoint for official inference.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--base-language", default="english")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    tokenizer_out = args.output_dir / "tokenizer.model"
    tokenizer_out.write_bytes(args.tokenizer.read_bytes())

    model = TTSModel.load_model(language=args.base_language)
    require_voice_cloning_access(model)
    install_tokenizer(model, tokenizer_out)
    flow_state = safetensors.torch.load_file(args.checkpoint / "flow_lm.safetensors")
    model.flow_lm.load_state_dict(flow_state, strict=True)
    weights_out = args.output_dir / "model.safetensors"
    safetensors.torch.save_file(
        {
            key: value.detach().cpu().contiguous()
            for key, value in model.state_dict().items()
        },
        weights_out,
    )

    base_config = yaml.safe_load(
        (CONFIGS_DIR / f"{args.base_language}.yaml").read_text(encoding="utf-8")
    )
    base_config["weights_path"] = str(weights_out.resolve())
    base_config["weights_path_without_voice_cloning"] = str(weights_out.resolve())
    base_config["flow_lm"]["lookup_table"]["tokenizer_path"] = str(tokenizer_out.resolve())
    base_config["flow_lm"]["lookup_table"]["n_bins"] = 4000
    config_out = args.output_dir / "hebrew.yaml"
    config_out.write_text(
        yaml.safe_dump(base_config, sort_keys=False, allow_unicode=True), encoding="utf-8"
    )
    print(f"Config: {config_out}")
    print(f"Weights: {weights_out}")
    print(f"Tokenizer: {tokenizer_out}")


if __name__ == "__main__":
    main()
