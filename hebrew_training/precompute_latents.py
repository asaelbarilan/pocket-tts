from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import safetensors.torch
import torch

from hebrew_training.data import read_jsonl
from hebrew_training.model_utils import choose_device, require_voice_cloning_access
from pocket_tts import TTSModel
from pocket_tts.data.audio import audio_read
from pocket_tts.data.audio_utils import convert_audio


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Freeze Kyutai Mimi and cache compact target/prompt latents."
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--output-manifest", type=Path, required=True)
    parser.add_argument("--base-language", default="english")
    parser.add_argument("--prompt-seconds", type=float, default=3.0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--limit", type=int)
    parser.add_argument(
        "--allow-self-prompt",
        action="store_true",
        help=(
            "Fall back to slicing the prompt from the target clip when a row has no "
            "prompt_audio_path. This reintroduces same-clip leakage and exists only for "
            "reproducing the old hebrew-20k behaviour."
        ),
    )
    return parser.parse_args()


def load_audio(path: Path, model, device: torch.device) -> torch.Tensor:
    audio, sample_rate = audio_read(path)
    audio = convert_audio(audio, sample_rate, model.sample_rate, 1)
    return audio.unsqueeze(0).to(device)


def main() -> None:
    args = parse_args()
    device = choose_device(args.device)
    model = TTSModel.load_model(language=args.base_language)
    require_voice_cloning_access(model)
    model.to(device).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    rows = list(read_jsonl(args.manifest))
    if args.limit:
        rows = rows[: args.limit]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.output_manifest.parent.mkdir(parents=True, exist_ok=True)

    missing_prompt = sum(1 for row in rows if not row.get("prompt_audio_path"))
    if missing_prompt and not args.allow_self_prompt:
        raise RuntimeError(
            f"{missing_prompt} of {len(rows)} rows have no prompt_audio_path. "
            "Rebuild the manifest with hebrew_training.prepare_data_v2, or pass "
            "--allow-self-prompt to accept same-clip leakage."
        )
    self_prompted = 0

    completed = 0
    with args.output_manifest.open("w", encoding="utf-8", newline="\n") as output:
        for row in rows:
            # Key the cache on both clips, so a manifest change can never silently reuse
            # latents built from a different (or self-) prompt.
            cache_key = (
                f"{args.base_language}|{row['audio_path']}|{row.get('prompt_audio_path', '')}"
            )
            digest = hashlib.sha256(cache_key.encode("utf-8")).hexdigest()[:16]
            cache_path = args.output_dir / f"{digest}.safetensors"
            if not cache_path.exists():
                audio = load_audio(Path(row["audio_path"]), model, device)
                prompt_path = row.get("prompt_audio_path")
                if prompt_path:
                    prompt_audio = load_audio(Path(prompt_path), model, device)
                else:
                    # Only reachable with --allow-self-prompt.
                    prompt_audio = audio
                    self_prompted += 1
                prompt_samples = min(
                    prompt_audio.shape[-1], int(args.prompt_seconds * model.sample_rate)
                )
                with torch.inference_mode():
                    # Upstream 891886a made encode_to_latent return time-major [B, T, C].
                    # It used to return [B, C, T], so this code transposed. Keeping that
                    # transpose after the merge would silently emit [32, frames] latents
                    # where train.py expects [frames, 32]. Latents cached before the merge
                    # are correct and do not need recomputing.
                    raw_target = model.mimi.encode_to_latent(audio).float()
                    raw_prompt = model.mimi.encode_to_latent(
                        prompt_audio[..., :prompt_samples]
                    ).float()
                    normalized_target = (
                        raw_target - model.flow_lm.emb_mean
                    ) / model.flow_lm.emb_std.clamp_min(1e-6)
                safetensors.torch.save_file(
                    {
                        "target": normalized_target[0].cpu().to(torch.float16).contiguous(),
                        "prompt": raw_prompt[0].cpu().to(torch.float16).contiguous(),
                    },
                    cache_path,
                )
            cached_row = dict(row)
            cached_row["latent_path"] = str(cache_path.resolve())
            cached_row["latent_base_language"] = args.base_language
            output.write(json.dumps(cached_row, ensure_ascii=False) + "\n")
            completed += 1
            if completed % 100 == 0 or completed == len(rows):
                print(f"Cached {completed}/{len(rows)}")

    if self_prompted:
        print(f"WARNING: {self_prompted} rows used a same-clip prompt (leakage).")


if __name__ == "__main__":
    main()
