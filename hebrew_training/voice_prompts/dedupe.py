"""Drop near-duplicate voice prompts from an extraction (the optional second run).

"Near-duplicate" is measured the way Pocket-TTS itself sees a voice prompt, not by speaker
identity alone. `TTSModel.get_state_for_audio_prompt` runs the prompt through Mimi's encoder
(`mimi.encode_to_latent`) and projects it with `flow_lm.speaker_proj_weight`; the transformer
then reads that sequence as a prefix. Whatever Mimi keeps of a clip -- timbre, recording channel
and room, loudness, pitch range, pace, delivery -- is therefore what the model can imitate, and
that is the diversity this pool is for.

Each prompt is embedded as the time-pooled statistics of that sequence: the mean and the standard
deviation over time of each channel (the mean is the static colour of the voice and channel, the
deviation the part that moves: prosody and expression). `--space` picks which sequence:

    latent          Mimi's encoder output (default). Mimi is frozen in every run here, so this
                    space is the same for the English base, a Hebrew checkpoint or a config.
    conditioning    the same after `speaker_proj_weight`: exactly what the transformer is fed,
                    but specific to the weights loaded (`--language`, `--config`, `--checkpoint`).

The pooled vectors are standardized per channel across the prompts of the folder (so channels with
a large scale do not decide everything) and compared by cosine similarity. These are not
speaker-verification scores, so there is no standard "same voice" threshold: the first run prints
the nearest-neighbour similarity quantiles of a sample, and you choose from them. Pooling also
keeps some phonetic content, which is why the prompts are 3-5 s of different sentences.

Two ways to thin the pool (pass one; with neither, only the report is printed):

    --threshold T   walk the prompts in manifest order and keep one only if its similarity to
                    every kept prompt is below T. The numbering is random, so which of a cluster
                    survives is not biased.
    --keep N        farthest-point selection: N prompts that cover the space as widely as
                    possible, whatever the density.

Non-destructive by default, next to `manifest.jsonl`:

    embeddings.npz            ids + pooled vectors + what produced them (reused by later runs)
    manifest.dedup.jsonl      the kept rows
    duplicates.jsonl          {id, duplicate_of, similarity} for every removed prompt

`--prune` makes the result final: it deletes the removed WAVs, keeps the untouched manifest as
`manifest.before_dedup.jsonl` and rewrites `manifest.jsonl` with the kept rows. Ids are never
renumbered.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from hebrew_training.data import read_jsonl
from hebrew_training.voice_prompts.extract import MANIFEST

EMBEDDINGS = "embeddings.npz"
SPACES = ("latent", "conditioning")
REPORT_SAMPLE = 2000


def load_prompt_model(args: argparse.Namespace):
    from pocket_tts import TTSModel

    return TTSModel.load_model(
        language=None if args.config else args.language,
        config=args.config,
        checkpoint=args.checkpoint,
    )


def source_label(args: argparse.Namespace) -> str:
    """What the cached vectors depend on: the space, and for `conditioning` the weights."""
    if args.space == "latent":
        return "latent"  # Mimi is frozen, so the weights loaded do not matter
    return f"conditioning|{args.language}|{args.config}|{args.checkpoint}"


def pooled_embedding(model, path: Path, space: str):
    """`[mean, std]` over time of the prompt sequence Pocket-TTS would encode for this WAV."""
    import torch

    from pocket_tts.data.audio import audio_read
    from pocket_tts.data.audio_utils import convert_audio

    audio, rate = audio_read(path)
    audio = convert_audio(audio, rate, model.config.mimi.sample_rate, 1)
    batch = audio.unsqueeze(0).to(model.device)
    with torch.no_grad():
        sequence = (
            model.mimi.encode_to_latent(batch).float()
            if space == "latent"
            else model._encode_audio(batch)
        )[0]
    return torch.cat([sequence.mean(dim=0), sequence.std(dim=0, unbiased=False)]).cpu().numpy()


def embed_missing(folder: Path, rows: list[dict], args: argparse.Namespace, cached: dict) -> dict:
    """Pooled vectors by id for every row, computing only what `cached` lacks."""
    todo = [row for row in rows if row["id"] not in cached]
    if not todo:
        return cached
    model = load_prompt_model(args)
    print(f"embedding {len(todo)} prompts ({args.space})", file=sys.stderr)
    for count, row in enumerate(todo, 1):
        cached[row["id"]] = pooled_embedding(model, folder / row["audio"], args.space)
        if count % 200 == 0:
            print(f"  {count}/{len(todo)}", file=sys.stderr, end="\r")
    print(file=sys.stderr)
    return cached


def load_cache(path: Path, label: str) -> dict:
    import numpy as np

    if not path.is_file():
        return {}
    with np.load(path) as data:
        if str(data["source"]) != label:
            return {}
        return dict(zip(data["ids"].tolist(), data["vectors"], strict=True))


def save_cache(path: Path, vectors: dict, label: str) -> None:
    import numpy as np

    ids = sorted(vectors)
    temporary = path.with_name(path.name + ".tmp.npz")
    np.savez(
        temporary, ids=np.array(ids), vectors=np.stack([vectors[i] for i in ids]), source=label
    )
    temporary.replace(path)


def fit_standardizer(pooled):
    """`(mean, scale)` per channel of a set of pooled vectors."""
    mean = pooled.mean(axis=0)
    scale = (pooled - mean).std(axis=0)
    return mean, scale


def standardize(pooled, fitted=None):
    """Unit vectors after a per-channel z-score: across `pooled` itself, or by `fitted` stats."""
    import numpy as np

    mean, scale = fitted if fitted is not None else fit_standardizer(pooled)
    unit = (pooled - mean) / np.where(scale > 1e-8, scale, 1.0)
    return (unit / np.maximum(np.linalg.norm(unit, axis=-1, keepdims=True), 1e-8)).astype("float32")


def nearest_neighbour_quantiles(matrix, sample: int = REPORT_SAMPLE, seed: int = 0):
    """Quantiles of each sampled prompt's highest similarity to any other prompt."""
    import numpy as np

    if len(matrix) < 2:
        return None
    chosen = np.random.default_rng(seed).choice(len(matrix), min(sample, len(matrix)), False)
    similarities = matrix[chosen] @ matrix.T
    similarities[np.arange(len(chosen)), chosen] = -np.inf
    return np.quantile(similarities.max(axis=1), [0.05, 0.25, 0.5, 0.75, 0.95])


def find_duplicates(ids: list[int], matrix, threshold: float) -> dict[int, tuple[int, float]]:
    """`{duplicate_id: (kept_id, similarity)}`, walking `ids` in order over unit vectors."""
    import numpy as np

    kept = np.empty_like(matrix)
    kept_ids: list[int] = []
    duplicates = {}
    for position, prompt in enumerate(ids):
        if kept_ids:
            similarities = kept[: len(kept_ids)] @ matrix[position]
            best = int(similarities.argmax())
            if similarities[best] >= threshold:
                duplicates[prompt] = (kept_ids[best], float(similarities[best]))
                continue
        kept[len(kept_ids)] = matrix[position]
        kept_ids.append(prompt)
    return duplicates


def select_diverse(ids: list[int], matrix, keep: int) -> dict[int, tuple[int, float]]:
    """Farthest-point selection of `keep` prompts; the rest map to their nearest selected one."""
    import numpy as np

    if keep >= len(ids):
        return {}
    nearest = np.zeros(len(ids), dtype=int)
    closest = matrix @ matrix[0]  # similarity to the nearest selected prompt so far
    selected = [0]
    while len(selected) < keep:
        pick = int(closest.argmin())
        selected.append(pick)
        similarities = matrix @ matrix[pick]
        better = similarities > closest
        closest = np.where(better, similarities, closest)
        nearest = np.where(better, pick, nearest)
    chosen = set(selected)
    return {
        ids[i]: (ids[int(nearest[i])], float(closest[i]))
        for i in range(len(ids))
        if i not in chosen
    }


def write_rows(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("folder", help="an extraction folder (prompts/ and manifest.jsonl)")
    parser.add_argument("--space", choices=SPACES, default="latent", help="see the module doc")
    parser.add_argument("--language", default="english", help="model to load for the embedding")
    parser.add_argument("--config", help="custom model config, instead of --language")
    parser.add_argument("--checkpoint", help="training checkpoint (.pt) to load")
    parser.add_argument("--threshold", type=float, help="cosine; >= is a duplicate")
    parser.add_argument("--keep", type=int, help="keep this many, as spread out as possible")
    parser.add_argument(
        "--prune", action="store_true", help="delete removed WAVs, rewrite manifest"
    )


def run(args: argparse.Namespace) -> None:
    if args.threshold is not None and args.keep is not None:
        raise SystemExit("pass --threshold or --keep, not both")
    folder = Path(args.folder)
    manifest = folder / MANIFEST
    if not manifest.is_file():
        raise SystemExit(f"{manifest} not found")
    rows = list(read_jsonl(manifest))
    if not rows:
        raise SystemExit(f"{manifest} is empty")

    cache_path, label = folder / EMBEDDINGS, source_label(args)
    vectors = load_cache(cache_path, label)
    before = len(vectors)
    vectors = embed_missing(folder, rows, args, vectors)
    if len(vectors) != before:
        save_cache(cache_path, vectors, label)

    import numpy as np

    ids = [row["id"] for row in rows]
    matrix = standardize(np.stack([vectors[i] for i in ids]))
    quantiles = nearest_neighbour_quantiles(matrix)
    if quantiles is not None:
        print(
            "nearest-neighbour similarity (p5/p25/p50/p75/p95): "
            + " / ".join(f"{q:.3f}" for q in quantiles)
        )
    if args.threshold is None and args.keep is None:
        print("pass --threshold T or --keep N to remove prompts")
        return

    duplicates = (
        find_duplicates(ids, matrix, args.threshold)
        if args.threshold is not None
        else select_diverse(ids, matrix, args.keep)
    )
    kept = [row for row in rows if row["id"] not in duplicates]
    write_rows(folder / "manifest.dedup.jsonl", kept)
    write_rows(
        folder / "duplicates.jsonl",
        [
            {"id": i, "duplicate_of": other, "similarity": round(similarity, 4)}
            for i, (other, similarity) in sorted(duplicates.items())
        ],
    )
    print(f"{len(rows)} prompts, {len(kept)} kept, {len(duplicates)} removed")

    if args.prune:
        for row in rows:
            if row["id"] in duplicates:
                (folder / row["audio"]).unlink(missing_ok=True)
        manifest.replace(folder / "manifest.before_dedup.jsonl")
        (folder / "manifest.dedup.jsonl").replace(manifest)
        print(f"pruned: {manifest} now lists the {len(kept)} kept prompts")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    add_arguments(parser)
    run(parser.parse_args(argv))


if __name__ == "__main__":
    main()
