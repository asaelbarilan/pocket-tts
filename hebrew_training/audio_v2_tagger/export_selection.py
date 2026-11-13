"""Freeze the latest tagger fit and select audio-v2 rows at P(rating >= 3) >= cutoff.

Writes an immutable model/labels/settings snapshot, a compact row-level selection, and
fetch_audio_v2's episode-level selection.json. Run with the tagger stopped so labels cannot
change during the export. Never downloads audio or modifies existing ratings.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
from pathlib import Path

import numpy as np

from hebrew_training import audio_v2_table as table_mod
from hebrew_training.audio_v2_tagger import model


def replay_labels(path: Path, table: table_mod.Table) -> dict[str, dict]:
    """Apply undo records, checking every label still points at the same feature row."""
    ratings: dict[str, dict] = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            if "undo" in record:
                ratings.pop(record["undo"], None)
                continue
            row = record["row"]
            key = f"{table.episode_ids[table.episode[row]]}_{round(table.start[row] * 1000)}"
            if record["key"] != key:
                raise ValueError(f"label {record['key']} does not match table row {row}")
            ratings[record["key"]] = record
    return ratings


def export(work: Path, output: Path, batch_size: int = 200_000) -> dict:
    if output.exists():
        raise FileExistsError(f"refusing to overwrite frozen export: {output}")
    if (work / "selection.json").exists():
        raise FileExistsError("selection.json already exists; preserve it or explicitly move it")
    settings_path = work / "tagger" / "settings.json"
    labels_path = work / "tagger" / "labels.jsonl"
    settings = json.loads(settings_path.read_text(encoding="utf-8"))
    label_bytes = labels_path.read_bytes()
    table = table_mod.load_table(work)
    ratings = replay_labels(labels_path, table)
    chosen = [r for r in ratings.values() if r["rating"] > 0]
    if len(chosen) < model.MIN_LABELS:
        raise ValueError(f"only {len(chosen)} rated rows; cannot fit")
    rows = np.array([r["row"] for r in chosen], dtype=int)
    y = np.array([r["rating"] for r in chosen], dtype=int)
    use_source = settings["use_source"]
    cutoff = float(settings["cutoff"])
    booster = model.fit(
        model.frame(
            table.x[rows], table.source[rows], len(table.sources), table_mod.FEATURES, use_source
        ),
        y,
        seed=len(rows),
    )
    keep = np.zeros(len(table), dtype=bool)
    for first in range(0, len(table), batch_size):
        last = min(first + batch_size, len(table))
        frame = model.frame(
            table.x[first:last],
            table.source[first:last],
            len(table.sources),
            table_mod.FEATURES,
            use_source,
        )
        keep[first:last] = model.p_at_least(model.predict(booster, frame), 3) >= cutoff
        print(f"scored {last:,}/{len(table):,} rows", flush=True)
    selected = np.flatnonzero(keep)
    if not len(selected):
        raise ValueError("model selected zero rows")
    episodes, _ = table_mod.order(work)
    total_s = np.bincount(table.episode, weights=table.duration, minlength=len(episodes))
    kept_s = np.bincount(
        table.episode[selected], weights=table.duration[selected], minlength=len(episodes)
    )
    picks = [
        {**episode, "speech_s": round(float(total_s[i]), 2), "kept_s": round(float(kept_s[i]), 2)}
        for i, episode in enumerate(episodes)
        if kept_s[i] > 0
    ]
    if labels_path.read_bytes() != label_bytes:
        raise RuntimeError("ratings changed during export; stop the tagger and retry")
    output.mkdir(parents=True)
    shutil.copy2(labels_path, output / "labels.jsonl")
    shutil.copy2(settings_path, output / "settings.json")
    booster.save_model(str(output / "model.ubj"))
    np.savez_compressed(
        output / "selected_rows.npz",
        episode_ids=np.asarray(table.episode_ids),
        episode=table.episode[selected],
        start=np.round(table.start[selected], 4),
        end=np.round(table.end[selected], 4),
    )
    info = {
        "rule": "P(rating >= 3) >= cutoff",
        "cutoff": cutoff,
        "use_source": use_source,
        "seed": len(rows),
        "rating_counts": {str(k): int((y == k).sum()) for k in (1, 2, 3)},
        "ratings": len(rows),
        "label_sha256": hashlib.sha256(label_bytes).hexdigest(),
        "features": list(table_mod.FEATURES),
        "total_rows": len(table),
        "selected_rows": len(selected),
        "selected_hours": round(float(table.duration[selected].sum()) / 3600, 2),
        "selected_episodes": len(picks),
        "download_gib": round(sum(e["audio_bytes"] for e in picks) / 2**30, 2),
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
    }
    (output / "fit.json").write_text(json.dumps(info, indent=2) + "\n", encoding="utf-8")
    selection = {
        "params": {"model_selection": str(output / "fit.json"), "cutoff": cutoff},
        "episodes": picks,
    }
    (output / "selection.json").write_text(
        json.dumps(selection, ensure_ascii=False), encoding="utf-8"
    )
    tmp = work / "selection.json.tmp"
    tmp.write_text(json.dumps(selection, ensure_ascii=False), encoding="utf-8")
    tmp.replace(work / "selection.json")
    return info


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(export(args.work, args.output), indent=2), flush=True)


if __name__ == "__main__":
    main()
