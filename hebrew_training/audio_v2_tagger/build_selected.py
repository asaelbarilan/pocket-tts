"""Resumable audio-v2 model-3 export, download and manifest build.

Run detached on the data host. The fit snapshot is created only once and stays unchanged
on resume; labels/settings remain in the tagger work directory for future labeling.
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import subprocess
import sys
from pathlib import Path

from hebrew_training.audio_v2_tagger.export_selection import export


def stage(name: str, path: Path) -> None:
    path.write_text(name + "\n", encoding="utf-8")
    print(f"STAGE: {name}", flush=True)


def run(args: argparse.Namespace) -> None:
    work, snapshot, corpus, manifest = args.work, args.snapshot, args.corpus, args.manifest
    status = snapshot.parent / (snapshot.name + ".status")
    done = snapshot.parent / (snapshot.name + ".done.json")
    if done.exists():
        print(f"already complete: {done}", flush=True)
        return
    if not (snapshot / "fit.json").exists():
        stage("export", status)
        info = export(work, snapshot)
    else:
        info = json.loads((snapshot / "fit.json").read_text(encoding="utf-8"))
        if not (work / "selection.json").exists():
            shutil.copy2(snapshot / "selection.json", work / "selection.json")
    print("frozen fit:", json.dumps(info), flush=True)
    selection = json.loads((work / "selection.json").read_text(encoding="utf-8"))
    if selection["params"]["model_selection"] != str(snapshot / "fit.json"):
        raise ValueError("selection.json does not belong to frozen fit")
    episodes = selection["episodes"]
    stage("fetch", status)
    for attempt in range(1, 4):
        subprocess.run(
            [
                sys.executable,
                "-m",
                "hebrew_training.fetch_audio_v2",
                "--work",
                str(work),
                "fetch",
                "--out",
                str(corpus),
                "--max-gb",
                str(math.ceil(info["download_gib"]) + 1),
                "--workers",
                str(args.fetch_workers),
            ],
            check=True,
        )
        missing = [
            e["id"]
            for e in episodes
            if not (corpus / e["id"] / "metadata.json").exists()
            or not (corpus / e["id"] / f"audio{e['ext']}").is_file()
            or not (corpus / e["id"] / "transcript.aligned.json").is_file()
        ]
        print(
            f"fetch attempt {attempt}: {len(episodes) - len(missing):,}/{len(episodes):,} complete",
            flush=True,
        )
        if not missing:
            break
    if missing:
        (snapshot / "missing_episodes.json").write_text(json.dumps(missing) + "\n")
        raise RuntimeError(f"{len(missing)} episode downloads incomplete; rerun to resume")
    stage("manifest", status)
    subprocess.run(
        [
            sys.executable,
            "-m",
            "hebrew_training.build_official_manifest",
            "--corpus",
            str(corpus),
            "--out-dir",
            str(manifest),
            "--audio-glob",
            "audio.m4a,audio.mp3",
            "--row-selection",
            str(snapshot / "selected_rows.npz"),
            "--spool-dir",
            str(args.spool_dir),
            "--workers",
            str(args.build_workers),
        ],
        check=True,
    )
    counts = {}
    for split in ("train", "valid"):
        path = manifest / f"{split}.jsonl"
        counts[split] = sum(1 for _ in path.open(encoding="utf-8"))
        if not counts[split]:
            raise RuntimeError(f"empty {path}; build is not complete")
    result = {
        **info,
        "manifest_rows": counts,
        "corpus": str(corpus),
        "manifest": str(manifest),
        "snapshot": str(snapshot),
    }
    done.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    stage("done", status)
    print("DATASET COMPLETE:", json.dumps(result), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--spool-dir", type=Path, required=True)
    parser.add_argument("--fetch-workers", type=int, default=8)
    parser.add_argument("--build-workers", type=int, default=4)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
