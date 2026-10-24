"""MILIM-Bench (renikud/MILIM-Bench) sentences for the test page's sample picker.

The dataset is downloaded once, at image build time (see the Dockerfile), into `MILIM_DIR`;
the server only reads `data/gold.tsv` from it. Locally:

    uv run python -m deploy.milim_bench                  # into deploy/milim_bench_data/

Each row is `Category`, `Text`, `Label`; the label is `token_index=IPA` pairs, where tokens are
the whitespace-separated words of the text (as in the benchmark's own evaluate.py).
"""

from __future__ import annotations

import csv
import logging
import sys
from pathlib import Path

logger = logging.getLogger("hebrew_tts_server.milim")

DATASET = "renikud/MILIM-Bench"
GOLD = Path("data") / "gold.tsv"
DEFAULT_DIR = Path(__file__).resolve().parent / "milim_bench_data"
MAX_BATCH = 5


def download(directory: Path, revision: str | None = None) -> None:
    from huggingface_hub import snapshot_download

    snapshot_download(DATASET, repo_type="dataset", revision=revision, local_dir=directory)


def parse_label(label: str, n_words: int) -> list[dict]:
    """`3=ʔitˈaχ 5=x` -> [{"index": 3, "ipa": "ʔitˈaχ"}, ...]; drops pairs that do not fit."""
    targets = []
    for part in label.split():
        index, _, ipa = part.partition("=")
        if index.isdigit() and ipa and int(index) < n_words:
            targets.append({"index": int(index), "ipa": ipa})
    return targets


class MilimBench:
    """Read-only samples with a stable id (the row number in gold.tsv)."""

    def __init__(self, directory: Path):
        self.directory = directory
        self.samples: list[dict] = []
        self.categories: list[dict] = []
        gold = directory / GOLD
        if not gold.is_file():
            logger.warning("MILIM-Bench not found at %s; the sample picker is disabled", gold)
            return
        with gold.open("r", encoding="utf-8-sig", newline="") as handle:
            for number, row in enumerate(csv.DictReader(handle, delimiter="\t")):
                text = row["Text"].strip()
                self.samples.append(
                    {
                        "id": number,
                        "category": row["Category"],
                        "text": text,
                        "targets": parse_label(row["Label"], len(text.split())),
                    }
                )
        counts: dict[str, int] = {}
        for sample in self.samples:
            counts[sample["category"]] = counts.get(sample["category"], 0) + 1
        self.categories = [{"name": name, "count": count} for name, count in counts.items()]
        logger.info("MILIM-Bench: %d samples, %d categories", len(self.samples), len(counts))

    @property
    def available(self) -> bool:
        return bool(self.samples)

    def get(self, sample_id: int) -> dict | None:
        if 0 <= sample_id < len(self.samples):
            return self.samples[sample_id]
        return None

    def payload(self) -> dict:
        return {"categories": self.categories, "samples": self.samples, "max_batch": MAX_BATCH}


if __name__ == "__main__":
    target = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_DIR
    download(target)
    print(f"downloaded {DATASET} to {target}")
