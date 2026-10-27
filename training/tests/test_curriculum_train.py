"""The curriculum wired into training.train.main(), with a tiny model and fake loaders.

Everything heavy (model build, audio, Mimi, DDP) is replaced; what runs for real is the
training loop itself: which dataset each micro-batch comes from, when validation and
checkpoints fire, what validate() does with the plan, and how a resume continues.
"""

import itertools
import json
from collections import Counter
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import pytest
import torch
import yaml
from torch import nn

from training.args import load_args
from training.train_utils import ProgressLog

# 3 + 4 + 3 steps; phase 1 has no ds1, phase 3 no ds2 (the shape of the original sketch).
DATASETS = {
    "ds1": {"train_jsonl": "a.jsonl", "valid_jsonl": "av.jsonl", "valid_weight": 0.2},
    "ds2": {"train_jsonl": "b.jsonl", "valid_jsonl": "bv.jsonl", "valid_weight": 0.4},
    "ds3": {"train_jsonl": "c.jsonl", "valid_jsonl": "cv.jsonl", "valid_weight": 0.4},
}
PHASES = [
    {"steps": 3, "mix": {"ds2": 50, "ds3": 50}},
    {"steps": 4, "mix": {"ds1": 20, "ds2": 40, "ds3": 40}},
    {"steps": 3, "mix": {"ds1": 60, "ds3": 40}},
]
BATCH, ACCUM = 2, 5  # 5 micro-batches a step: 20% is exactly one


class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(1))
        encode = SimpleNamespace(encode=lambda text: [1])
        self.flow_lm = SimpleNamespace(
            conditioner=SimpleNamespace(tokenizer=SimpleNamespace(sp=encode))
        )

    def forward(self, latents, mask, tokens, voice, **_):
        loss = self.weight.sum() * 0.0 + 1.0
        return loss, {"loss": loss.detach()}


class FakeLoader:
    """What make_loader returns: an endless stream of batches that remember their manifest."""

    def __init__(self, jsonl, rows=40):
        self.jsonl = jsonl
        self.entries = [None] * rows
        self.closed = False

    def __iter__(self):
        try:
            for _ in itertools.count():
                yield SimpleNamespace(jsonl=self.jsonl, text_tokens=[])
        finally:
            self.closed = True


class Harness:
    def __init__(self, tmp_path, data, *, max_steps=None, **top):
        raw = {"model_config": "x.yaml", "data": data, "run_dir": str(tmp_path / "run")}
        raw.update(batch_size=BATCH, grad_accum_steps=ACCUM, num_valid_batches=10)
        raw.update(compile=False, ema_decay=0, sample_sentences=[], log_freq=1, valid_freq=1000)
        raw.update(ckpt_freq=1000, **top)
        if max_steps is not None:
            raw["max_steps"] = max_steps
        tmp_path.mkdir(parents=True, exist_ok=True)
        path = tmp_path / "run.yaml"
        path.write_text(yaml.safe_dump(raw))
        self.args = load_args(path)
        self.tmp_path = tmp_path
        self.drawn: list[str] = []  # manifest of every micro-batch, in order
        self.valid_calls: list[tuple[str, int]] = []
        self.checkpoints: list[int] = []
        self.loaders: list[FakeLoader] = []
        self.progress_path = tmp_path / "progress.jsonl"

    def write_valid_manifests(self, rows):
        for spec in self.args.data.datasets.values():
            (self.tmp_path / spec["valid_jsonl"]).write_text("{}\n" * rows)
            spec["valid_jsonl"] = str(self.tmp_path / spec["valid_jsonl"])

    def run(self, start_step=0):
        from training import train

        model = TinyModel()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        fake_run = train.Run(
            args=self.args,
            model=model,
            wrapped=model,
            mimi=SimpleNamespace(sample_rate=24000, frame_rate=12.5),
            optimizer=optimizer,
            ema=None,
            start_step=start_step,
            device=torch.device("cpu"),
            rank=0,
            world_size=1,
            progress=ProgressLog(self.progress_path),
        )
        batch = torch.zeros(BATCH, 3, 4)

        def encode_batch(mimi, b, device):
            self.drawn.append(b.jsonl)
            return batch, torch.ones(BATCH, 3, dtype=torch.bool), batch, torch.ones(BATCH)

        def valid_pass(model, mimi, args, device, rank, world_size, jsonl, num_batches):
            self.valid_calls.append((jsonl, num_batches))
            return {"loss": {"av.jsonl": 1.0, "bv.jsonl": 2.0}.get(Path(jsonl).name, 4.0)}

        def make_loader(jsonl, *a, **k):
            loader = FakeLoader(jsonl)
            self.loaders.append(loader)
            return loader

        patches = {
            "setup": lambda config_path: fake_run,
            "make_loader": make_loader,
            "encode_batch": encode_batch,
            "_valid_pass": valid_pass,
            "save_checkpoint": lambda run_dir, step, *a: self.checkpoints.append(step),
            "wandb": mock.MagicMock(),
        }
        with mock.patch.multiple(train, **patches):
            train.main("ignored.yaml")
        return self

    def progress(self, event):
        lines = [json.loads(line) for line in self.progress_path.read_text().splitlines()]
        return [r for r in lines if r["type"] == event]

    def per_step(self):
        return [self.drawn[i : i + ACCUM] for i in range(0, len(self.drawn), ACCUM)]


@pytest.fixture
def harness(tmp_path):
    h = Harness(tmp_path, {"datasets": DATASETS, "curriculum": PHASES})
    h.write_valid_manifests(rows=400)
    return h


def test_every_phase_draws_its_mix(harness):
    harness.run()
    steps = harness.per_step()
    assert len(steps) == 10
    start = 0
    for end, phase in zip((3, 7, 10), PHASES, strict=True):
        counts = Counter(itertools.chain.from_iterable(steps[start:end]))
        total = (end - start) * ACCUM
        weights = {name: w / sum(phase["mix"].values()) for name, w in phase["mix"].items()}
        files = {"ds1": "a.jsonl", "ds2": "b.jsonl", "ds3": "c.jsonl"}
        assert set(counts) == {files[n] for n in weights}  # a 0% dataset is never drawn
        for name, share in weights.items():
            assert abs(counts[files[name]] - share * total) <= 1
        start = end


def test_phase_changes_are_logged(harness):
    harness.run()
    phases = harness.progress("phase")
    assert [(r["step"], r["phase"]) for r in phases] == [(1, 1), (4, 2), (8, 3)]
    assert phases[1]["mix"] == pytest.approx({"ds1": 0.2, "ds2": 0.4, "ds3": 0.4})
    # The train records carry the phase and per-dataset passes over the data.
    last = harness.progress("train")[-1]
    assert last["phase"] == 3
    assert set(last["epochs"]) == {"ds1", "ds2", "ds3"}


def test_validation_runs_at_every_phase_end_on_the_fixed_plan(harness):
    harness.run()
    # valid_freq is 1000, so these are all phase ends: steps 3, 7 and 10.
    assert [r["step"] for r in harness.progress("valid")] == [3, 7, 10]
    one_pass = [(c[0].rsplit("/", 1)[-1], c[1]) for c in harness.valid_calls[:3]]
    assert one_pass == [("av.jsonl", 2), ("bv.jsonl", 4), ("cv.jsonl", 4)]  # 0.2 / 0.4 / 0.4 of 10
    assert len(harness.valid_calls) == 9
    # Same benchmark at the end of every phase, whatever the phase trained on.
    assert harness.valid_calls[:3] == harness.valid_calls[3:6] == harness.valid_calls[6:]


def test_validation_reports_the_benchmark_and_each_dataset(harness):
    harness.run()
    metrics = harness.progress("valid")[0]["metrics"]
    # 2 batches of loss 1, 4 of loss 2, 4 of loss 4.
    assert metrics["loss"] == pytest.approx((2 * 1 + 4 * 2 + 4 * 4) / 10)
    assert metrics["ds1/loss"] == 1.0 and metrics["ds2/loss"] == 2.0 and metrics["ds3/loss"] == 4.0


def test_checkpoints_at_phase_ends_and_once_at_the_end(harness):
    harness.run()
    assert harness.checkpoints == [3, 7, 10]


def test_phase_end_checkpoints_can_be_turned_off(tmp_path):
    data = {"datasets": DATASETS, "curriculum": PHASES, "curriculum_checkpoint_at_phase_end": False}
    h = Harness(tmp_path, data)
    h.write_valid_manifests(rows=400)
    h.run()
    assert h.checkpoints == [10]
    assert [r["step"] for r in h.progress("valid")] == [3, 7, 10]  # evaluation is unaffected


def test_a_small_validation_set_gives_its_share_to_the_others(tmp_path):
    h = Harness(tmp_path, {"datasets": DATASETS, "curriculum": PHASES})
    h.write_valid_manifests(rows=400)
    # ds2 can supply 3 full batches (6 rows / batch 2), not the 4 it is owed.
    (tmp_path / "bv.jsonl").write_text("{}\n" * 6)
    h.run()
    plan = {c[0].rsplit("/", 1)[-1]: c[1] for c in h.valid_calls[:3]}
    assert plan == {"av.jsonl": 2, "bv.jsonl": 3, "cv.jsonl": 5}  # ds3 takes ds2's spare batch


def test_resuming_mid_phase_continues_the_mix(tmp_path):
    full = Harness(tmp_path / "full", {"datasets": DATASETS, "curriculum": PHASES})
    resumed = Harness(tmp_path / "resumed", {"datasets": DATASETS, "curriculum": PHASES})
    for h in (full, resumed):
        h.write_valid_manifests(rows=400)
    full.run()
    resumed.run(start_step=5)
    assert resumed.per_step() == full.per_step()[5:]
    assert [r["step"] for r in resumed.progress("valid")] == [7, 10]
    assert resumed.checkpoints == [7, 10]


def test_resuming_exactly_at_a_phase_boundary_starts_the_next_phase(tmp_path):
    h = Harness(tmp_path, {"datasets": DATASETS, "curriculum": PHASES})
    h.write_valid_manifests(rows=400)
    h.run(start_step=3)
    assert [(r["step"], r["phase"]) for r in h.progress("phase")] == [(4, 2), (8, 3)]


def test_a_plain_config_trains_and_validates_as_before(tmp_path):
    h = Harness(
        tmp_path, {"train_jsonl": "t.jsonl", "valid_jsonl": "v.jsonl"}, max_steps=4, valid_freq=2
    )
    h.run()
    assert set(h.drawn) == {"t.jsonl"} and len(h.drawn) == 4 * ACCUM
    assert h.valid_calls == [("v.jsonl", 10), ("v.jsonl", 10)]  # steps 2 and 4
    assert [r["step"] for r in h.progress("valid")] == [2, 4]
    assert h.progress("phase") == []
    assert "phase" not in h.progress("train")[0]


@pytest.mark.parametrize("mixed", [False, True])
@pytest.mark.parametrize("interrupted", [False, True])
def test_training_closes_loaders_before_distributed_shutdown(tmp_path, mixed, interrupted):
    data = (
        {"datasets": DATASETS, "curriculum": PHASES}
        if mixed
        else {"train_jsonl": "t.jsonl", "valid_jsonl": "v.jsonl"}
    )
    h = Harness(tmp_path, data, **({} if mixed else {"max_steps": 4}))
    if mixed:
        h.write_valid_manifests(rows=400)

    def shutdown():
        assert h.loaders and all(loader.closed for loader in h.loaders)

    with (
        mock.patch("training.train.any_across_ranks", return_value=interrupted),
        mock.patch("training.train.shutdown_distributed", side_effect=shutdown) as stopped,
    ):
        h.run()
    stopped.assert_called_once()
