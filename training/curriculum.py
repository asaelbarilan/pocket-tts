"""Training curriculum: several datasets, mixed in ratios that change over the run.

Config (under `data:` in the training yaml, instead of train_jsonl / valid_jsonl):

    datasets:
      knesset: {train_jsonl: ..., valid_jsonl: ..., valid_weight: 0.5}
      en:      {train_jsonl: ..., valid_jsonl: ..., valid_weight: 0.5}
    curriculum:                      # phases run in order; `steps` are optimizer steps
      - {steps: 3000, mix: {knesset: 1}}
      - {steps: 4000, mix: {knesset: 0.8, en: 0.2}}

The mix is a ratio of *samples* (relative weights, normalized), constant within a phase.
Each optimizer step draws `grad_accum_steps * world_size` micro-batches of `batch_size`
samples; every micro-batch comes from one dataset, picked by a deterministic quota so the
ratio is exact over a phase instead of noisy (a plain random draw of 8 micro-batches at
20% is anything from 0 to 5). A dataset that runs dry rolls over (the loader reshuffles
and starts again) and keeps its place across phases.

Validation does not follow the phases. It is a fixed benchmark: `valid_weight` splits the
`num_valid_batches` budget between datasets, and a dataset whose validation set is too
small to fill its share hands the rest to the others, in proportion to their weights.

Pure python on purpose (no torch, no I/O besides `count_lines`): the schedule must be the
same on every rank without any communication, and easy to unit test.
"""

import logging
import math
import re
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

NAME_RE = re.compile(r"[A-Za-z0-9_.-]+")
DATASET_KEYS = {"train_jsonl", "valid_jsonl", "valid_weight"}
PHASE_KEYS = {"steps", "mix"}
# Weights are quantized to this many parts so the quota arithmetic is exact (integers).
QUOTA_RESOLUTION = 1_000_000


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    train_jsonl: str
    valid_jsonl: str
    valid_weight: float


@dataclass(frozen=True)
class Phase:
    steps: int
    mix: dict[str, float]  # normalized to sum 1; only datasets with a positive weight


@dataclass(frozen=True)
class Curriculum:
    datasets: dict[str, DatasetSpec]
    phases: tuple[Phase, ...]
    checkpoint_at_phase_end: bool = True

    @property
    def total_steps(self) -> int:
        return sum(p.steps for p in self.phases)

    @property
    def boundaries(self) -> tuple[int, ...]:
        """Step count at the end of each phase: (S1, S2, S3) in a three-phase run."""
        ends, total = [], 0
        for phase in self.phases:
            total += phase.steps
            ends.append(total)
        return tuple(ends)

    def phase_index(self, step: int) -> int:
        """The phase that optimizer step `step` (0-based) belongs to; clamped to the last."""
        for index, end in enumerate(self.boundaries):
            if step < end:
                return index
        return len(self.phases) - 1

    def phase_start(self, index: int) -> int:
        return self.boundaries[index] - self.phases[index].steps

    @property
    def train_names(self) -> list[str]:
        """Datasets some phase draws from, in the order they were declared."""
        used = {name for phase in self.phases for name in phase.mix}
        return [name for name in self.datasets if name in used]

    @property
    def valid_names(self) -> list[str]:
        return [name for name, spec in self.datasets.items() if spec.valid_weight > 0]

    def describe(self) -> str:
        lines, start = [], 0
        for index, phase in enumerate(self.phases):
            mix = ", ".join(f"{name} {share:.1%}" for name, share in phase.mix.items())
            lines.append(f"  phase {index + 1}: steps {start + 1}-{start + phase.steps}  {mix}")
            start += phase.steps
        return "\n".join(lines)


def _number(value: Any, where: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{where} must be a number, got {value!r}")  # noqa: TRY004 - a bad config value, ValueError like the rest
    if not math.isfinite(value) or value < 0:
        raise ValueError(f"{where} must be a finite number >= 0, got {value!r}")
    return float(value)


def parse_curriculum(
    datasets: dict[str, Any], phases: list[Any], checkpoint_at_phase_end: bool = True
) -> Curriculum:
    """Validate the raw yaml values and build a Curriculum. Raises ValueError with the reason."""
    if not isinstance(datasets, dict) or not datasets:
        raise ValueError("data.datasets must be a non-empty mapping of name -> dataset")
    if not isinstance(phases, list) or not phases:
        raise ValueError("data.curriculum must be a non-empty list of phases")

    parsed_phases = []
    for index, raw in enumerate(phases, start=1):
        where = f"data.curriculum[{index}]"
        if not isinstance(raw, dict):
            raise ValueError(f"{where} must be a mapping with `steps` and `mix`")  # noqa: TRY004 - a bad config value, ValueError like the rest
        unknown = set(raw) - PHASE_KEYS
        if unknown:
            raise ValueError(
                f"{where}: unknown keys {sorted(unknown)}; expected {sorted(PHASE_KEYS)}"
            )
        missing = PHASE_KEYS - set(raw)
        if missing:
            raise ValueError(f"{where}: missing {sorted(missing)}")
        steps = raw["steps"]
        if isinstance(steps, bool) or not isinstance(steps, int) or steps < 1:
            raise ValueError(f"{where}.steps must be a positive integer, got {steps!r}")
        mix = raw["mix"]
        if not isinstance(mix, dict) or not mix:
            raise ValueError(f"{where}.mix must be a non-empty mapping of dataset -> weight")
        weights = {}
        for name, weight in mix.items():
            if name not in datasets:
                raise ValueError(f"{where}.mix names {name!r}, which is not in data.datasets")
            weights[name] = _number(weight, f"{where}.mix.{name}")
        total = sum(weights.values())
        if total <= 0:
            raise ValueError(f"{where}.mix has no positive weight")
        # Declared order, so ties in the quota are broken the same way everywhere.
        ordered = {n: weights[n] / total for n in datasets if weights.get(n, 0) > 0}
        parsed_phases.append(Phase(steps, ordered))

    used = {name for phase in parsed_phases for name in phase.mix}
    specs = {}
    for name, raw in datasets.items():
        where = f"data.datasets.{name}"
        if not isinstance(name, str) or not NAME_RE.fullmatch(name):
            raise ValueError(f"dataset name {name!r} must match {NAME_RE.pattern}")
        if not isinstance(raw, dict):
            raise ValueError(f"{where} must be a mapping")  # noqa: TRY004 - a bad config value, ValueError like the rest
        unknown = set(raw) - DATASET_KEYS
        if unknown:
            raise ValueError(
                f"{where}: unknown keys {sorted(unknown)}; expected {sorted(DATASET_KEYS)}"
            )
        if "valid_weight" not in raw:
            raise ValueError(
                f"{where}.valid_weight is required (0 keeps the dataset out of validation): "
                "it is the dataset's share of the fixed validation benchmark"
            )
        valid_weight = _number(raw["valid_weight"], f"{where}.valid_weight")
        train_jsonl, valid_jsonl = raw.get("train_jsonl", ""), raw.get("valid_jsonl", "")
        if name in used and not train_jsonl:
            raise ValueError(f"{where}.train_jsonl is required: the curriculum trains on it")
        if valid_weight > 0 and not valid_jsonl:
            raise ValueError(f"{where}.valid_jsonl is required: valid_weight is {valid_weight}")
        if name not in used and valid_weight == 0:
            raise ValueError(f"{where} is never trained on and has valid_weight 0: remove it")
        specs[name] = DatasetSpec(name, str(train_jsonl), str(valid_jsonl), valid_weight)
    return Curriculum(specs, tuple(parsed_phases), checkpoint_at_phase_end)


def apportion(
    weights: dict[str, float], total: int, caps: dict[str, int] | None = None, minimum: int = 0
) -> dict[str, int]:
    """Split `total` units in proportion to `weights`: integers summing to `total`.

    - A dataset whose proportional share exceeds its cap gets the cap, and the units it
      cannot take go to the others in proportion to their weights (repeated, since that can
      push another one past its cap).
    - Largest-remainder rounding, ties to the earlier dataset, so the result is stable.
    - Every dataset with a positive weight gets at least `minimum` (taken from the largest
      shares), and ValueError says so if that cannot be met.
    - If the caps cannot hold `total` the result sums to less; the caller decides if that
      matters.
    Datasets with weight 0 get 0.
    """
    caps = caps or {}
    result = {name: 0 for name in weights}
    active = [n for n, w in weights.items() if w > 0]
    if not active:
        return result
    cap = {n: caps.get(n, math.inf) for n in active}
    if minimum * len(active) > total:
        raise ValueError(f"{total} units cannot give each of {len(active)} datasets {minimum}")
    for name in active:
        if cap[name] < minimum:
            raise ValueError(f"{name!r} holds {cap[name]} units but needs at least {minimum}")

    remaining = total
    free = list(active)
    while free:
        weight_sum = sum(weights[n] for n in free)
        full = [n for n in free if remaining * weights[n] / weight_sum >= cap[n]]
        if not full:
            break
        for name in full:
            result[name] = int(cap[name])
            remaining -= int(cap[name])
            free.remove(name)
    if free:
        weight_sum = sum(weights[n] for n in free)
        shares = {n: remaining * weights[n] / weight_sum for n in free}
        for name in free:
            result[name] = math.floor(shares[name])
        leftover = remaining - sum(result[n] for n in free)
        by_remainder = sorted(free, key=lambda n: (-(shares[n] - result[n]), free.index(n)))
        for name in by_remainder[:leftover]:
            result[name] += 1

    for name in active:
        while result[name] < minimum:
            donors = [d for d in active if d != name and result[d] > minimum]
            if not donors:
                raise ValueError(f"cannot give {name!r} its minimum of {minimum}")
            donor = max(donors, key=lambda d: result[d])
            result[donor] -= 1
            result[name] += 1
    return result


def plan_validation(
    curriculum: Curriculum,
    line_counts: dict[str, int],
    batch_size: int,
    world_size: int,
    num_valid_batches: int,
) -> dict[str, int]:
    """Validation batches per rank for each dataset (0 for datasets outside the benchmark).

    The unit is a batch, because that is what the loader produces. A dataset can supply
    `lines // world_size // batch_size` batches per rank without wrapping around and
    repeating rows; the smallest rank shard is used so every rank gets the same plan.
    """
    names = curriculum.valid_names
    if not names:
        return {}
    if num_valid_batches < len(names):
        raise ValueError(
            f"num_valid_batches is {num_valid_batches} but {len(names)} datasets are validated "
            f"({', '.join(names)}): each needs at least one batch"
        )
    caps = {name: line_counts[name] // world_size // batch_size for name in names}
    for name, cap in caps.items():
        if cap < 1:
            raise ValueError(
                f"validation set of {name!r} has {line_counts[name]} rows: not one full batch of "
                f"{batch_size} per rank on {world_size} ranks. Lower batch_size or add rows."
            )
    weights = {name: curriculum.datasets[name].valid_weight for name in names}
    plan = apportion(weights, num_valid_batches, caps, minimum=1)
    if sum(plan.values()) < num_valid_batches:
        logger.warning(
            f"validation sets hold only {sum(plan.values())} of the {num_valid_batches} "
            "batches asked for; every one is used in full"
        )
    return plan


def count_lines(path: str) -> int:
    """Rows the loader will see (it shards every line, blank or not)."""
    with Path(path).open("rb") as f:
        return sum(1 for _ in f)


def combine_validation(
    per_dataset: dict[str, dict[str, float]], batches: dict[str, int]
) -> dict[str, float]:
    """The benchmark's own metrics (batch-weighted mean, so the sample mean) plus each
    dataset's under a `<name>/<metric>` key."""
    out: dict[str, float] = {}
    total = sum(batches[name] for name in per_dataset)
    metrics = next(iter(per_dataset.values())).keys() if per_dataset else []
    for metric in metrics:
        out[metric] = sum(per_dataset[name][metric] * batches[name] for name in per_dataset) / total
    for name, values in per_dataset.items():
        for metric, value in values.items():
            out[f"{name}/{metric}"] = value
    return out


class MixSchedule:
    """Which dataset each of a step's micro-batch slots is drawn from.

    A step has `slots = grad_accum_steps * world_size` slots. Within a phase they are
    assigned by smooth weighted round-robin: every slot adds each dataset's weight to its
    credit, the dataset with the most credit takes the slot and pays 1. After n slots each
    dataset has been picked within 1 of n * weight, and the credit carries across steps, so
    20% of 8 slots a step is 1.6 on average and exactly 20% over a phase. The sequence
    depends only on the curriculum, so all ranks agree without talking to each other.

    Steps must be asked for in order; `fast_forward` resumes at any step.
    """

    def __init__(self, curriculum: Curriculum, slots_per_step: int):
        if slots_per_step < 1:
            raise ValueError(f"slots_per_step must be >= 1, got {slots_per_step}")
        self.curriculum = curriculum
        self.slots_per_step = slots_per_step
        self._quota = [self._quantize(phase.mix) for phase in curriculum.phases]
        self._phase = -1
        self._credit: dict[str, int] = {}
        self._next_step = 0

    @staticmethod
    def _quantize(mix: dict[str, float]) -> dict[str, int]:
        return apportion(mix, QUOTA_RESOLUTION)

    def fast_forward(self, step: int) -> None:
        """Position the schedule as if `step` optimizer steps had completed."""
        index = self.curriculum.phase_index(step)
        start = self.curriculum.phase_start(index)
        self._phase, self._credit, self._next_step = -1, {}, start
        for replayed in range(start, step):
            self.slots(replayed)

    def slots(self, step: int) -> list[str]:
        if step != self._next_step:
            raise ValueError(f"schedule is at step {self._next_step}, asked for step {step}")
        index = self.curriculum.phase_index(step)
        if index != self._phase:
            self._phase = index
            self._credit = {name: 0 for name in self._quota[index]}
        quota = {n: q for n, q in self._quota[index].items() if q > 0}
        picked = []
        for _ in range(self.slots_per_step):
            for name, share in quota.items():
                self._credit[name] += share
            name = max(quota, key=lambda n: self._credit[n])  # first wins a tie: declared order
            self._credit[name] -= QUOTA_RESOLUTION
            picked.append(name)
        self._next_step += 1
        return picked


Stream = tuple[Iterator, int]  # (endless iterator of batches, rows in this rank's shard)


class CurriculumLoader:
    """This rank's batches for a run following `schedule`.

    One endless stream per dataset, opened the first time a slot needs it (so a dataset a
    later phase introduces costs nothing until then) and kept for the rest of the run:
    reaching a phase again continues where the dataset stopped. A stream that reaches the
    end of its data starts over reshuffled, which is the rollover.
    """

    def __init__(
        self,
        schedule: MixSchedule,
        open_stream: Callable[[str], Stream],
        rank: int,
        world_size: int,
        batch_size: int,
    ):
        self.schedule = schedule
        self._open = open_stream
        self.rank = rank
        self.world_size = world_size
        self.batch_size = batch_size
        self._streams: dict[str, Stream] = {}
        self.batches_drawn: dict[str, int] = {}
        self._step = -1
        self._names: list[str] = []

    def next_batch(self, step: int, micro: int):
        """The batch for micro-batch `micro` of optimizer step `step` on this rank."""
        if step != self._step:
            self._names, self._step = self.schedule.slots(step), step
        name = self._names[micro * self.world_size + self.rank]
        if name not in self._streams:
            self._streams[name] = self._open(name)
            self.batches_drawn[name] = 0
        self.batches_drawn[name] += 1
        return next(self._streams[name][0])

    def epochs(self) -> dict[str, float]:
        """Passes over each opened dataset's shard so far (rollovers show as whole numbers)."""
        return {
            name: self.batches_drawn[name] * self.batch_size / max(1, rows)
            for name, (_, rows) in self._streams.items()
        }

    def close(self) -> None:
        for stream, _ in self._streams.values():
            close = getattr(stream, "close", None)
            if close is not None:
                close()
        self._streams.clear()
