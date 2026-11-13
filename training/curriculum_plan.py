"""Plan a training curriculum before the run: epochs per dataset, and edits to the config.

The trainer reports each dataset's epochs (rollovers) only while it runs. They are known in
advance, because the mix is an exact quota (training/curriculum.py, `MixSchedule`):

    samples of dataset d in phase p = steps_p * effective_batch * share_{p,d}
    epochs(d) = samples / rows_d          (a rank's shard is rows_d / world_size: same ratio)

This module holds that arithmetic, the inverse (target epochs -> steps and mix), the
discovery of a dataset's manifests from its folder, and a rewrite of just the `data.datasets`
/ `data.curriculum` part of a training yaml that leaves every other line, comments included,
as it was. Pure python (no torch, no audio): `training/plan_curriculum.py` is the interactive
front end.
"""

import difflib
import json
import math
import os
import re
import shlex
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from training.curriculum import NAME_RE, Curriculum, parse_curriculum

DEFAULT_MICRO_BATCH = 8  # TrainArgs.batch_size
DEFAULT_ACCUM = 1
DEFAULT_WORLD_SIZE = 8
DEFAULT_VALID_WEIGHT = 0.1
DEFAULT_MAX_DURATION_SEC = 30.0
# Rows' `duration` key. Words carry only word/start/end, so the first match is the row's.
_DURATION = re.compile(rb'"duration"\s*:\s*([0-9.eE+-]+)')
# Most processed stage first: aligned rows carry everything the earlier stages add
# (hebrew_training/manifest_files.py).
STAGE_SUFFIXES = ("_aligned", "_diac", "")
VALID_SPLIT_NAMES = ("valid", "validation", "val", "dev")
GENERIC_FOLDERS = {"data", "manifest", "manifests", "jsonl", "train", "output", "out"}
CACHE_PATH = Path.home() / ".cache" / "pocket-tts" / "curriculum_stats.json"


class PlanError(ValueError):
    """A request the plan cannot honour; the message says why."""


# --- dataset size ------------------------------------------------------------------------


@dataclass(frozen=True)
class DatasetStats:
    rows: int
    hours: float | None  # one epoch's audio, each row cut to max_duration_sec; None: unknown
    source: str  # "scanned" | "cached" | "manual"


def scan_manifest(path: Path, max_duration_sec: float) -> tuple[int, float]:
    """(rows, hours of one epoch). Rows are lines, blank or not: that is what the loader shards."""
    rows, seconds = 0, 0.0
    with path.open("rb") as handle:
        for line in handle:
            rows += 1
            match = _DURATION.search(line)
            if match:
                try:
                    seconds += min(float(match.group(1)), max_duration_sec)
                except ValueError:
                    pass
    return rows, seconds / 3600


class StatsStore:
    """Row counts and hours of manifests, scanned once and remembered.

    The cache is for the machine that cannot see the data (the manifests live on the training
    server, the planning happens on a laptop): a size is entered once with `set_manual`, or
    comes from an earlier scan, and is used whenever the file is not there.
    """

    def __init__(self, cache_path: Path | None, max_duration_sec: float):
        self.cache_path = cache_path
        self.max_duration_sec = max_duration_sec
        self._memory: dict[str, DatasetStats] = {}
        self._cache: dict[str, dict[str, Any]] = {}
        if cache_path is not None and cache_path.exists():
            try:
                self._cache = json.loads(cache_path.read_text())
            except (OSError, ValueError):
                self._cache = {}

    @staticmethod
    def _key(path: str) -> str:
        return str(Path(path).expanduser().absolute())

    def get(self, path: str) -> DatasetStats | None:
        if not path:
            return None
        key = self._key(path)
        if key in self._memory:
            return self._memory[key]
        file = Path(key)
        entry = self._cache.get(key)
        stats = None
        if file.is_file():
            stat = file.stat()
            fresh = (
                entry is not None
                and entry.get("size") == stat.st_size
                and entry.get("mtime") == stat.st_mtime
                and entry.get("cap") == self.max_duration_sec
                and entry.get("source") == "scanned"
            )
            if fresh:
                stats = DatasetStats(entry["rows"], entry["hours"], "cached")
            else:
                rows, hours = scan_manifest(file, self.max_duration_sec)
                stats = DatasetStats(rows, hours, "scanned")
                self._cache[key] = {
                    "rows": rows,
                    "hours": hours,
                    "size": stat.st_size,
                    "mtime": stat.st_mtime,
                    "cap": self.max_duration_sec,
                    "source": "scanned",
                }
                self.save()
        elif entry is not None:
            hours = entry["hours"] if entry.get("cap") == self.max_duration_sec else None
            stats = DatasetStats(entry["rows"], hours, "cached")
        if stats is not None:
            self._memory[key] = stats
        return stats

    def set_manual(self, path: str, rows: int, hours: float | None) -> DatasetStats:
        if rows < 1:
            raise PlanError(f"rows must be >= 1, got {rows}")
        key = self._key(path)
        stats = DatasetStats(rows, hours, "manual")
        self._memory[key] = stats
        self._cache[key] = {
            "rows": rows,
            "hours": hours,
            "cap": self.max_duration_sec,
            "source": "manual",
        }
        self.save()
        return stats

    def save(self) -> None:
        if self.cache_path is None:
            return
        try:
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            self.cache_path.write_text(json.dumps(self._cache, indent=1))
        except OSError:
            pass  # a cache that cannot be written only costs a rescan


# --- the arithmetic ----------------------------------------------------------------------


def normalize(mix: dict[str, float]) -> dict[str, float]:
    total = sum(w for w in mix.values() if w > 0)
    if total <= 0:
        raise PlanError("a mix needs at least one positive weight")
    return {name: w / total for name, w in mix.items() if w > 0}


@dataclass(frozen=True)
class PhaseForecast:
    index: int  # 0-based
    start_step: int  # first step, 1-based
    steps: int
    share: dict[str, float]  # normalized mix
    samples: dict[str, float]  # drawn in this phase
    gained: dict[str, float | None]  # epochs added by this phase (None: rows unknown)
    cumulative: dict[str, float | None]  # epochs at the end of this phase

    @property
    def end_step(self) -> int:
        return self.start_step + self.steps - 1


def forecast(
    phases: list[tuple[int, dict[str, float]]], rows: dict[str, int | None], batch: int
) -> list[PhaseForecast]:
    """Epochs of every dataset after each phase. `phases` is (steps, relative weights)."""
    if batch < 1:
        raise PlanError(f"effective batch must be >= 1, got {batch}")
    drawn = {name: 0.0 for name in rows}
    out, start = [], 1
    for index, (steps, mix) in enumerate(phases):
        share = normalize(mix)
        samples = {name: steps * batch * share.get(name, 0.0) for name in rows}
        for name, count in samples.items():
            drawn[name] += count
        gained = {n: (samples[n] / rows[n] if rows[n] else None) for n in rows}
        cumulative = {n: (drawn[n] / rows[n] if rows[n] else None) for n in rows}
        out.append(PhaseForecast(index, start, steps, share, samples, gained, cumulative))
        start += steps
    return out


def advice(
    plan_forecast: list[PhaseForecast],
    rows: dict[str, int | None],
    low: float,
    high: float,
    micro_batch: int | None = None,
    world_size: int | None = None,
) -> list[str]:
    """What to look at: datasets that end the run under `low` or over `high` epochs."""
    notes = []
    used = [n for n in rows if any(p.share.get(n) for p in plan_forecast)]
    final = plan_forecast[-1].cumulative if plan_forecast else {}
    for name in used:
        epochs = final.get(name)
        if epochs is None:
            notes.append(f"{name}: size unknown, no epoch forecast (use `rows {name} <rows>`)")
            continue
        if epochs < low:
            unseen = max(0.0, 1 - epochs)
            notes.append(
                f"{name}: {epochs:.2f} epochs, under {low:g} ({unseen:.0%} of its rows are never drawn)"
            )
        elif epochs > high:
            notes.append(
                f"{name}: {epochs:.1f} epochs, over {high:g} (it rolls over {int(epochs)} times)"
            )
        if micro_batch and world_size and rows[name] and rows[name] // world_size < micro_batch:
            notes.append(
                f"{name}: {rows[name]} rows over {world_size} ranks cannot fill a batch of {micro_batch}"
            )
    return notes


def solve_phase(
    rows: dict[str, int | None],
    targets: dict[str, float],
    batch: int,
    drawn_before: dict[str, float],
    steps: int | None = None,
) -> tuple[int, dict[str, float]]:
    """Steps and mix that bring each dataset to its target cumulative epochs at the phase's end.

    `drawn_before` is the samples each dataset has had in earlier phases. A dataset missing
    from `targets` (or already past its target) draws nothing in this phase. With `steps`
    given the mix is the same proportion, and the phase may fall short of (or pass) the targets:
    run `forecast` on the result to see by how much.
    """
    needed = {}
    for name, target in targets.items():
        if target < 0 or not math.isfinite(target):
            raise PlanError(f"target for {name} must be >= 0, got {target}")
        if not rows.get(name):
            raise PlanError(f"{name} has no known size: set it with `rows` first")
        short = target * rows[name] - drawn_before.get(name, 0.0)
        if short > 0:
            needed[name] = short
    if not needed:
        raise PlanError("every target is already reached by the earlier phases: nothing to draw")
    total = sum(needed.values())
    needed_steps = max(1, math.ceil(total / batch))
    mix = {name: round(value / total, 6) for name, value in needed.items()}
    mix = {name: max(weight, 1e-6) for name, weight in mix.items()}
    return (needed_steps if steps is None else steps), mix


def scale_others(mix: dict[str, float], name: str, percent: float) -> dict[str, float]:
    """`name` at `percent` of the phase; the others keep their proportions in what is left."""
    if not 0 <= percent <= 100:
        raise PlanError(f"a share is 0-100 percent, got {percent}")
    others = {n: w for n, w in mix.items() if n != name and w > 0}
    rest = 100 - percent
    result: dict[str, float] = {}
    if others and rest > 0:
        total = sum(others.values())
        result.update({n: round(rest * w / total, 4) for n, w in others.items()})
    elif rest > 0 and percent < 100:
        raise PlanError(f"{name} is the only dataset in the phase, so it cannot be below 100%")
    if percent > 0:
        result[name] = percent
    if not result:
        raise PlanError("a phase needs at least one dataset")
    return result


# --- finding a dataset from its folder ---------------------------------------------------


@dataclass(frozen=True)
class FoundDataset:
    folder: Path
    name: str
    train: Path | None
    valid: Path | None
    stage: str  # "aligned" | "diac" | "built" of the train manifest
    present: tuple[str, ...]  # every manifest file seen, to show the choice
    notes: tuple[str, ...] = ()


def clean_path(text: str) -> Path:
    """A pasted path: quotes, shell escapes (a drag from Finder), ~."""
    text = text.strip()
    try:
        parts = shlex.split(text)
    except ValueError:
        parts = [text]
    if len(parts) == 1:
        text = parts[0]
    return Path(text).expanduser()


def suggest_name(folder: Path, taken: set[str]) -> str:
    base = folder.name
    if base.lower() in GENERIC_FOLDERS and folder.parent.name:
        base = folder.parent.name
    name = re.sub(r"[^A-Za-z0-9_.-]+", "_", base).strip("_.-") or "dataset"
    candidate, n = name, 2
    while candidate in taken:
        candidate, n = f"{name}_{n}", n + 1
    return candidate


def _candidates(folder: Path, splits: tuple[str, ...]) -> list[tuple[str, str, Path]]:
    found = []
    for split in splits:
        for suffix in STAGE_SUFFIXES:
            path = folder / f"{split}{suffix}.jsonl"
            if path.is_file():
                found.append((split, suffix.lstrip("_") or "built", path))
    return found


def _row_problem(row: Any) -> str | None:
    if not isinstance(row, dict):
        return "is not a JSON object"
    if not isinstance(row.get("path"), str):
        return "has no string `path`"
    if not isinstance(row.get("duration"), int | float):
        return "has no numeric `duration`"
    words = row.get("words")
    if not isinstance(words, list) or not words:
        return "has no `words` list (the loader cuts the voice prompt at word boundaries)"
    return None


def check_manifest(path: Path, rows: int = 5) -> list[str]:
    """Problems in the first rows, as readable notes; empty when it looks like a manifest."""
    notes = []
    checked = 0
    try:
        with path.open(encoding="utf-8") as handle:
            for number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except ValueError:
                    notes.append(f"{path.name} row {number} is not JSON")
                    break
                problem = _row_problem(row)
                if problem:
                    notes.append(f"{path.name} row {number} {problem}")
                    break
                if checked == 0 and not Path(row["path"]).expanduser().exists():
                    notes.append(
                        f"{path.name}: first audio file {row['path']} not found from here "
                        "(fine if the data lives on the training machine)"
                    )
                checked += 1
                if checked == rows:
                    break
    except (OSError, UnicodeDecodeError) as error:
        notes.append(f"{path.name} cannot be read: {error}")
    if checked == 0 and not notes:
        notes.append(f"{path.name} is empty")
    return notes


def discover_dataset(folder: Path, taken: set[str] | None = None) -> FoundDataset:
    """Find the train / valid manifests of a dataset folder (the layout of manifest_files.py).

    The most processed stage wins (`_aligned` over `_diac` over the plain build). A pasted
    jsonl file stands for its folder. Falls back to a `train*.jsonl` one level down.
    """
    folder = folder.expanduser()
    if folder.is_file():
        folder = folder.parent
    if not folder.is_dir():
        raise PlanError(f"{folder} is not a folder on this machine")
    root = folder
    cands = _candidates(folder, ("train", *VALID_SPLIT_NAMES))
    if not any(c[0] == "train" for c in cands):
        for child in sorted(p for p in folder.iterdir() if p.is_dir()):
            nested = _candidates(child, ("train", *VALID_SPLIT_NAMES))
            if any(c[0] == "train" for c in nested):
                folder, cands = child, nested
                break
    train = next((c for c in cands if c[0] == "train"), None)
    if train is None:
        raise PlanError(
            f"no train.jsonl / train_diac.jsonl / train_aligned.jsonl in {root} (or one folder down)"
        )
    valid = next((c for split in VALID_SPLIT_NAMES for c in cands if c[0] == split), None)
    notes = []
    if valid is None:
        notes.append("no validation manifest found: valid_weight will be 0 (not validated)")
    elif valid[1] != train[1]:
        notes.append(f"train is `{train[1]}` but valid is `{valid[1]}`: the stages differ")
    notes.extend(check_manifest(train[2]))
    if valid is not None:
        notes.extend(check_manifest(valid[2], rows=2))
    present = tuple(f"{c[2].name}" for c in cands)
    return FoundDataset(
        folder,
        suggest_name(folder, taken or set()),
        train[2],
        valid[2] if valid else None,
        train[1],
        present,
        tuple(notes),
    )


def config_path_for(path: Path) -> str:
    """How a manifest is written in the config: relative to the working directory when inside it."""
    absolute = path.expanduser().absolute()
    try:
        return (
            os.path.relpath(absolute, Path.cwd())
            if absolute.is_relative_to(Path.cwd())
            else str(absolute)
        )
    except ValueError:
        return str(absolute)


# --- the plan ----------------------------------------------------------------------------


@dataclass
class DatasetEntry:
    train_jsonl: str
    valid_jsonl: str = ""
    valid_weight: float = DEFAULT_VALID_WEIGHT


@dataclass
class PhaseEntry:
    steps: int
    mix: dict[str, float]


@dataclass
class Plan:
    datasets: dict[str, DatasetEntry] = field(default_factory=dict)
    phases: list[PhaseEntry] = field(default_factory=list)
    checkpoint_at_phase_end: bool = True
    micro_batch: int = DEFAULT_MICRO_BATCH
    grad_accum: int = DEFAULT_ACCUM
    world_size: int = DEFAULT_WORLD_SIZE
    batch_override: int | None = None
    max_duration_sec: float = DEFAULT_MAX_DURATION_SEC

    @property
    def effective_batch(self) -> int:
        return self.batch_override or self.micro_batch * self.grad_accum * self.world_size

    @property
    def total_steps(self) -> int:
        return sum(p.steps for p in self.phases)

    def raw_datasets(self) -> dict[str, dict[str, Any]]:
        out = {}
        for name, entry in self.datasets.items():
            raw: dict[str, Any] = {"train_jsonl": entry.train_jsonl}
            if entry.valid_jsonl:
                raw["valid_jsonl"] = entry.valid_jsonl
            raw["valid_weight"] = entry.valid_weight
            out[name] = raw
        return out

    def raw_phases(self) -> list[dict[str, Any]]:
        return [{"steps": p.steps, "mix": dict(p.mix)} for p in self.phases]

    def validate(self) -> Curriculum:
        """The trainer's own checks (parse_curriculum); PlanError with its message."""
        try:
            return parse_curriculum(
                self.raw_datasets(), self.raw_phases(), self.checkpoint_at_phase_end
            )
        except ValueError as error:
            raise PlanError(str(error)) from error

    def phase(self, number: int) -> PhaseEntry:
        if not 1 <= number <= len(self.phases):
            raise PlanError(f"there is no phase {number}: the plan has {len(self.phases)}")
        return self.phases[number - 1]

    def add_dataset(self, name: str, entry: DatasetEntry) -> None:
        if not NAME_RE.fullmatch(name):
            raise PlanError(f"dataset name {name!r} must match {NAME_RE.pattern}")
        if name in self.datasets:
            raise PlanError(f"dataset {name!r} already exists")
        self.datasets[name] = entry

    def remove_dataset(self, name: str) -> list[int]:
        """Drop a dataset; returns the (1-based) phases left with nothing to draw."""
        if name not in self.datasets:
            raise PlanError(f"no dataset {name!r}")
        del self.datasets[name]
        emptied = []
        for number, phase in enumerate(self.phases, 1):
            phase.mix.pop(name, None)
            if not phase.mix:
                emptied.append(number)
        return emptied

    def set_mix(self, number: int, mix: dict[str, float]) -> None:
        phase = self.phase(number)
        for name in mix:
            if name not in self.datasets:
                raise PlanError(f"no dataset {name!r} (have: {', '.join(self.datasets)})")
        phase.mix = {n: w for n, w in normalize(mix).items()}
        phase.mix = {n: round(w * 100, 4) for n, w in phase.mix.items()}

    def set_share(self, number: int, name: str, percent: float) -> None:
        phase = self.phase(number)
        if name not in self.datasets:
            raise PlanError(f"no dataset {name!r} (have: {', '.join(self.datasets)})")
        phase.mix = scale_others(phase.mix, name, percent)

    def rows(self, store: StatsStore) -> dict[str, int | None]:
        out: dict[str, int | None] = {}
        for name, entry in self.datasets.items():
            stats = store.get(entry.train_jsonl)
            out[name] = stats.rows if stats else None
        return out

    def forecast(self, store: StatsStore) -> list[PhaseForecast]:
        if not self.phases:
            return []
        return forecast(
            [(p.steps, p.mix) for p in self.phases], self.rows(store), self.effective_batch
        )

    @classmethod
    def from_config(cls, path: Path, text: str | None = None) -> "Plan":
        text = path.read_text() if text is None else text
        raw = yaml.safe_load(text) or {}
        data = raw.get("data") or {}
        plan = cls(
            micro_batch=int(raw.get("batch_size", DEFAULT_MICRO_BATCH)),
            grad_accum=int(raw.get("grad_accum_steps", DEFAULT_ACCUM)),
            checkpoint_at_phase_end=bool(data.get("curriculum_checkpoint_at_phase_end", True)),
            max_duration_sec=float(data.get("max_duration_sec", DEFAULT_MAX_DURATION_SEC)),
        )
        if data.get("datasets") or data.get("curriculum"):
            for name, spec in (data.get("datasets") or {}).items():
                plan.datasets[str(name)] = DatasetEntry(
                    str(spec.get("train_jsonl", "")),
                    str(spec.get("valid_jsonl", "")),
                    float(spec.get("valid_weight", 0)),
                )
            for phase in data.get("curriculum") or []:
                plan.phases.append(
                    PhaseEntry(
                        int(phase["steps"]), {str(k): float(v) for k, v in phase["mix"].items()}
                    )
                )
        elif data.get("train_jsonl"):
            # A plain config: one dataset, one phase of max_steps. Saving turns it into a curriculum.
            name = suggest_name(Path(data["train_jsonl"]).parent, set())
            plan.datasets[name] = DatasetEntry(
                str(data["train_jsonl"]), str(data.get("valid_jsonl", "")), 1.0
            )
            plan.phases.append(PhaseEntry(int(raw.get("max_steps", 100_000)), {name: 1.0}))
        return plan


# --- writing the config back -------------------------------------------------------------


def _scalar(value: str) -> str:
    """A yaml scalar for a path: plain when it reads back as the same string, else JSON-quoted
    (which is valid yaml)."""
    try:
        if re.fullmatch(r"[A-Za-z0-9_./\\@+-]+", value) and yaml.safe_load(value) == value:
            return value
    except yaml.YAMLError:
        pass
    return json.dumps(value, ensure_ascii=False)


def fmt_number(value: float) -> str:
    return str(int(value)) if float(value).is_integer() else f"{value:.12g}"


def render_blocks(plan: Plan, indent: str = "  ") -> tuple[list[str], list[str]]:
    """The `datasets:` and `curriculum:` blocks (lines) of the plan, at `indent`."""
    datasets = [f"{indent}datasets:"]
    for name, entry in plan.datasets.items():
        datasets.append(f"{indent * 2}{name}:")
        datasets.append(f"{indent * 3}train_jsonl: {_scalar(entry.train_jsonl)}")
        if entry.valid_jsonl:
            datasets.append(f"{indent * 3}valid_jsonl: {_scalar(entry.valid_jsonl)}")
        datasets.append(f"{indent * 3}valid_weight: {fmt_number(entry.valid_weight)}")
    curriculum = [f"{indent}curriculum:"]
    for phase in plan.phases:
        mix = ", ".join(f"{n}: {fmt_number(w)}" for n, w in phase.mix.items())
        curriculum.append(f"{indent * 2}- steps: {phase.steps}")
        curriculum.append(f"{indent * 3}mix: {{{mix}}}")
    return datasets, curriculum


def _code(line: str) -> str:
    return re.sub(r"\s+#.*$", "", line).rstrip()


def keep_comments(old: list[str], new: list[str]) -> list[str]:
    """`new`, except that a line whose content is unchanged stays as it was, comment included."""
    matcher = difflib.SequenceMatcher(
        None, [_code(x) for x in old], [_code(x) for x in new], autojunk=False
    )
    out: list[str] = []
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        out.extend(old[i1:i2] if tag == "equal" else new[j1:j2])
    return out


def _indent_of(line: str) -> int:
    return len(line) - len(line.lstrip())


def _data_span(lines: list[str]) -> tuple[int, int]:
    start = next((i for i, line in enumerate(lines) if re.match(r"^data:\s*(#.*)?$", line)), None)
    if start is None:
        raise PlanError("the config has no top-level `data:` section")
    end = len(lines)
    for i in range(start + 1, len(lines)):
        if lines[i].strip() and _indent_of(lines[i]) == 0:
            end = i
            break
    while end > start + 1 and not lines[end - 1].strip():
        end -= 1
    return start, end


def _child_indent(lines: list[str], start: int, end: int) -> int:
    for i in range(start + 1, end):
        if lines[i].strip() and not lines[i].lstrip().startswith("#"):
            return _indent_of(lines[i])
    return 2


def _child_span(lines: list[str], key: str) -> tuple[int, int] | None:
    start, end = _data_span(lines)
    indent = _child_indent(lines, start, end)
    pattern = re.compile(rf"^ {{{indent}}}{re.escape(key)}:(\s|$)")
    for i in range(start + 1, end):
        if pattern.match(lines[i]):
            stop = i + 1
            while stop < end and (not lines[stop].strip() or _indent_of(lines[stop]) > indent):
                stop += 1
            while stop > i + 1 and not lines[stop - 1].strip():
                stop -= 1
            return i, stop
    return None


def _set_child(lines: list[str], key: str, new: list[str], after: str | None = None) -> None:
    span = _child_span(lines, key)
    if span is not None:
        lines[span[0] : span[1]] = keep_comments(lines[span[0] : span[1]], new)
        return
    anchor = _child_span(lines, after) if after else None
    if anchor is not None:
        at = anchor[1]
    else:
        at = _data_span(lines)[1]
    lines[at:at] = new


def render_config(original: str, plan: Plan) -> str:
    """`original` with only the plan's datasets, curriculum, phase-end flag (and max_steps, when
    the file sets it) replaced. Everything else is left byte for byte."""
    curriculum = plan.validate()
    lines = original.splitlines()
    start, end = _data_span(lines)
    indent = " " * _child_indent(lines, start, end)
    datasets, phases = render_blocks(plan, indent)

    # A plain config's two manifests become the first dataset: drop them where they were.
    plain = [_child_span(lines, key) for key in ("train_jsonl", "valid_jsonl")]
    first_plain = min((s[0] for s in plain if s), default=None)
    for key in ("valid_jsonl", "train_jsonl"):
        span = _child_span(lines, key)
        if span is not None:
            if first_plain is not None and span[0] == first_plain:
                lines[span[0] : span[1]] = datasets + phases
                datasets = phases = []
            else:
                del lines[span[0] : span[1]]
    if datasets:  # not already placed where the plain manifests were
        _set_child(lines, "datasets", datasets)
        _set_child(lines, "curriculum", phases, after="datasets")

    flag = (
        f"{indent}curriculum_checkpoint_at_phase_end: {str(plan.checkpoint_at_phase_end).lower()}"
    )
    if _child_span(lines, "curriculum_checkpoint_at_phase_end") is not None:
        _set_child(lines, "curriculum_checkpoint_at_phase_end", [flag])
    elif not plan.checkpoint_at_phase_end:
        _set_child(lines, "curriculum_checkpoint_at_phase_end", [flag], after="curriculum")

    for i, line in enumerate(lines):
        if re.match(r"^max_steps:\s", line):
            comment = re.search(r"\s(#.*)$", line)
            lines[i] = f"max_steps: {curriculum.total_steps}" + (
                f"  {comment.group(1)}" if comment else ""
            )

    result = "\n".join(lines) + ("\n" if original.endswith("\n") else "")
    check_render(result, plan)
    return result


def _canonical(value: Any) -> str:
    """JSON of `value` with numbers compared as rounded floats: the file prints 12 digits."""

    def plain(v: Any) -> Any:
        if isinstance(v, dict):
            return {str(k): plain(x) for k, x in v.items()}
        if isinstance(v, list):
            return [plain(x) for x in v]
        return round(float(v), 9) if isinstance(v, int | float) and not isinstance(v, bool) else v

    return json.dumps(plain(value), sort_keys=True)


def check_render(text: str, plan: Plan) -> None:
    """The rewritten file must read back as exactly the plan; otherwise nothing is written."""
    try:
        data = (yaml.safe_load(text) or {})["data"]
    except (yaml.YAMLError, KeyError, TypeError) as error:
        raise PlanError(
            f"the rewritten config does not parse ({error}); nothing was written"
        ) from error
    got = {"datasets": data.get("datasets") or {}, "curriculum": data.get("curriculum") or []}
    expect = {"datasets": plan.raw_datasets(), "curriculum": plan.raw_phases()}
    if _canonical(got) != _canonical(expect) or "train_jsonl" in data or "valid_jsonl" in data:
        raise PlanError(
            "the config's data section is shaped in a way this tool cannot rewrite safely "
            "(it did not read back as the plan); edit the file by hand"
        )


def unified_diff(before: str, after: str, name: str) -> str:
    return "".join(
        difflib.unified_diff(
            before.splitlines(True),
            after.splitlines(True),
            f"{name} (current)",
            f"{name} (new)",
            n=2,
        )
    )
