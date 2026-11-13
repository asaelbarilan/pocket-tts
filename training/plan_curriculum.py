"""Plan and edit a training curriculum: see each dataset's epochs per phase before the run.

    uv run python -m training.plan_curriculum training/configs/finetune_hebrew_curriculum.yaml
    uv run python -m training.plan_curriculum CONFIG --show        # print the forecast and exit

Run it from the repo root (manifest paths in a config are relative to it). The table answers
"how many times does each dataset roll over?" from the config and the manifests' row counts,
at the effective batch size (batch_size x grad_accum_steps x ranks; `--world-size` says how
many ranks, `batch N` overrides the product). Inside, `help` lists the commands: add a dataset
by pasting its folder, set phase lengths and shares, `solve` a phase from target epochs, and
`save` (a diff first). Saving rewrites only `data.datasets` / `data.curriculum` of the yaml
(and `max_steps` when the file has one); the rest of the file is left as it is.

If the manifests live on the training server, scan them there once (the sizes are cached in
~/.cache/pocket-tts/curriculum_stats.json), or enter them with `rows <dataset> <rows> [hours]`.
"""

import argparse
import copy
import shlex
import sys
from collections.abc import Callable
from pathlib import Path

from training.curriculum_plan import (
    CACHE_PATH,
    DEFAULT_VALID_WEIGHT,
    DEFAULT_WORLD_SIZE,
    DatasetEntry,
    PhaseEntry,
    Plan,
    PlanError,
    StatsStore,
    advice,
    clean_path,
    config_path_for,
    discover_dataset,
    render_config,
    solve_phase,
    unified_diff,
)

HELP = """\
  show                          the forecast again
  add [folder]                  add a dataset: paste its folder, the manifests are picked up
  rm <dataset>                  remove a dataset (from every phase)
  steps <phase> <n>             phase length in optimizer steps (15000, 15k, 1.5m)
  share <phase> <dataset> <%>   dataset's share of a phase; the others keep their proportions
  mix <phase> a=90 b=5 c=5      set a whole phase's mix (normalized to 100%)
  phase add [steps]             append a phase (asks for the mix)
  phase rm <phase>              drop a phase
  solve [phase]                 from target epochs per dataset to the steps and mix that hit them
  batch <n>                     effective batch size to assume (default: batch x accum x ranks)
  world <n>                     number of ranks (GPUs); recomputes the effective batch
  rows <dataset> <rows> [h]     enter a dataset's size when its manifest is not on this machine
  valid <dataset> <weight>      its share of the validation benchmark
  ckpt on|off                   checkpoint and validate at each phase end
  warn <low> <high>             epochs under / over which a dataset is flagged
  diff                          what `save` would change
  save [path]                   write the config (shows the diff, asks first)
  quit
A phase is 1-based. Dataset names may be abbreviated while they stay unambiguous."""


def parse_int(text: str) -> int:
    raw = text.strip().lower().replace(",", "").replace("_", "")
    scale = {"k": 1_000, "m": 1_000_000}.get(raw[-1:], 1)
    try:
        value = float(raw[:-1] if scale != 1 else raw) * scale
    except ValueError:
        raise PlanError(f"{text!r} is not a number") from None
    if value != int(value) or value < 1:
        raise PlanError(f"{text!r} is not a whole number >= 1")
    return int(value)


def parse_float(text: str) -> float:
    try:
        return float(text.strip().rstrip("%"))
    except ValueError:
        raise PlanError(f"{text!r} is not a number") from None


def epochs_text(value: float | None) -> str:
    if value is None:
        return "?"
    if value < 10:
        return f"{value:.2f}"
    return f"{value:.1f}" if value < 100 else f"{value:.0f}"


def table(header: list[str], rows: list[list[str]]) -> list[str]:
    widths = [max(len(r[i]) for r in [header, *rows]) for i in range(len(header))]
    out = []
    for row in [header, *rows]:
        cells = [row[0].ljust(widths[0])] + [c.rjust(w) for c, w in zip(row[1:], widths[1:])]
        out.append("  " + "  ".join(cells).rstrip())
    return out


def render_forecast(plan: Plan, store: StatsStore, low: float, high: float) -> str:
    batch = plan.effective_batch
    rows = plan.rows(store)
    lines = []
    source = (
        f"{batch} (set by `batch`)"
        if plan.batch_override
        else f"{batch} = {plan.micro_batch} batch x {plan.grad_accum} accum x {plan.world_size} ranks"
    )
    lines.append(
        f"effective batch {source}  |  {plan.total_steps:,} steps  |  "
        f"{plan.total_steps * batch:,} samples"
    )
    if not plan.phases:
        return "\n".join([*lines, "no phases yet: `phase add`"])
    result = plan.forecast(store)
    lines.append("")
    for f in result:
        mix = " + ".join(f"{n} {s:.4g}%" for n, s in ((n, s * 100) for n, s in f.share.items()))
        lines.append(
            f"  phase {f.index + 1}: steps {f.start_step:,}-{f.end_step:,} ({f.steps:,})  {mix}"
        )
    lines.append("")
    header = ["dataset", "rows", "hours/ep"] + [f"P{f.index + 1} +ep (total)" for f in result]
    header += ["epochs", "hours seen"]
    body = []
    for name in plan.datasets:
        stats = store.get(plan.datasets[name].train_jsonl)
        mark = "" if stats is None or stats.source == "scanned" else "~"
        cells = [
            name,
            f"{stats.rows:,}{mark}" if stats else "?",
            f"{stats.hours:,.1f}{mark}" if stats and stats.hours is not None else "-",
        ]
        for f in result:
            if not f.share.get(name):
                cells.append("-")
            else:
                gained, cumulative = f.gained[name], f.cumulative[name]
                cells.append(
                    "?" if gained is None else f"+{epochs_text(gained)} ({epochs_text(cumulative)})"
                )
        total = result[-1].cumulative[name]
        cells.append("-" if total == 0 else epochs_text(total))
        seen = total * stats.hours if (stats and stats.hours is not None and total) else None
        cells.append("-" if seen is None else f"{seen:,.0f}")
        body.append(cells)
    lines.extend(table(header, body))
    if any(
        store.get(e.train_jsonl) and store.get(e.train_jsonl).source != "scanned"
        for e in plan.datasets.values()
    ):
        lines.append("  ~ size from the cache or typed in, not read from the manifest")
    notes = advice(result, rows, low, high, plan.micro_batch, plan.world_size)
    if notes:
        lines.append("")
        lines.extend(f"  ! {note}" for note in notes)
    try:
        plan.validate()
    except PlanError as error:
        lines.extend(["", f"  ! not savable yet: {error}"])
    return "\n".join(lines)


class Planner:
    def __init__(
        self,
        path: Path,
        plan: Plan,
        store: StatsStore,
        text: str,
        read: Callable[[str], str] = input,
        write: Callable[[str], None] = print,
        low: float = 1.0,
        high: float = 10.0,
    ):
        self.path, self.plan, self.store, self.text = path, plan, store, text
        self.read, self.write = read, write
        self.low, self.high = low, high
        self.dirty = False
        self.done = False

    # --- plumbing

    def ask(self, prompt: str, default: str = "") -> str:
        suffix = f" [{default}]" if default else ""
        answer = self.read(f"{prompt}{suffix}: ").strip()
        return answer or default

    def confirm(self, prompt: str) -> bool:
        return self.ask(f"{prompt} [y/N]").lower() in {"y", "yes"}

    def show(self) -> None:
        self.write(render_forecast(self.plan, self.store, self.low, self.high))

    def resolve(self, token: str) -> str:
        names = list(self.plan.datasets)
        if token in names:
            return token
        matches = [n for n in names if n.startswith(token)]
        if len(matches) == 1:
            return matches[0]
        if not matches:
            raise PlanError(f"no dataset {token!r} (have: {', '.join(names) or 'none'})")
        raise PlanError(f"{token!r} is ambiguous: {', '.join(matches)}")

    def parse_mix(self, tokens: list[str]) -> dict[str, float]:
        mix = {}
        for token in tokens:
            name, sep, value = token.replace(":", "=").partition("=")
            if not sep:
                raise PlanError(f"{token!r}: write it as name=percent")
            mix[self.resolve(name)] = parse_float(value)
        if not mix:
            raise PlanError("name at least one dataset, e.g. plenum=90 hifitts2=10")
        return mix

    def phase_number(self, token: str | None, default: int | None = None) -> int:
        if token is None:
            if default is None:
                raise PlanError("which phase? (a number, 1-based)")
            return default
        return parse_int(token)

    # --- the loop

    def run(self) -> None:
        self.write("Curriculum planner. `help` lists the commands.\n")
        self.show()
        while not self.done:
            try:
                line = self.read("\n> ").strip()
            except EOFError:
                self.write("")
                break
            if not line:
                continue
            self.execute(line)
        if self.dirty:
            self.write("(unsaved changes were not written)")

    def execute(self, line: str) -> None:
        try:
            words = shlex.split(line)
        except ValueError:
            words = line.split()
        command, args = words[0].lower(), words[1:]
        handler = getattr(self, f"cmd_{command}", None)
        if handler is None:
            self.write(f"unknown command {command!r}; `help` lists them")
            return
        try:
            handler(args)
        except PlanError as error:
            self.write(f"! {error}")
        except EOFError:  # input ended in the middle of a question
            self.done = True

    def changed(self) -> None:
        self.dirty = True
        self.show()

    # --- commands

    def cmd_help(self, args: list[str]) -> None:
        self.write(HELP)

    def cmd_show(self, args: list[str]) -> None:
        self.show()

    def cmd_quit(self, args: list[str]) -> None:
        if self.dirty and not self.confirm("discard unsaved changes?"):
            return
        self.dirty = False
        self.done = True

    cmd_q = cmd_exit = cmd_quit

    def cmd_add(self, args: list[str]) -> None:
        text = " ".join(args) if args else self.ask("dataset folder (paste the path)")
        if not text:
            return
        found = discover_dataset(clean_path(text), set(self.plan.datasets))
        self.write(f"  train: {found.train}  ({found.stage})")
        self.write(f"  valid: {found.valid or 'none'}")
        if len(found.present) > 2:
            self.write(f"  in the folder: {', '.join(found.present)}")
        for note in found.notes:
            self.write(f"  ! {note}")
        train, valid = (
            config_path_for(found.train),
            config_path_for(found.valid) if found.valid else "",
        )
        self.write("  reading the manifest...")
        stats = self.store.get(train)
        if stats:
            hours = f", {stats.hours:,.1f} h per epoch" if stats.hours is not None else ""
            self.write(f"  {stats.rows:,} rows{hours}")
        name = self.ask("name", found.name)
        if not valid:
            weight = 0.0
        else:
            weight = parse_float(
                self.ask("valid_weight (relative share of validation)", str(DEFAULT_VALID_WEIGHT))
            )
        self.plan.add_dataset(name, DatasetEntry(train, valid, weight))
        answer = self.ask("share of each phase, e.g. `2=5 3=5` (percent; blank = not used yet)")
        for token in answer.split():
            number, _, percent = token.partition("=")
            self.plan.set_share(self.phase_number(number), name, parse_float(percent))
        self.changed()

    def cmd_rm(self, args: list[str]) -> None:
        name = self.resolve(args[0] if args else self.ask("dataset to remove"))
        emptied = self.plan.remove_dataset(name)
        if emptied:
            self.write(f"! phase(s) {emptied} have no dataset left: set their mix")
        self.changed()

    def cmd_steps(self, args: list[str]) -> None:
        if len(args) != 2:
            raise PlanError("usage: steps <phase> <n>")
        self.plan.phase(self.phase_number(args[0])).steps = parse_int(args[1])
        self.changed()

    def cmd_share(self, args: list[str]) -> None:
        if len(args) != 3:
            raise PlanError("usage: share <phase> <dataset> <percent>")
        self.plan.set_share(self.phase_number(args[0]), self.resolve(args[1]), parse_float(args[2]))
        self.changed()

    def cmd_mix(self, args: list[str]) -> None:
        if len(args) < 2:
            raise PlanError("usage: mix <phase> name=percent ...")
        self.plan.set_mix(self.phase_number(args[0]), self.parse_mix(args[1:]))
        self.changed()

    def cmd_phase(self, args: list[str]) -> None:
        sub = args[0] if args else ""
        if sub == "add":
            steps = parse_int(args[1] if len(args) > 1 else self.ask("steps"))
            previous = self.plan.phases[-1].mix if self.plan.phases else {}
            same = " ".join(f"{n}={w:.4g}" for n, w in previous.items())
            answer = self.ask("mix (name=percent ...)", same)
            self.plan.phases.append(PhaseEntry(steps, {}))
            try:
                self.plan.set_mix(len(self.plan.phases), self.parse_mix(answer.split()))
            except PlanError:
                self.plan.phases.pop()
                raise
            self.changed()
        elif sub == "rm" and len(args) == 2:
            number = self.phase_number(args[1])
            self.plan.phase(number)
            del self.plan.phases[number - 1]
            self.changed()
        else:
            raise PlanError("usage: phase add [steps] | phase rm <phase>")

    def cmd_batch(self, args: list[str]) -> None:
        self.plan.batch_override = parse_int(args[0]) if args else None
        self.show()

    def cmd_world(self, args: list[str]) -> None:
        if len(args) != 1:
            raise PlanError("usage: world <ranks>")
        self.plan.world_size = parse_int(args[0])
        self.plan.batch_override = None
        self.show()

    def cmd_rows(self, args: list[str]) -> None:
        if len(args) not in (2, 3):
            raise PlanError("usage: rows <dataset> <rows> [hours per epoch]")
        entry = self.plan.datasets[self.resolve(args[0])]
        hours = parse_float(args[2]) if len(args) == 3 else None
        self.store.set_manual(entry.train_jsonl, parse_int(args[1]), hours)
        self.show()

    def cmd_valid(self, args: list[str]) -> None:
        if len(args) != 2:
            raise PlanError("usage: valid <dataset> <weight>")
        self.plan.datasets[self.resolve(args[0])].valid_weight = parse_float(args[1])
        self.changed()

    def cmd_ckpt(self, args: list[str]) -> None:
        if len(args) != 1 or args[0] not in {"on", "off"}:
            raise PlanError("usage: ckpt on|off")
        self.plan.checkpoint_at_phase_end = args[0] == "on"
        self.changed()

    def cmd_warn(self, args: list[str]) -> None:
        if len(args) != 2:
            raise PlanError("usage: warn <low> <high>")
        self.low, self.high = parse_float(args[0]), parse_float(args[1])
        self.show()

    def cmd_solve(self, args: list[str]) -> None:
        plan = self.plan
        number = self.phase_number(args[0] if args else None, len(plan.phases) or 1)
        if number > len(plan.phases) + 1 or number < 1:
            raise PlanError(f"phase {number}: the plan has {len(plan.phases)}")
        rows = plan.rows(self.store)
        known = [n for n in plan.datasets if rows[n]]
        if not known:
            raise PlanError("no dataset has a known size yet (`rows <dataset> <rows>`)")
        before = plan.forecast(self.store)[: number - 1]
        drawn = {n: sum(f.samples[n] for f in before) for n in rows}
        self.write(
            f"Target epochs (cumulative) at the end of phase {number}. Blank = the dataset draws "
            "nothing in this phase."
        )
        targets = {}
        for name in known:
            reached = (before[-1].cumulative[name] if before else 0.0) or 0.0
            answer = self.ask(f"  {name} ({rows[name]:,} rows, at {epochs_text(reached)} now)")
            if answer:
                targets[name] = parse_float(answer)
        steps_text = self.ask("steps (blank = as many as the targets need)")
        steps = parse_int(steps_text) if steps_text else None
        solved_steps, mix = solve_phase(rows, targets, plan.effective_batch, drawn, steps)
        trial = copy.deepcopy(plan)
        entry = PhaseEntry(solved_steps, {n: w * 100 for n, w in mix.items()})
        if number > len(trial.phases):
            trial.phases.append(entry)
        else:
            trial.phases[number - 1] = entry
        self.write(f"\nProposed phase {number}:")
        self.write(render_forecast(trial, self.store, self.low, self.high))
        if self.confirm("apply"):
            self.plan = trial
            self.changed()

    def cmd_diff(self, args: list[str]) -> None:
        after = render_config(self.text, self.plan)
        self.write(unified_diff(self.text, after, self.path.name) or "(no change)")

    def cmd_save(self, args: list[str]) -> None:
        target = Path(args[0]) if args else self.path
        after = render_config(self.text, self.plan)
        diff = unified_diff(self.text, after, self.path.name)
        if not diff:
            self.write("(no change)")
            return
        self.write(diff)
        if not self.confirm(f"write {target}"):
            return
        target.write_text(after)
        if target == self.path:
            self.text, self.dirty = after, False
        self.write(f"wrote {target}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("config", type=Path, help="training yaml")
    parser.add_argument("--show", action="store_true", help="print the forecast and exit")
    parser.add_argument("--world-size", type=int, default=DEFAULT_WORLD_SIZE, help="ranks (GPUs)")
    parser.add_argument(
        "--batch", type=int, help="effective batch size (default: batch_size x accum x ranks)"
    )
    parser.add_argument(
        "--warn-low", type=float, default=1.0, help="flag a dataset below this many epochs"
    )
    parser.add_argument("--warn-high", type=float, default=10.0, help="flag one above this many")
    parser.add_argument("--cache", type=Path, default=CACHE_PATH, help="manifest size cache")
    args = parser.parse_args(argv)

    text = args.config.read_text()
    plan = Plan.from_config(args.config, text)
    plan.world_size, plan.batch_override = args.world_size, args.batch
    store = StatsStore(args.cache, plan.max_duration_sec)
    planner = Planner(args.config, plan, store, text, low=args.warn_low, high=args.warn_high)
    if args.show:
        planner.show()
        return 0
    planner.run()
    return 0


if __name__ == "__main__":
    sys.exit(main())
