"""The curriculum planner: epoch forecast, its inverse, folder discovery, config rewrite, REPL."""

import json
from collections import Counter
from pathlib import Path

import pytest

from training.args import load_args
from training.curriculum import MixSchedule, parse_curriculum
from training.curriculum_plan import (
    DatasetEntry,
    PhaseEntry,
    Plan,
    PlanError,
    StatsStore,
    discover_dataset,
    forecast,
    render_config,
    scale_others,
    solve_phase,
    unified_diff,
)
from training.plan_curriculum import Planner, parse_int

TEMPLATE = Path(__file__).parent.parent / "configs" / "finetune_hebrew_curriculum.yaml"


def write_manifest(path: Path, rows: int, duration: float = 10.0) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    row = {
        "path": "a.wav",
        "start": 0.0,
        "duration": duration,
        "words": [{"word": "x", "start": 0, "end": 1}],
    }
    path.write_text("".join(json.dumps(row) + "\n" for _ in range(rows)))
    return path


# --- the arithmetic ------------------------------------------------------------------------


def test_forecast_matches_the_schedule_the_trainer_runs():
    datasets = {
        n: {"train_jsonl": "x", "valid_jsonl": "y", "valid_weight": 1} for n in ("a", "b", "c")
    }
    phases = [{"steps": 40, "mix": {"a": 1}}, {"steps": 60, "mix": {"a": 0.7, "b": 0.2, "c": 0.1}}]
    slots, micro = 8, 4
    schedule = MixSchedule(parse_curriculum(datasets, phases), slots)
    drawn = Counter()
    for step in range(100):
        drawn.update(schedule.slots(step))
    rows = {"a": 1000, "b": 500, "c": 250}
    result = forecast([(p["steps"], p["mix"]) for p in phases], rows, batch=slots * micro)
    for name, size in rows.items():
        simulated = drawn[name] * micro  # samples
        assert sum(p.samples[name] for p in result) == pytest.approx(
            simulated, abs=micro
        )  # one micro-batch
        assert result[-1].cumulative[name] == pytest.approx(simulated / size, abs=micro / size)


def test_forecast_accumulates_across_phases_and_keeps_unknown_sizes_unknown():
    result = forecast([(10, {"a": 1}), (10, {"b": 1}), (10, {"a": 1})], {"a": 320, "b": None}, 32)
    assert [p.cumulative["a"] for p in result] == [1.0, 1.0, 2.0]
    assert result[1].gained["b"] is None
    assert result[0].start_step == 1 and result[1].start_step == 11 and result[2].end_step == 30


def test_solve_phase_hits_the_targets():
    rows = {"a": 1000, "b": 400}
    steps, mix = solve_phase(rows, {"a": 3, "b": 5}, batch=50, drawn_before={})
    result = forecast([(steps, mix)], rows, 50)[0]
    assert result.cumulative["a"] == pytest.approx(3, rel=0.01)
    assert result.cumulative["b"] == pytest.approx(5, rel=0.01)


def test_solve_phase_counts_what_earlier_phases_drew():
    rows = {"a": 1000}
    steps, mix = solve_phase(rows, {"a": 3}, batch=100, drawn_before={"a": 2000})
    assert (steps, mix) == (10, {"a": 1.0})  # one epoch (1000 samples) still to go
    with pytest.raises(PlanError, match="already reached"):
        solve_phase(rows, {"a": 1}, batch=100, drawn_before={"a": 2000})


def test_scale_others_keeps_proportions():
    assert scale_others({"a": 60, "b": 30, "c": 10}, "c", 25) == {"a": 50, "b": 25, "c": 25}
    assert scale_others({"a": 1}, "b", 10) == {"a": 90, "b": 10}
    assert scale_others({"a": 1, "b": 1}, "a", 100) == {"a": 100}
    with pytest.raises(PlanError):
        scale_others({"a": 1}, "a", 50)


def test_parse_int_accepts_suffixes():
    assert [parse_int(t) for t in ("15000", "15k", "1.5m", "2,500")] == [
        15000,
        15000,
        1_500_000,
        2500,
    ]
    with pytest.raises(PlanError):
        parse_int("0.5")


# --- sizes ---------------------------------------------------------------------------------


def test_stats_scan_counts_rows_and_crops_hours(tmp_path):
    path = write_manifest(tmp_path / "t.jsonl", rows=4, duration=40.0)
    store = StatsStore(tmp_path / "cache.json", max_duration_sec=30.0)
    stats = store.get(str(path))
    assert stats.rows == 4
    assert stats.hours == pytest.approx(4 * 30 / 3600)  # each row is cut to 30 s
    path.unlink()  # the manifest is on another machine now: the cache answers
    again = StatsStore(tmp_path / "cache.json", max_duration_sec=30.0).get(str(path))
    assert (again.rows, again.source) == (4, "cached")


def test_manual_sizes_stand_in_for_missing_manifests(tmp_path):
    store = StatsStore(tmp_path / "cache.json", 30.0)
    assert store.get("/nowhere/train.jsonl") is None
    store.set_manual("/nowhere/train.jsonl", 1234, None)
    assert StatsStore(tmp_path / "cache.json", 30.0).get("/nowhere/train.jsonl").rows == 1234


# --- discovery -----------------------------------------------------------------------------


def test_discovery_prefers_the_most_processed_stage(tmp_path):
    folder = tmp_path / "plenum"
    for name in ("train", "valid", "train_diac", "train_aligned", "valid_aligned"):
        write_manifest(folder / f"{name}.jsonl", 2)
    found = discover_dataset(folder, taken={"plenum"})
    assert found.train.name == "train_aligned.jsonl" and found.valid.name == "valid_aligned.jsonl"
    assert found.stage == "aligned" and found.name == "plenum_2"


def test_discovery_from_a_pasted_file_or_nested_folder_and_missing_valid(tmp_path):
    write_manifest(tmp_path / "ds" / "out" / "train.jsonl", 2)
    found = discover_dataset(tmp_path / "ds")
    assert found.train == tmp_path / "ds" / "out" / "train.jsonl" and found.valid is None
    assert any("no validation" in n for n in found.notes)
    assert discover_dataset(found.train).train == found.train
    with pytest.raises(PlanError, match="no train"):
        discover_dataset(tmp_path)


def test_discovery_flags_rows_the_loader_cannot_use(tmp_path):
    (tmp_path / "train.jsonl").write_text('{"path": "a.wav", "duration": 3}\n')
    assert any("words" in n for n in discover_dataset(tmp_path).notes)


# --- the config rewrite --------------------------------------------------------------------


def test_rewrite_changes_only_the_curriculum_and_the_loader_reads_it_back(tmp_path):
    text = TEMPLATE.read_text()
    plan = Plan.from_config(TEMPLATE, text)
    plan.phases[1].steps = 100_000
    plan.set_share(2, "libritts_r", 20)
    plan.add_dataset("new_he", DatasetEntry("data/new/train.jsonl", "data/new/valid.jsonl", 0.3))
    plan.phases.append(PhaseEntry(5000, {"new_he": 100}))
    out = render_config(text, plan)

    assert out[: out.index("\ndata:")] == text[: text.index("\ndata:")]  # everything before
    assert out[out.index("\nflow:") :] == text[text.index("\nflow:") :]  # and after, as it was
    assert "# Hebrew" in out and "# English, to keep it speaking English" in out  # dataset notes
    assert "# Validate and checkpoint at each phase end too" in out
    assert "sample_sentences:" in out and out.count("lr: 2e-4") == 1

    path = tmp_path / "c.yaml"
    path.write_text(out)
    curriculum = load_args(path).data.resolve_curriculum()
    assert curriculum.boundaries == (15000, 115000, 120000)
    assert curriculum.phases[1].mix["libritts_r"] == pytest.approx(0.2)
    assert curriculum.phases[2].mix == {"new_he": 1.0}
    assert Plan.from_config(path).raw_phases() == plan.raw_phases()


def test_rewrite_of_an_unchanged_plan_is_the_same_file(tmp_path):
    text = TEMPLATE.read_text()
    assert render_config(text, Plan.from_config(TEMPLATE, text)) == text  # comments included


def test_rewrite_of_one_edit_touches_only_that_line(tmp_path):
    text = TEMPLATE.read_text()
    plan = Plan.from_config(TEMPLATE, text)
    plan.phases[0].steps = 12_000
    changed = [
        line
        for line in unified_diff(text, render_config(text, plan), "c").splitlines()
        if line.startswith(("+", "-")) and not line.startswith(("+++", "---"))
    ]
    assert len(changed) == 2 and "12000" in changed[1]


def test_a_plain_config_becomes_a_one_dataset_curriculum(tmp_path):
    text = (
        "model_config: x.yaml\n"
        "data:\n"
        "  train_jsonl: data/he/train.jsonl\n"
        "  valid_jsonl: data/he/valid.jsonl\n"
        "  max_duration_sec: 20.0\n"
        "batch_size: 8\n"
        "max_steps: 5000  # all of it\n"
    )
    plan = Plan.from_config(tmp_path / "c.yaml", text)
    assert [p.steps for p in plan.phases] == [5000] and plan.max_duration_sec == 20.0
    plan.add_dataset("en", DatasetEntry("data/en/train.jsonl", "data/en/valid.jsonl", 0.2))
    plan.phases.append(PhaseEntry(3000, {"he": 80, "en": 20}))
    out = render_config(text, plan)
    assert "max_steps: 8000  # all of it" in out
    assert "train_jsonl: data/he/train.jsonl" in out and "\n  train_jsonl:" not in out
    path = tmp_path / "c.yaml"
    path.write_text(out)
    assert load_args(path).data.resolve_curriculum().total_steps == 8000


def test_an_invalid_plan_is_never_written():
    text = TEMPLATE.read_text()
    plan = Plan.from_config(TEMPLATE, text)
    plan.remove_dataset("plenum")
    with pytest.raises(PlanError):
        render_config(text, plan)


# --- the interactive session ---------------------------------------------------------------


def session(tmp_path, answers, config=TEMPLATE):
    out, script = [], iter(answers)
    text = config.read_text()
    plan = Plan.from_config(config, text)
    store = StatsStore(None, plan.max_duration_sec)
    for entry in plan.datasets.values():
        store.set_manual(entry.train_jsonl, 100_000, None)
    path = tmp_path / "cfg.yaml"
    path.write_text(text)

    def read(prompt):
        out.append(prompt)
        try:
            return next(script)
        except StopIteration:
            raise EOFError from None

    planner = Planner(path, plan, store, text, read=read, write=out.append)
    planner.run()
    return planner, "\n".join(out), path


def test_session_adds_a_dataset_from_its_folder_and_saves(tmp_path):
    write_manifest(tmp_path / "ivrit" / "train_aligned.jsonl", 3)
    write_manifest(tmp_path / "ivrit" / "valid_aligned.jsonl", 2)
    planner, out, path = session(
        tmp_path, [f"add {tmp_path / 'ivrit'}", "", "0.2", "2=10", "save", "y", "quit"]
    )
    assert "train_aligned.jsonl" in out and "3 rows" in out
    saved = load_args(path).data.resolve_curriculum()
    assert saved.datasets["ivrit"].train_jsonl.endswith("train_aligned.jsonl")
    assert saved.phases[1].mix["ivrit"] == pytest.approx(0.1)
    assert saved.phases[1].mix["plenum"] == pytest.approx(0.9 * 0.9)
    assert not planner.dirty


def test_session_solve_from_target_epochs(tmp_path):
    planner, out, _ = session(tmp_path, ["solve 2", "10", "2", "2", "", "y", "quit", "y"])
    # 3 datasets of 100k rows at batch 64. Phase 1 drew 960k samples = 9.6 epochs of plenum.
    # Targets at the end of phase 2: plenum 10 (40k more), each English set 2 (200k) = 440k.
    assert planner.plan.phases[1].steps == 6875  # 440000 / 64
    assert planner.plan.phases[1].mix["plenum"] == pytest.approx(100 * 40_000 / 440_000, rel=1e-3)
    assert "Proposed phase 2" in out


def test_session_reports_errors_and_survives(tmp_path):
    _, out, _ = session(tmp_path, ["steps 9 100", "bogus", "share 1 nope 5", "quit"])
    assert "there is no phase 9" in out and "unknown command" in out and "no dataset 'nope'" in out


def test_show_flags_both_directions(tmp_path):
    _, out, _ = session(tmp_path, ["quit"])
    assert "over 10" in out  # plenum rolls over ~139 times on a 100k-row manifest
