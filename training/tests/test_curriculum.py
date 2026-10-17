"""The training curriculum: config parsing, the per-step mix, validation allocation."""

import itertools
from collections import Counter

import pytest
import yaml

from training.args import DataArgs, load_args
from training.curriculum import (
    CurriculumLoader,
    MixSchedule,
    apportion,
    combine_validation,
    count_lines,
    parse_curriculum,
    plan_validation,
)

# The sketch this feature came from: DS1 joins at S1 and takes over at S2, DS2 leaves at S2.
DATASETS = {
    "ds1": {"train_jsonl": "a.jsonl", "valid_jsonl": "av.jsonl", "valid_weight": 0.2},
    "ds2": {"train_jsonl": "b.jsonl", "valid_jsonl": "bv.jsonl", "valid_weight": 0.4},
    "ds3": {"train_jsonl": "c.jsonl", "valid_jsonl": "cv.jsonl", "valid_weight": 0.4},
}
PHASES = [
    {"steps": 100, "mix": {"ds2": 50, "ds3": 50}},
    {"steps": 150, "mix": {"ds1": 20, "ds2": 40, "ds3": 40}},
    {"steps": 50, "mix": {"ds1": 60, "ds3": 40}},
]


def curriculum(**overrides):
    return parse_curriculum(overrides.get("datasets", DATASETS), overrides.get("phases", PHASES))


def error_of(**overrides) -> str:
    with pytest.raises(ValueError) as info:
        curriculum(**overrides)
    return str(info.value)


# --- parsing -----------------------------------------------------------------------------


def test_phases_normalize_and_report_their_boundaries():
    c = curriculum()
    assert c.total_steps == 300
    assert c.boundaries == (100, 250, 300)
    assert c.phases[0].mix == {"ds2": 0.5, "ds3": 0.5}
    assert c.phases[1].mix == pytest.approx({"ds1": 0.2, "ds2": 0.4, "ds3": 0.4})
    assert "ds1" not in c.phases[0].mix  # a 0% dataset is simply absent


def test_phase_index_follows_the_boundaries():
    c = curriculum()
    assert [c.phase_index(s) for s in (0, 99, 100, 249, 250, 299, 300, 10_000)] == [
        0, 0, 1, 1, 2, 2, 2, 2,
    ]  # fmt: skip


def test_zero_weight_may_be_written_out():
    phases = [{"steps": 5, "mix": {"ds1": 0, "ds2": 1}}]
    assert curriculum(phases=phases).phases[0].mix == {"ds2": 1.0}


@pytest.mark.parametrize(
    ("phases", "message"),
    [
        ([], "non-empty list"),
        ([{"steps": 0, "mix": {"ds1": 1}}], "positive integer"),
        ([{"steps": 1.5, "mix": {"ds1": 1}}], "positive integer"),
        ([{"steps": True, "mix": {"ds1": 1}}], "positive integer"),
        ([{"steps": 5, "mix": {"nope": 1}}], "not in data.datasets"),
        ([{"steps": 5, "mix": {"ds1": -1}}], ">= 0"),
        ([{"steps": 5, "mix": {"ds1": float("nan")}}], ">= 0"),
        ([{"steps": 5, "mix": {"ds1": "half"}}], "must be a number"),
        ([{"steps": 5, "mix": {"ds1": 0}}], "no positive weight"),
        ([{"steps": 5, "mix": {}}], "non-empty mapping"),
        ([{"steps": 5}], "missing"),
        ([{"steps": 5, "mix": {"ds1": 1}, "lr": 1}], "unknown keys"),
    ],
)
def test_bad_phases_are_rejected(phases, message):
    assert message in error_of(phases=phases)


def test_valid_weight_must_be_explicit():
    datasets = {**DATASETS, "ds1": {"train_jsonl": "a.jsonl", "valid_jsonl": "av.jsonl"}}
    assert "valid_weight is required" in error_of(datasets=datasets)


def test_a_validated_dataset_needs_its_manifest():
    datasets = {**DATASETS, "ds1": {"train_jsonl": "a.jsonl", "valid_weight": 1}}
    assert "valid_jsonl is required" in error_of(datasets=datasets)


def test_a_trained_dataset_needs_its_manifest():
    datasets = {**DATASETS, "ds2": {"valid_jsonl": "bv.jsonl", "valid_weight": 1}}
    assert "train_jsonl is required" in error_of(datasets=datasets)


def test_a_dataset_that_is_never_used_is_rejected():
    datasets = {**DATASETS, "extra": {"train_jsonl": "x.jsonl", "valid_weight": 0}}
    assert "never trained on" in error_of(datasets=datasets)


def test_a_validation_only_dataset_is_allowed():
    datasets = {**DATASETS, "held_out": {"valid_jsonl": "h.jsonl", "valid_weight": 1}}
    c = curriculum(datasets=datasets)
    assert "held_out" not in c.train_names
    assert "held_out" in c.valid_names


def test_unknown_dataset_keys_and_bad_names_are_rejected():
    datasets = {**DATASETS, "ds1": {**DATASETS["ds1"], "lr": 1}}
    assert "unknown keys" in error_of(datasets=datasets)
    datasets = {**DATASETS, "a/b": DATASETS["ds1"]}
    assert "must match" in error_of(datasets=datasets)


# --- the mix -----------------------------------------------------------------------------


def run_schedule(c, slots, steps=None, start=0):
    schedule = MixSchedule(c, slots)
    schedule.fast_forward(start)
    return [schedule.slots(s) for s in range(start, steps if steps is not None else c.total_steps)]


@pytest.mark.parametrize("slots", [1, 2, 8, 64])
def test_each_phase_hits_its_ratio_exactly(slots):
    c = curriculum()
    per_step = run_schedule(c, slots)
    start = 0
    for phase, end in zip(c.phases, c.boundaries, strict=True):
        counts = Counter(itertools.chain.from_iterable(per_step[start:end]))
        total = (end - start) * slots
        for name, share in phase.mix.items():
            # Within one slot of exact, however short the phase is.
            assert abs(counts[name] - share * total) <= 1, (name, counts, total)
        assert set(counts) == set(phase.mix)  # a 0% dataset is never drawn
        start = end


def test_the_ratio_holds_over_short_windows_too():
    # 20% of 8 slots is 1.6 a step: never wildly off in any window of steps.
    c = curriculum(phases=[{"steps": 400, "mix": {"ds1": 0.2, "ds2": 0.8}}])
    per_step = run_schedule(c, 8)
    for first in range(0, 400, 25):
        window = list(itertools.chain.from_iterable(per_step[first : first + 25]))
        assert abs(window.count("ds1") - 0.2 * len(window)) <= 1


def test_the_schedule_is_deterministic():
    c = curriculum()
    assert run_schedule(c, 8) == run_schedule(c, 8)


@pytest.mark.parametrize("resume_at", [0, 1, 99, 100, 101, 180, 249, 250, 299])
def test_resuming_continues_the_same_sequence(resume_at):
    c = curriculum()
    full = run_schedule(c, 8)
    resumed = run_schedule(c, 8, start=resume_at)
    assert resumed == full[resume_at:]


def test_the_schedule_must_be_read_in_order():
    schedule = MixSchedule(curriculum(), 4)
    schedule.slots(0)
    with pytest.raises(ValueError, match="asked for step 5"):
        schedule.slots(5)


def test_a_phase_starts_fresh_whatever_the_previous_one_left_over():
    c = curriculum(
        phases=[
            {"steps": 1, "mix": {"ds1": 1, "ds2": 1, "ds3": 1}},  # 3 slots, perfectly even
            {"steps": 1, "mix": {"ds1": 1, "ds2": 1, "ds3": 1}},
        ]
    )
    first, second = run_schedule(c, 2)
    assert first == second


# --- apportioning ------------------------------------------------------------------------


def test_apportion_is_proportional_and_sums_to_total():
    assert apportion({"a": 0.2, "b": 0.4, "c": 0.4}, 50) == {"a": 10, "b": 20, "c": 20}
    out = apportion({"a": 1, "b": 1, "c": 1}, 10)
    assert sum(out.values()) == 10 and sorted(out.values()) == [3, 3, 4]


def test_apportion_gives_a_dataset_that_is_full_its_cap_and_shares_the_rest():
    out = apportion({"a": 0.2, "b": 0.4, "c": 0.4}, 50, caps={"b": 5})
    assert out["b"] == 5
    assert out["a"] + out["c"] == 45
    assert out["a"] == 15 and out["c"] == 30  # 1:2, the relative weights of a and c


def test_apportion_cascades_when_redistribution_fills_another_dataset():
    out = apportion({"a": 1, "b": 1, "c": 1}, 30, caps={"a": 2, "b": 12})
    assert out == {"a": 2, "b": 12, "c": 16}


def test_apportion_returns_less_when_every_dataset_is_full():
    out = apportion({"a": 1, "b": 1}, 100, caps={"a": 3, "b": 4})
    assert out == {"a": 3, "b": 4}


def test_apportion_guarantees_the_minimum_and_skips_zero_weights():
    out = apportion({"a": 1000, "b": 1, "c": 0}, 10, minimum=1)
    assert out["b"] == 1 and out["c"] == 0 and sum(out.values()) == 10


def test_apportion_reports_an_impossible_minimum():
    with pytest.raises(ValueError, match="cannot give each of 3"):
        apportion({"a": 1, "b": 1, "c": 1}, 2, minimum=1)


@pytest.mark.parametrize("seed", range(20))
def test_apportion_invariants_on_arbitrary_input(seed):
    import random

    rng = random.Random(seed)
    names = [f"d{i}" for i in range(rng.randint(1, 6))]
    weights = {n: rng.random() + 0.01 for n in names}
    caps = {n: rng.randint(1, 30) for n in names if rng.random() < 0.6}
    total = rng.randint(len(names), 80)
    out = apportion(weights, total, caps, minimum=1)
    assert all(out[n] >= 1 for n in names)
    assert all(out[n] <= caps.get(n, total) for n in names)
    assert sum(out.values()) == min(total, sum(caps.get(n, total) for n in names))


# --- validation plan ---------------------------------------------------------------------


def test_validation_splits_the_budget_by_valid_weight():
    c = curriculum()
    counts = {"ds1": 10_000, "ds2": 10_000, "ds3": 10_000}
    plan = plan_validation(c, counts, batch_size=8, world_size=2, num_valid_batches=50)
    assert plan == {"ds1": 10, "ds2": 20, "ds3": 20}


def test_a_small_validation_set_hands_its_share_to_the_others():
    c = curriculum()
    # ds2 holds 3 batches per rank (96 rows / 4 ranks / batch 8 = 3), wanted 20.
    counts = {"ds1": 10_000, "ds2": 96, "ds3": 10_000}
    plan = plan_validation(c, counts, batch_size=8, world_size=4, num_valid_batches=50)
    assert plan["ds2"] == 3
    assert sum(plan.values()) == 50
    assert plan["ds1"] * 2 == pytest.approx(plan["ds3"], abs=1)  # 0.2 : 0.4 survives


def test_a_dataset_outside_the_benchmark_is_not_planned():
    datasets = {**DATASETS, "ds1": {**DATASETS["ds1"], "valid_weight": 0}}
    c = curriculum(datasets=datasets)
    counts = {"ds2": 10_000, "ds3": 10_000}
    assert plan_validation(c, counts, 8, 1, 50) == {"ds2": 25, "ds3": 25}


def test_no_validated_dataset_means_no_validation():
    datasets = {n: {**d, "valid_weight": 0} for n, d in DATASETS.items()}
    assert plan_validation(curriculum(datasets=datasets), {}, 8, 1, 50) == {}


def test_validation_errors_name_the_dataset_and_the_fix():
    c = curriculum()
    with pytest.raises(ValueError, match="at least one batch"):
        plan_validation(c, {}, 8, 1, 2)
    counts = {"ds1": 10_000, "ds2": 5, "ds3": 10_000}
    with pytest.raises(ValueError, match="ds2.*not one full batch"):
        plan_validation(c, counts, 8, 1, 50)


def test_the_plan_is_the_same_for_every_rank_count_it_is_given():
    c = curriculum()
    counts = {"ds1": 1001, "ds2": 1003, "ds3": 1007}  # not divisible by the world size
    assert plan_validation(c, counts, 4, 8, 20) == plan_validation(c, counts, 4, 8, 20)


def test_combine_validation_weights_datasets_by_their_batches():
    per = {"a": {"loss": 1.0, "eos": 0.0}, "b": {"loss": 4.0, "eos": 3.0}}
    out = combine_validation(per, {"a": 30, "b": 10})
    assert out["loss"] == pytest.approx(1.75)
    assert out["eos"] == pytest.approx(0.75)
    assert out["a/loss"] == 1.0 and out["b/eos"] == 3.0


def test_count_lines_counts_every_line(tmp_path):
    path = tmp_path / "m.jsonl"
    path.write_text('{"a": 1}\n\n{"a": 2}\n')
    assert count_lines(str(path)) == 3


# --- the loader --------------------------------------------------------------------------


class FakeStreams:
    """Endless per-dataset streams of (name, n); remembers what was opened."""

    def __init__(self, sizes):
        self.sizes = sizes
        self.opened = []

    def __call__(self, name):
        self.opened.append(name)
        return (iter((name, i) for i in itertools.count()), self.sizes[name])


def test_ranks_split_a_steps_slots_and_draw_every_dataset_in_ratio():
    c = curriculum()
    world, accum = 2, 4
    loaders = []
    for rank in range(world):
        schedule = MixSchedule(c, accum * world)
        schedule.fast_forward(100)  # the start of phase 2: 20 / 40 / 40
        streams = FakeStreams({"ds1": 80, "ds2": 80, "ds3": 80})
        loaders.append(CurriculumLoader(schedule, streams, rank, world, 8))
    drawn = Counter()
    for step in range(100, 250):
        for micro in range(accum):
            for loader in loaders:
                drawn[loader.next_batch(step, micro)[0]] += 1
    # Every rank walks the same schedule, so together they used exactly the phase's ratio.
    total = sum(drawn.values())
    assert total == 150 * accum * world
    assert drawn["ds1"] / total == pytest.approx(0.2, abs=0.002)
    assert drawn["ds2"] / total == pytest.approx(0.4, abs=0.002)
    assert drawn["ds3"] / total == pytest.approx(0.4, abs=0.002)


def test_a_loader_started_mid_run_matches_one_that_ran_from_the_start():
    c = curriculum()
    streams = FakeStreams({"ds1": 80, "ds2": 80, "ds3": 80})
    fresh = CurriculumLoader(MixSchedule(c, 8), streams, 0, 1, 8)
    names = [[fresh.next_batch(s, m)[0] for m in range(8)] for s in range(300)]
    schedule = MixSchedule(c, 8)
    schedule.fast_forward(180)
    resumed = CurriculumLoader(schedule, FakeStreams({"ds1": 80, "ds2": 80, "ds3": 80}), 0, 1, 8)
    assert [[resumed.next_batch(s, m)[0] for m in range(8)] for s in range(180, 300)] == names[180:]


def test_datasets_are_opened_when_first_needed():
    c = curriculum()
    streams = FakeStreams({"ds1": 80, "ds2": 80, "ds3": 80})
    loader = CurriculumLoader(MixSchedule(c, 4), streams, 0, 1, 8)
    for step in range(100):  # phase 1 never touches ds1
        for micro in range(4):
            loader.next_batch(step, micro)
    assert sorted(streams.opened) == ["ds2", "ds3"]
    for micro in range(4):
        loader.next_batch(100, micro)
    assert "ds1" in streams.opened


def test_a_dataset_keeps_its_place_across_a_pause():
    # ds2 is on in phases 1 and 2 and off in 3; ds1 off, on, on: nothing restarts.
    c = curriculum()
    loader = CurriculumLoader(
        MixSchedule(c, 4), FakeStreams({"ds1": 8, "ds2": 8, "ds3": 8}), 0, 1, 8
    )
    seen = {"ds1": [], "ds2": [], "ds3": []}
    for step in range(300):
        for micro in range(4):
            name, index = loader.next_batch(step, micro)
            seen[name].append(index)
    for indices in seen.values():
        assert indices == list(range(len(indices)))  # one unbroken stream per dataset


def test_epochs_count_rollovers():
    c = curriculum(phases=[{"steps": 10, "mix": {"ds1": 1}}])
    loader = CurriculumLoader(MixSchedule(c, 2), FakeStreams({"ds1": 40}), 0, 1, 8)
    for step in range(10):
        for micro in range(2):
            loader.next_batch(step, micro)
    # 20 batches x 8 rows over a 40-row shard: the dataset rolled over 3 times.
    assert loader.epochs() == {"ds1": 4.0}


def test_closing_closes_every_stream():
    closed = []

    class Stream:
        def __iter__(self):
            return self

        def __next__(self):
            return "batch"

        def close(self):
            closed.append(True)

    c = curriculum(phases=[{"steps": 2, "mix": {"ds1": 1, "ds2": 1}}])
    loader = CurriculumLoader(MixSchedule(c, 2), lambda name: (Stream(), 8), 0, 1, 8)
    loader.next_batch(0, 0)
    loader.next_batch(0, 1)
    loader.close()
    assert len(closed) == 2


# --- config ------------------------------------------------------------------------------


def write_config(tmp_path, data, **top):
    path = tmp_path / "run.yaml"
    path.write_text(yaml.safe_dump({"model_config": "x.yaml", "data": data, **top}))
    return path


def test_the_run_is_as_long_as_its_phases(tmp_path):
    args = load_args(write_config(tmp_path, {"datasets": DATASETS, "curriculum": PHASES}))
    assert args.max_steps == 300
    assert args.data.resolve_curriculum().boundaries == (100, 250, 300)


def test_a_max_steps_that_disagrees_with_the_phases_is_an_error(tmp_path):
    path = write_config(tmp_path, {"datasets": DATASETS, "curriculum": PHASES}, max_steps=1000)
    with pytest.raises(ValueError, match="phases add up to 300"):
        load_args(path)
    # The matching value is fine to write out.
    load_args(write_config(tmp_path, {"datasets": DATASETS, "curriculum": PHASES}, max_steps=300))


def test_curriculum_and_single_manifest_are_alternatives():
    with pytest.raises(ValueError, match="alternatives"):
        DataArgs(train_jsonl="a.jsonl", datasets=DATASETS, curriculum=PHASES)


def test_datasets_without_phases_are_rejected():
    with pytest.raises(ValueError, match="data.curriculum"):
        DataArgs(datasets=DATASETS)


def test_a_plain_config_has_no_curriculum(tmp_path):
    args = load_args(write_config(tmp_path, {"train_jsonl": "a.jsonl", "valid_jsonl": "v.jsonl"}))
    assert args.data.resolve_curriculum() is None
    assert args.max_steps == 100_000


def test_the_resolved_config_round_trips(tmp_path):
    from training.args import save_args

    args = load_args(write_config(tmp_path, {"datasets": DATASETS, "curriculum": PHASES}))
    save_args(args, tmp_path / "args.yaml")
    again = load_args(tmp_path / "args.yaml")
    assert again.data.resolve_curriculum() == args.data.resolve_curriculum()
