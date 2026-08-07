from argparse import Namespace

import pytest

from hebrew_training.server_launcher import (
    artifact_paths,
    precompute_commands,
    training_command,
    validate_model_kind,
)


def test_teacher_uses_model_specific_latents_and_larger_head(tmp_path):
    args = Namespace(
        artifacts_dir=tmp_path,
        base_language="french_24l",
        kind="teacher",
        run_dir=tmp_path / "run",
        device="cuda",
        steps=12000,
        gradient_accumulation=8,
        head_samples=None,
        head_batch_multiplier=8,
        eval_every=250,
        eval_samples=64,
        save_every=3000,
        smoke=False,
    )
    command = training_command(args)
    paths = artifact_paths(tmp_path, "french_24l")
    assert str(paths["train_latents"]) in command
    assert command[command.index("--head-samples") + 1] == "500"
    assert command[command.index("--loss-mode") + 1] == "flow"


def test_precompute_separates_base_model_cache(tmp_path):
    args = Namespace(artifacts_dir=tmp_path, base_language="german_24l", device="cuda", limit=3)
    commands = precompute_commands(args)
    assert len(commands) == 2
    assert all("german_24l" in " ".join(command) for command in commands)
    assert all(command[-2:] == ["--limit", "3"] for command in commands)


def test_model_kind_mismatch_is_rejected():
    with pytest.raises(ValueError, match="24l"):
        validate_model_kind("teacher", "english")
    with pytest.raises(ValueError, match="6-layer"):
        validate_model_kind("student", "french_24l")


def test_smoke_never_writes_checkpoint(tmp_path):
    args = Namespace(
        artifacts_dir=tmp_path,
        base_language="english",
        kind="student",
        run_dir=tmp_path / "smoke",
        device="cuda",
        steps=6000,
        gradient_accumulation=8,
        head_samples=None,
        head_batch_multiplier=8,
        eval_every=250,
        eval_samples=64,
        save_every=1000,
        smoke=True,
    )
    command = training_command(args)
    assert command[command.index("--steps") + 1] == "30"
    assert command[command.index("--save-every") + 1] == "0"
    assert "--skip-final-checkpoint" in command
