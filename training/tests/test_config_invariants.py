"""The shipped configs must be runnable as-is.

Every defect these catch has actually shipped: a scratch config that stopped
before the quality transition, a batch size a quarter of the floor it needs,
and a teacher path pointing at an architecture the distill step cannot load.
"""

from pathlib import Path

import pytest

from training.args import TrainArgs, _from_dict, load_args
from training.modules.builders import load_model_config

CONFIGS = Path(__file__).resolve().parents[1] / "configs"
SCRATCH = CONFIGS / "scratch.yaml"
DISTILL = CONFIGS / "depth_distill.yaml"

# Below 64 rows per optimizer step the acoustic-quality transition arrives late
# or not at all, and 400k steps is where expressivity settles (see README).
MIN_EFFECTIVE_BATCH = 64
MIN_SCRATCH_STEPS = 400_000


@pytest.mark.parametrize("path", sorted(CONFIGS.glob("*.yaml")), ids=lambda p: p.name)
def test_config_parses(path: Path):
    load_args(path)


def test_scratch_reaches_the_effective_batch_floor():
    args = load_args(SCRATCH)
    assert args.batch_size * args.grad_accum_steps >= MIN_EFFECTIVE_BATCH, (
        "scratch must reach 64 rows per step on a single GPU: "
        f"{args.batch_size} x {args.grad_accum_steps}"
    )


def test_scratch_runs_past_the_quality_transition():
    assert load_args(SCRATCH).max_steps >= MIN_SCRATCH_STEPS


def test_scratch_builds_the_reference_teacher_depth():
    args = load_args(SCRATCH)
    config = load_model_config(args.model_config, args.model_overrides)
    assert config.flow_lm.transformer.num_layers == 24


def test_distill_teacher_is_deeper_than_its_student():
    args = load_args(DISTILL)
    student = load_model_config(args.model_config, args.model_overrides)
    teacher = load_model_config(args.distill_teacher_config, args.distill_teacher_overrides)
    assert teacher.flow_lm.transformer.num_layers > student.flow_lm.transformer.num_layers
    # Depth distillation copies every non-backbone tensor, so the rest must match.
    assert teacher.flow_lm.transformer.d_model == student.flow_lm.transformer.d_model


def test_distill_teacher_weights_point_at_the_scratch_run():
    args = load_args(DISTILL)
    assert args.distill_teacher_weights, "the distill config must name a teacher checkpoint"
    assert str(load_args(SCRATCH).run_dir) in args.distill_teacher_weights, (
        "the documented path is scratch -> distill; the teacher checkpoint should come from "
        f"{load_args(SCRATCH).run_dir}"
    )


class TestArgValidation:
    """Misconfigurations that used to run and quietly do the wrong thing."""

    def test_num_ckpt_keep_zero_is_rejected(self):
        with pytest.raises(ValueError, match="num_ckpt_keep"):
            TrainArgs(num_ckpt_keep=0)

    def test_zero_frequencies_are_rejected(self):
        for field in ("valid_freq", "ckpt_freq", "log_freq"):
            with pytest.raises(ValueError, match=field):
                TrainArgs(**{field: 0})

    def test_distillation_without_a_teacher_is_rejected(self):
        with pytest.raises(ValueError, match="teacher"):
            TrainArgs(distill_cfg_coef=1.5, start_from_pretrained=False)

    def test_teacher_config_without_weights_is_rejected(self):
        with pytest.raises(ValueError, match="distill_teacher_weights"):
            TrainArgs(distill_teacher_config="x.yaml")

    def test_unknown_keys_are_rejected(self):
        """A key the parser doesn't recognize is a setting the user thinks is applied."""
        with pytest.raises(ValueError, match="distill_seed_layers"):
            _from_dict(TrainArgs, {"distill_seed_layers": "first"})


class TestVocabSizeFollowsTheTokenizer:
    """n_bins is read from the tokenizer file, so the vocabulary size is chosen once."""

    @staticmethod
    def _tokenizer(tmp_path: Path, vocab_size: int) -> Path:
        import random

        import sentencepiece as spm

        rng = random.Random(0)
        letters = "אבגדהוזחטיכלמנספרשתabcdefghij"
        corpus = tmp_path / "corpus.txt"
        corpus.write_text(
            "\n".join(
                " ".join("".join(rng.choices(letters, k=rng.randint(2, 6))) for _ in range(8))
                for _ in range(300)
            ),
            encoding="utf-8",
        )
        spm.SentencePieceTrainer.train(
            input=str(corpus),
            model_prefix=str(tmp_path / "tok"),
            vocab_size=vocab_size,
            model_type="unigram",
            character_coverage=1.0,
            bos_id=1,
            eos_id=2,
            unk_id=0,
            pad_id=-1,
        )
        return tmp_path / "tok.model"

    @pytest.mark.parametrize("vocab_size", [60, 80])
    def test_n_bins_is_the_tokenizers_vocab_size(self, tmp_path, vocab_size):
        tokenizer = self._tokenizer(tmp_path, vocab_size)
        args = load_args(CONFIGS / "finetune_hebrew.yaml")
        overrides = {**args.model_overrides, "flow_lm.lookup_table.tokenizer_path": str(tokenizer)}
        config = load_model_config(args.model_config, overrides)
        assert config.flow_lm.lookup_table.n_bins == vocab_size

    def test_a_stale_n_bins_override_loses_to_the_tokenizer(self, tmp_path):
        tokenizer = self._tokenizer(tmp_path, 60)
        overrides = {
            "flow_lm.lookup_table.tokenizer_path": str(tokenizer),
            "flow_lm.lookup_table.n_bins": 4000,
        }
        config = load_model_config("pocket_tts/config/english_2026-04_24l.yaml", overrides)
        assert config.flow_lm.lookup_table.n_bins == 60

    def test_a_missing_tokenizer_is_an_error_naming_it(self):
        overrides = {"flow_lm.lookup_table.tokenizer_path": "tokenizers/does-not-exist.model"}
        with pytest.raises(FileNotFoundError, match="does-not-exist"):
            load_model_config("pocket_tts/config/english_2026-04_24l.yaml", overrides)

    def test_released_tokenizers_keep_the_configs_n_bins(self):
        config = load_model_config("pocket_tts/config/english_2026-04_24l.yaml", {})
        assert config.flow_lm.lookup_table.tokenizer_path.startswith("hf://")
        assert config.flow_lm.lookup_table.n_bins == 4000
