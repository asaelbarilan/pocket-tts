"""Unit tests for the pure functions the training loop leans on."""

import pytest
import torch
from torch import nn

from training.args import OptimArgs, TrainArgs
from training.checkpointing import EMA
from training.modules.samplers import LSD
from training.scripts.shrink_checkpoint import select_layers, shrink
from training.train_utils import lr_at, pick_sample_voice, write_samples


def _args(**kw) -> TrainArgs:
    optim = OptimArgs(**kw.pop("optim", {}))
    return TrainArgs(optim=optim, **kw)


class TestLrSchedule:
    def test_warmup_ends_at_the_configured_lr(self):
        args = _args(optim={"lr": 2e-4, "warmup_steps": 100, "schedule": "constant"})
        assert lr_at(0, args) == pytest.approx(2e-6)
        assert lr_at(99, args) == pytest.approx(2e-4)
        assert lr_at(100, args) == pytest.approx(2e-4)

    def test_cosine_decays_to_the_floor_at_max_steps(self):
        args = _args(max_steps=1000, optim={"lr": 1e-3, "warmup_steps": 0, "schedule": "cosine"})
        assert lr_at(0, args) == pytest.approx(1e-3, rel=1e-3)
        assert lr_at(1000, args) == pytest.approx(0.0, abs=1e-9)
        assert lr_at(500, args) == pytest.approx(5e-4, rel=1e-2)

    def test_cosine_respects_lr_min_ratio(self):
        args = _args(
            max_steps=1000,
            optim={"lr": 1e-3, "warmup_steps": 0, "schedule": "cosine", "lr_min_ratio": 0.1},
        )
        assert lr_at(1000, args) == pytest.approx(1e-4, rel=1e-6)

    def test_constant_stays_flat(self):
        args = _args(max_steps=1000, optim={"lr": 2e-4, "warmup_steps": 0, "schedule": "constant"})
        assert lr_at(10, args) == lr_at(999, args) == pytest.approx(2e-4)


class TestSelectLayers:
    def test_keeps_the_ends_of_the_stack(self):
        assert select_layers(24, 6) == [0, 1, 2, 21, 22, 23]

    def test_identity_when_depths_match(self):
        assert select_layers(6, 6) == list(range(6))

    def test_rejects_growing_the_stack(self):
        with pytest.raises(AssertionError):
            select_layers(6, 24)

    def test_shrink_renumbers_kept_layers_contiguously(self):
        state = {f"transformer.layers.{i}.w": torch.tensor([float(i)]) for i in range(24)}
        state["out_eos.weight"] = torch.zeros(1)
        out, keep = shrink(state, 6)
        assert keep == [0, 1, 2, 21, 22, 23]
        assert [k for k in out if k.startswith("transformer")] == [
            f"transformer.layers.{i}.w" for i in range(6)
        ]
        # renumbering must preserve which teacher layer each slot came from
        assert out["transformer.layers.3.w"].item() == 21.0
        assert "out_eos.weight" in out, "non-backbone tensors are copied unchanged"


class TestEMA:
    def _model(self):
        m = nn.Linear(2, 2, bias=True)
        with torch.no_grad():
            m.weight.fill_(1.0)
            m.bias.fill_(1.0)
        return m

    def test_tracks_only_trainable_parameters(self):
        m = self._model()
        m.bias.requires_grad_(False)
        ema = EMA(m, 0.9)
        assert set(ema.shadow) == {"weight"}

    def test_update_moves_towards_the_live_weights(self):
        m = self._model()
        ema = EMA(m, 0.5)
        with torch.no_grad():
            m.weight.fill_(3.0)
        ema.update(m)
        assert ema.shadow["weight"][0, 0].item() == pytest.approx(2.0)

    def test_loading_a_partial_shadow_keeps_untracked_weights(self):
        """A distilled student freezes its heads, so its shadow covers only part
        of the model; the rest must survive loading rather than be overwritten."""
        m = self._model()
        ema = EMA(m, 0.9)
        ema.load_state_dict({"weight": torch.full((2, 2), 7.0)})
        assert ema.shadow["weight"][0, 0].item() == pytest.approx(7.0)
        assert "bias" not in ema.shadow or ema.shadow["bias"][0].item() == pytest.approx(1.0)


class TestDistillProb:
    def _v(self, s, t, x):
        return x * (s + t)

    def test_skip_rate_matches_probability(self):
        m = LSD(distill_prob=0.25)
        m.train()
        outs = [m.loss(self._v, torch.randn(2, 4), torch.randn(2, 4)) for _ in range(400)]
        rate = sum("flow_distill" in met for _, met, _ in outs) / 400
        assert 0.15 < rate < 0.35

    def test_ranks_agree_on_every_draw(self):
        """The skip draw is rank-synchronized: two instances (= two DDP ranks)
        must take the same branch on the same step."""
        a, b = LSD(distill_prob=0.25), LSD(distill_prob=0.25)
        a.train(), b.train()
        sa = [
            "flow_distill" in a.loss(self._v, torch.randn(2, 4), torch.randn(2, 4))[1]
            for _ in range(100)
        ]
        sb = [
            "flow_distill" in b.loss(self._v, torch.randn(2, 4), torch.randn(2, 4))[1]
            for _ in range(100)
        ]
        assert sa == sb

    def test_eval_mode_always_computes_the_full_loss(self):
        """Valid losses must stay comparable across distill_prob settings."""
        m = LSD(distill_prob=0.25)
        m.eval()
        for _ in range(30):
            _, met, _ = m.loss(self._v, torch.randn(2, 4), torch.randn(2, 4))
            assert "flow_distill" in met


class TestSampleVoice:
    def test_trims_latents_and_audio_and_prefers_2_to_5_sec(self):
        frame_rate = 12.5
        sample_rate = 24000
        # Row 0: 1.2s (15 frames), Row 1: 3.0s (38 frames), Row 2: 2.4s (30 frames), Row 3: 6.0s (75 frames)
        num_frames = torch.tensor([15, 38, 30, 75])
        max_T = 75
        latents = torch.zeros(4, max_T, 4)
        audio = torch.zeros(4, 1, max_T * 1920)
        for b, n in enumerate(num_frames.tolist()):
            latents[b, :n] = float(b + 1)
            audio[b, 0, : n * 1920] = float(b + 1)

        chosen_rows = set()
        for step in range(1, 30):
            v_lat, v_wav = pick_sample_voice(
                latents, num_frames, audio, frame_rate, sample_rate, seed=42, step=step
            )
            row_id = int(v_lat[0, 0].item()) - 1
            chosen_rows.add(row_id)
            expected_frames = int(num_frames[row_id].item())
            assert v_lat.shape == (expected_frames, 4)
            assert torch.all(v_lat == float(row_id + 1))
            assert v_wav.shape == (expected_frames * 1920,)
            assert torch.all(v_wav == float(row_id + 1))

        # Only rows 1 (3.04s) and 2 (2.4s) are in [2.0, 5.0]s, and both should be visited across steps.
        assert chosen_rows == {1, 2}

    def test_falls_back_to_any_row_when_none_in_2_to_5_sec(self):
        frame_rate = 12.5
        sample_rate = 24000
        num_frames = torch.tensor([15, 20])  # 1.2s and 1.6s
        latents = torch.ones(2, 40, 4)
        audio = torch.ones(2, 1, 40 * 1920)
        v_lat, v_wav = pick_sample_voice(
            latents, num_frames, audio, frame_rate, sample_rate, seed=0, step=2500
        )
        assert v_lat.shape[0] in {15, 20}
        assert v_wav.shape[0] == v_lat.shape[0] * 1920

    def test_write_samples_stores_and_returns_trimmed_voice_prompt(self, tmp_path):
        class _DummyModel(nn.Module):
            def generate(self, tokens, voices, temp, cfg_coef, texts):
                assert all(v.shape == (25, 4) for v in voices)
                return [torch.ones(10, 4) for _ in tokens]

        class _DummyMimi(nn.Module):
            sample_rate = 24000
            frame_rate = 12.5
            encoder_frame_rate = 12.5

            def decode_from_latent(self, latents, state):
                return torch.full((1, 1, latents.shape[1] * 1920), 0.1)

        args = _args(sample_sentences=["שלום עולם", "בדיקה"])
        voice_latents = torch.ones(25, 4)
        voice_audio = torch.full((25 * 1920,), 0.25)

        written = write_samples(
            _DummyModel(),
            _DummyMimi(),
            lambda s: [1, 2, 3],
            args,
            tmp_path,
            step=2500,
            voice_latents=voice_latents,
            device=torch.device("cpu"),
            voice_audio=voice_audio,
        )
        assert [label for label, _ in written] == ["[voice prompt]", "שלום עולם", "בדיקה"]
        assert (tmp_path / "samples" / "step00002500_voice.wav").is_file()
        assert (tmp_path / "samples" / "step00002500_0.wav").is_file()
        assert (tmp_path / "samples" / "step00002500_1.wav").is_file()
