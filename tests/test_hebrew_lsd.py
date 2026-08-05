from __future__ import annotations

from types import SimpleNamespace

import torch
from torch import nn

from hebrew_training.lsd import AdaptiveLossWeight, _lsd_loss, flow_matching_lsd_loss
from hebrew_training.train import LatentDataset, load_checkpoint, save_checkpoint


class ConstantFlow(nn.Module):
    def __init__(self, latent_dim: int):
        super().__init__()
        self.velocity = nn.Parameter(torch.randn(latent_dim))

    def forward(self, condition, source_time, target_time, state):
        del condition, source_time, target_time
        return self.velocity.expand_as(state)


class TinyFlow(nn.Module):
    def __init__(self, condition_dim: int, latent_dim: int):
        super().__init__()
        self.projection = nn.Linear(condition_dim + latent_dim + 2, latent_dim)

    def forward(self, condition, source_time, target_time, state):
        return self.projection(torch.cat([condition, source_time, target_time, state], dim=-1))


def test_constant_velocity_is_exact_lsd_flow_map():
    torch.manual_seed(1)
    flow = ConstantFlow(latent_dim=4)
    condition = torch.randn(5, 3)
    clean = torch.randn(5, 4)

    loss = _lsd_loss(flow, condition, clean, adaptive_weight=None)

    torch.testing.assert_close(loss, torch.zeros_like(loss), atol=1e-6, rtol=0)


def test_lsd_jvp_backpropagates_finite_flow_head_gradients():
    torch.manual_seed(2)
    flow = TinyFlow(condition_dim=3, latent_dim=4)
    condition = torch.randn(6, 3)
    clean = torch.randn(6, 4)

    loss = _lsd_loss(flow, condition, clean, adaptive_weight=None)
    loss.backward()

    assert torch.isfinite(loss)
    assert all(parameter.grad is not None for parameter in flow.parameters())
    assert all(torch.isfinite(parameter.grad).all() for parameter in flow.parameters())


def test_lsd_jvp_runs_during_no_grad_evaluation():
    flow = TinyFlow(condition_dim=3, latent_dim=4)
    with torch.no_grad():
        loss = _lsd_loss(
            flow, condition=torch.randn(4, 3), clean=torch.randn(4, 4), adaptive_weight=None
        )

    assert torch.isfinite(loss)


def test_paper_split_and_adaptive_weight_are_trainable():
    torch.manual_seed(3)
    flow = TinyFlow(condition_dim=8, latent_dim=4)
    adaptive_weight = AdaptiveLossWeight(channels=128)
    condition = torch.randn(32, 8)
    clean = torch.randn(32, 4)

    total, flow_loss, lsd_loss = flow_matching_lsd_loss(
        flow, condition, clean, lsd_fraction=0.25, adaptive_weight=adaptive_weight
    )
    total.backward()

    assert all(torch.isfinite(value) for value in (total, flow_loss, lsd_loss))
    assert adaptive_weight.weight.grad is not None
    assert torch.isfinite(adaptive_weight.weight.grad).all()
    assert all(parameter.grad is not None for parameter in flow.parameters())


def test_adaptive_weight_returns_one_scalar_per_example():
    module = AdaptiveLossWeight(channels=128)
    source_time = torch.tensor([[0.0], [0.25], [0.75]])
    target_time = torch.tensor([[0.5], [0.5], [1.0]])

    result = module(source_time, target_time)

    assert result.shape == (3,)
    assert torch.isfinite(result).all()


def test_checkpoint_round_trip_includes_adaptive_weight(tmp_path):
    model = SimpleNamespace(flow_lm=nn.Linear(3, 2))
    adaptive_weight = AdaptiveLossWeight(channels=128)
    parameters = list(model.flow_lm.parameters()) + list(adaptive_weight.parameters())
    optimizer = torch.optim.AdamW(parameters, lr=1e-3)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
    expected_model = {
        key: value.detach().clone() for key, value in model.flow_lm.state_dict().items()
    }
    expected_weight = {
        key: value.detach().clone() for key, value in adaptive_weight.state_dict().items()
    }

    checkpoint = save_checkpoint(
        model, optimizer, scheduler, tmp_path, step=7, adaptive_loss_weight=adaptive_weight
    )
    with torch.no_grad():
        for parameter in model.flow_lm.parameters():
            parameter.add_(10)
        adaptive_weight.weight.add_(10)

    loaded_step = load_checkpoint(
        model, optimizer, scheduler, checkpoint, adaptive_loss_weight=adaptive_weight
    )

    assert loaded_step == 7
    for key, value in model.flow_lm.state_dict().items():
        torch.testing.assert_close(value, expected_model[key])
    for key, value in adaptive_weight.state_dict().items():
        torch.testing.assert_close(value, expected_weight[key])


def test_legacy_english_latents_are_rejected_for_24_layer_foreign_base(tmp_path):
    manifest = tmp_path / "latents.jsonl"
    manifest.write_text('{"latent_path": "unused.safetensors", "text": "test"}\n')

    LatentDataset(manifest, expected_base_language="english")
    try:
        LatentDataset(manifest, expected_base_language="french_24l")
    except ValueError as error:
        assert "Recompute target and prompt latents" in str(error)
    else:
        raise AssertionError("foreign base model accepted legacy English latents")
