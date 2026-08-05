from __future__ import annotations

import math
from collections.abc import Callable

import torch
from torch import nn
from torch.nn import functional as F


class AdaptiveLossWeight(nn.Module):
    """Learn ``w_psi(source_time, target_time)`` from deterministic Fourier features.

    This mirrors the adaptive log-variance weighting used by the official Flow Maps LSD
    reference implementation: separate positional embeddings are magnitude-preserving
    averaged, then projected to one scalar. It is a training-only auxiliary module and is
    not part of the exported Pocket TTS inference model.
    """

    def __init__(self, channels: int = 128, max_period: float = 10_000.0):
        super().__init__()
        if channels <= 0 or channels % 2:
            raise ValueError("Adaptive loss-weight channels must be a positive even number")
        half = channels // 2
        frequencies = torch.exp(
            -math.log(max_period) * torch.arange(half, dtype=torch.float32) / half
        )
        self.channels = channels
        self.register_buffer("frequencies", frequencies)
        self.weight = nn.Parameter(torch.randn(1, channels))

    def _embed(self, time_value: torch.Tensor) -> torch.Tensor:
        time_value = time_value.float().reshape(-1, 1)
        angles = time_value * self.frequencies
        return math.sqrt(2.0) * torch.cat([torch.cos(angles), torch.sin(angles)], dim=-1)

    def forward(self, source_time: torch.Tensor, target_time: torch.Tensor) -> torch.Tensor:
        source_embedding = self._embed(source_time)
        target_embedding = self._embed(target_time)
        embedding = (source_embedding + target_embedding) / math.sqrt(2.0)
        # The reference implementation magnitude-normalizes its projection on every
        # forward pass. F.normalize is the equivalent operation for a linear weight.
        normalized_weight = F.normalize(self.weight.float(), dim=-1)
        return F.linear(embedding, normalized_weight).squeeze(-1)


def _weighted_error(
    error: torch.Tensor,
    source_time: torch.Tensor,
    target_time: torch.Tensor,
    adaptive_weight: AdaptiveLossWeight | None,
) -> torch.Tensor:
    per_example = error.float().square().mean(dim=-1)
    if adaptive_weight is None:
        return per_example.mean()
    log_variance = adaptive_weight(source_time, target_time)
    return (torch.exp(-log_variance) * per_example + log_variance).mean()


def _lsd_loss(
    flow_net: Callable[[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor],
    condition: torch.Tensor,
    clean: torch.Tensor,
    adaptive_weight: AdaptiveLossWeight | None,
) -> torch.Tensor:
    """Compute the time-reversed equivalent of CALM paper Eq. 6.

    The paper parameterizes data at time 0 and noise at time 1. Pocket TTS decoding runs
    in the opposite direction (noise at 0 to data at 1), so ``source_time <= target_time``.
    The learned map is

        map(x, source, target) = x + (target - source) * F(x, source, target).

    ``torch.func.jvp`` computes its derivative with respect to ``target``. The velocity at
    the mapped point is evaluated with the current flow head under stop-gradient, matching
    the paper's ``F_phi^-`` self-distillation target.
    """

    noise = torch.randn_like(clean)
    source_time = torch.rand((clean.shape[0], 1), device=clean.device, dtype=clean.dtype)
    target_time = source_time + (1.0 - source_time) * torch.rand_like(source_time)
    source_state = (1.0 - source_time) * noise + source_time * clean

    def flow_map(target: torch.Tensor) -> torch.Tensor:
        velocity = flow_net(condition, source_time, target, source_state)
        return source_state + (target - source_time) * velocity

    mapped_state, target_derivative = torch.func.jvp(
        flow_map, (target_time,), (torch.ones_like(target_time),)
    )
    with torch.no_grad():
        detached_velocity = flow_net(
            condition.detach(), target_time.detach(), target_time.detach(), mapped_state.detach()
        )
    return _weighted_error(
        target_derivative - detached_velocity, source_time, target_time, adaptive_weight
    )


def flow_matching_lsd_loss(
    flow_net: Callable[[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor],
    condition: torch.Tensor,
    clean: torch.Tensor,
    lsd_fraction: float,
    adaptive_weight: AdaptiveLossWeight | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Split flow-head examples between diagonal FM and off-diagonal LSD.

    CALM uses 75% flow matching and 25% LSD. The split is performed after the head batch
    multiplier, exactly where the paper describes increasing the number of independent
    flow-head examples while reusing the expensive backbone output.
    """

    if not 0.0 < lsd_fraction < 1.0:
        raise ValueError("lsd_fraction must be strictly between 0 and 1")
    if clean.shape[0] < 2:
        raise ValueError("FM/LSD training requires at least two flow-head examples")
    if condition.shape[0] != clean.shape[0]:
        raise ValueError("condition and clean must have the same flow-head batch size")

    total_examples = clean.shape[0]
    flow_examples = int(total_examples * (1.0 - lsd_fraction))
    flow_examples = max(1, min(total_examples - 1, flow_examples))
    order = torch.randperm(total_examples, device=clean.device)
    flow_indices = order[:flow_examples]
    lsd_indices = order[flow_examples:]

    flow_condition = condition[flow_indices]
    flow_clean = clean[flow_indices]
    noise = torch.randn_like(flow_clean)
    time_value = torch.rand(
        (flow_clean.shape[0], 1), device=flow_clean.device, dtype=flow_clean.dtype
    )
    interpolated = (1.0 - time_value) * noise + time_value * flow_clean
    predicted_velocity = flow_net(flow_condition, time_value, time_value, interpolated)
    flow_loss = _weighted_error(
        predicted_velocity - (flow_clean - noise), time_value, time_value, adaptive_weight
    )

    lsd_loss = _lsd_loss(flow_net, condition[lsd_indices], clean[lsd_indices], adaptive_weight)
    combined = (flow_examples * flow_loss + len(lsd_indices) * lsd_loss) / total_examples
    return combined, flow_loss, lsd_loss
