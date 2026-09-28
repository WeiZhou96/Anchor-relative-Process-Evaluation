"""Matched causal learning recipes for the supervised development pilot."""

from __future__ import annotations

import hashlib
import math

import torch
from torch import nn
from torch.nn import functional as F


class PrefixMean(nn.Module):
    def __init__(self, in_dim: int = 512) -> None:
        super().__init__()
        self.head = nn.Linear(in_dim, 5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        denominator = torch.arange(1, x.shape[1] + 1, device=x.device, dtype=x.dtype)[:, None]
        averaging = torch.tril(torch.ones(x.shape[1], x.shape[1], device=x.device, dtype=x.dtype)) / denominator
        return self.head(averaging[None] @ x)


class GRU(nn.Module):
    def __init__(self, in_dim: int = 512) -> None:
        super().__init__()
        self.core = nn.GRU(in_dim, 128, batch_first=True)
        self.head = nn.Linear(128, 5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.core(x)[0])


class CausalTransformer(nn.Module):
    def __init__(self, in_dim: int = 512) -> None:
        super().__init__()
        self.project = nn.Linear(in_dim, 128)
        self.block = nn.TransformerEncoderLayer(128, 4, dim_feedforward=256, dropout=0, batch_first=True)
        self.head = nn.Linear(128, 5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        length = x.shape[1]
        position = torch.arange(length, device=x.device, dtype=x.dtype)[:, None]
        scale = torch.exp(torch.arange(0, 128, 2, device=x.device, dtype=x.dtype) * (-math.log(10000) / 128))
        encoding = torch.zeros(length, 128, device=x.device, dtype=x.dtype)
        encoding[:, 0::2] = torch.sin(position * scale)
        encoding[:, 1::2] = torch.cos(position * scale)
        mask = torch.triu(torch.ones(length, length, device=x.device, dtype=torch.bool), diagonal=1)
        return self.head(self.block(self.project(x) + encoding[None], src_mask=mask))


ARMS = ["mean_ce", "gru_ce", "gru_time", "gru_suffix", "transformer_ce"]


def build_model(arm: str, in_dim: int = 512) -> nn.Module:
    if arm == "mean_ce":
        return PrefixMean(in_dim)
    if arm.startswith("gru_"):
        return GRU(in_dim)
    if arm == "transformer_ce":
        return CausalTransformer(in_dim)
    raise ValueError(arm)


def state_hash(model: nn.Module) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        digest.update(name.encode())
        digest.update(value.detach().cpu().numpy().tobytes())
    return digest.hexdigest()


def objective(
    logits: torch.Tensor, labels: torch.Tensor, weights: torch.Tensor, class_weights: torch.Tensor, arm: str
) -> torch.Tensor:
    """Apply a normalized per-clip objective, ignoring every padded readout."""
    batch, steps, classes = logits.shape
    per = F.cross_entropy(
        logits.reshape(-1, classes), labels[:, None].expand(batch, steps).reshape(-1), reduction="none"
    ).reshape(batch, steps)
    if arm == "gru_suffix":
        per = torch.where(weights > 0, per, torch.zeros_like(per))
        per = torch.cummax(per.flip(1), dim=1).values.flip(1)
    elif arm == "gru_time":
        index = torch.arange(1, steps + 1, dtype=weights.dtype, device=weights.device)
        weights = weights * index[None]
        weights = weights / weights.sum(1, keepdim=True)
    return ((per * weights).sum(1) * class_weights[labels]).mean()


def causal_ema(probabilities: torch.Tensor, alpha: float = 0.5) -> torch.Tensor:
    values = [probabilities[:, 0]]
    for j in range(1, probabilities.shape[1]):
        values.append(alpha * probabilities[:, j] + (1 - alpha) * values[-1])
    return torch.stack(values, dim=1)
