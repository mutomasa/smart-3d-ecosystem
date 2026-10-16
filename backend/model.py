"""Small MLP policy network (same shape as the SB3 default MlpPolicy actor: 64-64, tanh)."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import torch
from torch import nn


class PolicyNet(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int, hidden: tuple[int, ...] = (64, 64)) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        last = obs_dim
        for width in hidden:
            layers += [nn.Linear(last, width), nn.Tanh()]
            last = width
        layers.append(nn.Linear(last, action_dim))
        self.net = nn.Sequential(*layers)
        self.hidden = hidden

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs)


@dataclass
class ModelMetadata:
    """Side-car JSON describing how a checkpoint was produced."""

    model_version: str
    kind: str  # "imitation" | "ppo" | ...
    obs_dim: int
    action_dim: int
    hidden: list[int] = field(default_factory=lambda: [64, 64])
    contract_version: str = ""
    training: dict = field(default_factory=dict)


def metadata_path(checkpoint: Path) -> Path:
    return checkpoint.with_suffix(".json")


def save_checkpoint(net: PolicyNet, meta: ModelMetadata, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(net.state_dict(), path)
    metadata_path(path).write_text(json.dumps(asdict(meta), indent=2), encoding="utf-8")


def load_checkpoint(path: Path, device: torch.device) -> tuple[PolicyNet, ModelMetadata]:
    meta = ModelMetadata(**json.loads(metadata_path(path).read_text(encoding="utf-8")))
    net = PolicyNet(meta.obs_dim, meta.action_dim, tuple(meta.hidden))
    net.load_state_dict(torch.load(path, map_location=device, weights_only=True))
    return net.to(device).eval(), meta
