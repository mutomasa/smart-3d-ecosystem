"""PyTorch MLP policy with batched inference on CUDA (falls back to CPU)."""

from __future__ import annotations

import logging
import os
from pathlib import Path

import numpy as np
import torch

from backend.actions import clamp_actions, postprocess
from backend.model import ModelMetadata, load_checkpoint
from backend.normalization import ObservationNormalizer
from backend.policy import Policy, PolicyRegistry, RulesPolicy

log = logging.getLogger(__name__)

DEFAULT_CHECKPOINT = Path(__file__).parent / "models" / "policy.pt"
KIND_LABELS = {"imitation": "ML (imitation of rules)", "ppo": "ML (PPO)"}


def select_device(force_cpu: bool = False) -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() and not force_cpu else "cpu")


class MlPolicy:
    """normalize -> MLP -> postprocess, one forward pass for all groups of a request."""

    name = "ml"

    def __init__(self, net: torch.nn.Module, meta: ModelMetadata, device: torch.device,
                 normalizer: ObservationNormalizer | None = None) -> None:
        self.net, self.meta, self.device = net, meta, device
        self.normalizer = normalizer or ObservationNormalizer()
        self.kind = meta.kind
        self.model_version = meta.model_version
        self.label = f"{KIND_LABELS.get(meta.kind, 'ML')} {meta.model_version}"

    @classmethod
    def from_checkpoint(cls, path: Path, device: torch.device) -> MlPolicy:
        net, meta = load_checkpoint(path, device)
        return cls(net, meta, device)

    @torch.inference_mode()
    def act(self, raw_obs: np.ndarray) -> np.ndarray:
        obs = torch.from_numpy(self.normalizer(np.atleast_2d(raw_obs))).to(self.device)
        raw = self.net(obs).float().cpu().numpy()
        return clamp_actions(postprocess(raw))


class UntrainedHeuristicPolicy(RulesPolicy):
    """Served as `ml` when no checkpoint exists, and labelled honestly as untrained."""

    name = "ml"
    kind = "heuristic-untrained"
    model_version = "untrained"
    label = "Heuristic policy (untrained)"


def build_registry(checkpoint: Path | None = None, force_cpu: bool | None = None) -> PolicyRegistry:
    checkpoint = checkpoint or Path(os.environ.get("POLICY_CHECKPOINT", DEFAULT_CHECKPOINT))
    force_cpu = force_cpu if force_cpu is not None else os.environ.get("FORCE_CPU") == "1"
    registry = PolicyRegistry()
    registry.register(_load_ml(checkpoint, select_device(force_cpu)))
    return registry


def _load_ml(checkpoint: Path, device: torch.device) -> Policy:
    if not checkpoint.exists():
        log.warning("no ML checkpoint at %s; serving the untrained heuristic", checkpoint)
        return UntrainedHeuristicPolicy()
    try:
        policy = MlPolicy.from_checkpoint(checkpoint, device)
    except Exception:  # corrupt / incompatible checkpoint must not take the server down
        log.exception("failed to load %s; serving the untrained heuristic", checkpoint)
        return UntrainedHeuristicPolicy()
    log.info("loaded %s on %s", policy.label, device)
    return policy
