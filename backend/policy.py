"""Group policies served over the WebSocket.

Every policy maps a batch of RAW observations (N, obs_dim) to actions (N, 6):
[unit direction x, y, z, predator_avoidance_weight, food_attraction_weight, cohesion_weight].
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np

from backend.actions import clamp_actions
from shared.contract import PolicyContract, load_contract

# Mirrors `RULES` in frontend/src/sim/policies.ts.
FLEE_DISTANCE = 25.0
HUNGRY_ENERGY = 0.7
FLEE_WEIGHTS = (1.0, 0.1, 0.8)
FORAGE_WEIGHTS = (0.6, 1.0, 0.4)
CRUISE_WEIGHTS = (0.6, 0.3, 0.5)


class Policy(Protocol):
    name: str
    kind: str
    model_version: str

    def act(self, raw_obs: np.ndarray) -> np.ndarray: ...


@dataclass(frozen=True)
class _Idx:
    pred_rel: slice
    pred_dist: int
    food_rel: slice
    energy: int
    pred_present: int
    food_present: int

    @classmethod
    def from_contract(cls, c: PolicyContract) -> _Idx:
        i = c.feature_index
        return cls(
            pred_rel=slice(i("nearest_predator_relative_x"), i("nearest_predator_relative_z") + 1),
            pred_dist=i("nearest_predator_distance"),
            food_rel=slice(i("nearest_food_relative_x"), i("nearest_food_relative_z") + 1),
            energy=i("mean_energy"),
            pred_present=i("predator_present"),
            food_present=i("food_present"),
        )


class RulesPolicy:
    """Handcrafted heuristic: flee a close predator, forage when hungry, otherwise cruise."""

    name = "rules"
    kind = "heuristic"
    model_version = "rules-v1"

    def __init__(self, contract: PolicyContract | None = None) -> None:
        self.contract = contract or load_contract()
        self.idx = _Idx.from_contract(self.contract)

    def act(self, raw_obs: np.ndarray) -> np.ndarray:
        obs = np.atleast_2d(np.asarray(raw_obs, dtype=np.float32))
        actions = np.zeros((obs.shape[0], self.contract.action_dim), dtype=np.float32)
        actions[:, 3:] = CRUISE_WEIGHTS
        flee = (obs[:, self.idx.pred_present] > 0.5) & (obs[:, self.idx.pred_dist] < FLEE_DISTANCE)
        forage = ~flee & (obs[:, self.idx.food_present] > 0.5) & (obs[:, self.idx.energy] < HUNGRY_ENERGY)
        actions[flee, :3] = -obs[flee, self.idx.pred_rel]
        actions[flee, 3:] = FLEE_WEIGHTS
        actions[forage, :3] = obs[forage, self.idx.food_rel]
        actions[forage, 3:] = FORAGE_WEIGHTS
        return clamp_actions(actions)


@dataclass(frozen=True)
class PolicyInfo:
    name: str
    available: bool
    kind: str
    model_version: str
    label: str
    device: str = "cpu"


class PolicyRegistry:
    """Looks up server-side policies by name. `boids` runs entirely in the browser."""

    def __init__(self, policies: dict[str, Policy] | None = None) -> None:
        self._policies: dict[str, Policy] = policies or {"rules": RulesPolicy()}

    def get(self, name: str) -> Policy | None:
        return self._policies.get(name)

    def register(self, policy: Policy) -> None:
        self._policies[policy.name] = policy

    def describe(self) -> list[PolicyInfo]:
        infos = [PolicyInfo("boids", True, "client", "boids", "Boids only (browser)")]
        for name in ("rules", "ml"):
            infos.append(self._info(name))
        return infos

    def _info(self, name: str) -> PolicyInfo:
        policy = self._policies.get(name)
        if policy is None:
            return PolicyInfo(name, False, "unavailable", "-", f"{name} (not loaded)")
        label = getattr(policy, "label", None) or f"{policy.kind} ({policy.model_version})"
        device = str(getattr(policy, "device", "cpu"))
        return PolicyInfo(name, True, policy.kind, policy.model_version, label, device)
