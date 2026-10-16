"""Loader for the observation/action contract in ``shared/schemas/policy.json``."""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

SCHEMA_PATH = Path(__file__).parent / "schemas" / "policy.json"


@dataclass(frozen=True)
class PolicyContract:
    version: str
    feature_names: tuple[str, ...]
    action_names: tuple[str, ...]
    world_half_extents: tuple[float, float, float]
    max_speed: float
    max_distance: float
    max_density: float
    max_energy: float
    max_groups: int
    max_message_bytes: int
    request_timeout_ms: int

    @property
    def obs_dim(self) -> int:
        return len(self.feature_names)

    @property
    def action_dim(self) -> int:
        return len(self.action_names)

    def feature_index(self, name: str) -> int:
        return self.feature_names.index(name)


def _parse(raw: dict) -> PolicyContract:
    norm = raw["normalization"]
    limits = raw["limits"]
    return PolicyContract(
        version=raw["version"],
        feature_names=tuple(raw["observation"]["features"]),
        action_names=tuple(raw["action"]["outputs"]),
        world_half_extents=tuple(raw["coordinate_system"]["world_half_extents"]),
        max_speed=float(norm["max_speed"]),
        max_distance=float(norm["max_distance"]),
        max_density=float(norm["max_density"]),
        max_energy=float(norm["max_energy"]),
        max_groups=int(limits["max_groups"]),
        max_message_bytes=int(limits["max_message_bytes"]),
        request_timeout_ms=int(limits["request_timeout_ms"]),
    )


@lru_cache(maxsize=1)
def load_contract(path: Path = SCHEMA_PATH) -> PolicyContract:
    return _parse(json.loads(path.read_text(encoding="utf-8")))
