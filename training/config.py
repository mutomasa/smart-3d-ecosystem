"""Environment / reward configuration (JSON files in training/configs/)."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

CONFIG_DIR = Path(__file__).parent / "configs"


@dataclass(frozen=True)
class RewardWeights:
    food: float = 1.0
    survival: float = 0.2
    cohesion: float = 0.1
    prey_lost: float = 2.0
    collision: float = 0.05
    steering: float = 0.01


@dataclass(frozen=True)
class EnvConfig:
    n_fish: int = 48
    predators: tuple[int, int] = (1, 2)
    food: tuple[int, int] = (2, 3)
    predator_speed: tuple[float, float] = (9.5, 11.5)
    episode_seconds: float = 30.0
    decision_hz: float = 10.0
    sim_dt: float = 1 / 30
    # Boids parameters (mirrors frontend/src/sim/config.ts defaults)
    max_speed: float = 10.0
    min_speed: float = 3.0
    max_force: float = 25.0
    perception: float = 4.0
    separation_radius: float = 1.6
    collision_radius: float = 0.3
    boundary_margin: float = 8.0
    w_sep: float = 1.6
    w_align: float = 1.0
    w_coh: float = 0.8
    w_boundary: float = 2.5
    w_predator: float = 3.0
    w_food: float = 1.0
    w_ml: float = 1.0
    danger_radius: float = 18.0
    catch_radius: float = 1.4
    digest_time: float = 1.0
    food_amount: float = 40.0
    eat_radius: float = 3.0
    sense_radius: float = 35.0
    eat_rate: float = 0.5
    energy_decay: float = 0.02
    reward: RewardWeights = field(default_factory=RewardWeights)

    @property
    def substeps(self) -> int:
        return max(1, round(1 / (self.decision_hz * self.sim_dt)))

    @property
    def max_decisions(self) -> int:
        return int(self.episode_seconds * self.decision_hz)

    def to_dict(self) -> dict:
        return asdict(self)


def load_config(name_or_path: str | Path = "default") -> EnvConfig:
    path = Path(name_or_path)
    if not path.suffix:
        path = CONFIG_DIR / f"{name_or_path}.json"
    raw = json.loads(path.read_text(encoding="utf-8"))
    reward = RewardWeights(**raw.pop("reward", {}))
    for key in ("predators", "food", "predator_speed"):
        if key in raw:
            raw[key] = tuple(raw[key])
    return EnvConfig(**raw, reward=reward)
