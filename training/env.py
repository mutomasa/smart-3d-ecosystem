"""Gymnasium environment: one representative fish group controlled at 10 Hz.

Observation: the 20 shared features (normalized with backend.normalization, like inference).
Action: 6 raw outputs in [-RAW_ACTION_LIMIT, RAW_ACTION_LIMIT], post-processed exactly as in
backend.ml_policy (direction clamped to unit length, sigmoid weights).
"""

from __future__ import annotations

from typing import ClassVar

import gymnasium as gym
import numpy as np

from backend.actions import RAW_ACTION_LIMIT, postprocess
from backend.normalization import ObservationNormalizer
from shared.contract import load_contract
from training.config import EnvConfig
from training.school import SchoolSim, StepEvents

CONTRACT = load_contract()


def raw_features(sim: SchoolSim) -> np.ndarray:
    """Raw observation in schema order (same definitions as frontend/src/sim/groups.ts)."""
    c = CONTRACT
    f = np.zeros(c.obs_dim, dtype=np.float32)
    alive = sim.alive
    center = sim.center()
    f[0:3] = center
    f[3:6] = sim.vel[alive].mean(axis=0) if alive.any() else 0.0
    f[6:10], f[18] = _nearest(center, sim.pred_pos, np.ones(len(sim.pred_pos), dtype=bool))
    f[10:14], f[19] = _nearest(center, sim.food_pos, sim.food_amount > 0)
    f[14] = sim.energy[alive].mean() if alive.any() else 0.0
    dispersion = sim.dispersion()
    f[15] = min(1.0, alive.sum() / (4 / 3 * np.pi * (dispersion + 1) ** 3))
    f[16] = dispersion
    f[17] = 0.0  # a single group: no neighbouring groups
    return f


def _nearest(center: np.ndarray, points: np.ndarray, active: np.ndarray) -> tuple[np.ndarray, float]:
    if not active.any():
        return np.array([0, 0, 0, CONTRACT.max_distance], dtype=np.float32), 0.0
    rel = points[active] - center
    dist = np.linalg.norm(rel, axis=1)
    k = int(dist.argmin())
    return np.array([*rel[k], min(CONTRACT.max_distance, dist[k])], dtype=np.float32), 1.0


class FishSchoolEnv(gym.Env):
    metadata: ClassVar[dict] = {"render_modes": []}

    def __init__(self, config: EnvConfig | None = None) -> None:
        super().__init__()
        self.cfg = config or EnvConfig()
        self.normalizer = ObservationNormalizer()
        self.observation_space = gym.spaces.Box(-5.0, 5.0, (CONTRACT.obs_dim,), np.float32)
        self.action_space = gym.spaces.Box(-RAW_ACTION_LIMIT, RAW_ACTION_LIMIT, (CONTRACT.action_dim,), np.float32)
        self.sim: SchoolSim | None = None
        self.decisions = 0
        self.prev_action = np.zeros(CONTRACT.action_dim, dtype=np.float32)
        self.totals: dict[str, float] = {}

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        self.sim = SchoolSim(self.cfg, self.np_random)
        self.decisions = 0
        self.prev_action = postprocess(np.zeros((1, CONTRACT.action_dim)))[0]
        self.totals = {"food": 0.0, "caught": 0, "collisions": 0}
        return self._obs(), {}

    def step(self, action: np.ndarray):
        assert self.sim is not None, "call reset() first"
        processed = postprocess(np.asarray(action, dtype=np.float32)[None])[0]
        self.sim.set_strategy(processed[:3], processed[3:])
        events = self._advance()
        reward = self._reward(events, processed)
        self.prev_action = processed
        self.decisions += 1
        terminated = not self.sim.alive.any()
        truncated = self.decisions >= self.cfg.max_decisions
        return self._obs(), reward, terminated, truncated, self._info()

    def heuristic_raw_obs(self) -> np.ndarray:
        """Raw (unnormalized) features, used by the rules baseline."""
        return raw_features(self.sim)

    # ---- internals ----------------------------------------------------------------------
    def _obs(self) -> np.ndarray:
        return self.normalizer(raw_features(self.sim))

    def _advance(self) -> StepEvents:
        """Sums food / catches over the substeps; collisions are the mean touching pairs per
        substep, so the penalty does not scale with the simulation rate."""
        total = StepEvents()
        collisions = 0
        for _ in range(self.cfg.substeps):
            e = self.sim.step(self.cfg.sim_dt)
            total.food += e.food
            total.caught += e.caught
            collisions += e.collisions
        total.collisions = collisions / self.cfg.substeps
        self.totals["food"] += total.food
        self.totals["caught"] += total.caught
        self.totals["collisions"] += total.collisions
        return total

    def _reward(self, e: StepEvents, action: np.ndarray) -> float:
        w = self.cfg.reward
        step_s = self.cfg.substeps * self.cfg.sim_dt
        survival = self.sim.alive.mean() * step_s
        cohesion = np.exp(-self.sim.dispersion() / 10.0) * step_s
        steering = float(np.sum((action - self.prev_action) ** 2))
        return float(
            w.food * e.food + w.survival * survival + w.cohesion * cohesion
            - w.prey_lost * e.caught - w.collision * e.collisions - w.steering * steering
        )

    def _info(self) -> dict:
        return {
            "survival_rate": float(self.sim.alive.mean()),
            "food_collected": self.totals["food"],
            "mean_energy": float(self.sim.energy[self.sim.alive].mean()) if self.sim.alive.any() else 0.0,
            "dispersion": self.sim.dispersion(),
            "collisions": self.totals["collisions"],
            "caught": self.totals["caught"],
        }
