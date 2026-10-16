"""Synthetic raw observations covering the situations the rules policy distinguishes."""

from __future__ import annotations

import numpy as np

from shared.contract import PolicyContract, load_contract


def _random_dirs(rng: np.random.Generator, n: int) -> np.ndarray:
    v = rng.normal(size=(n, 3))
    return v / np.linalg.norm(v, axis=1, keepdims=True)


def sample_observations(n: int, rng: np.random.Generator, contract: PolicyContract | None = None) -> np.ndarray:
    c = contract or load_contract()
    i = c.feature_index
    obs = np.zeros((n, c.obs_dim), dtype=np.float32)
    half = np.array(c.world_half_extents)
    obs[:, i("group_center_x"):i("group_center_z") + 1] = rng.uniform(-1, 1, (n, 3)) * half
    obs[:, i("group_velocity_x"):i("group_velocity_z") + 1] = _random_dirs(rng, n) * rng.uniform(3, 10, (n, 1))
    _sample_target(obs, rng, c, "predator", rng.random(n) < 0.6, max_dist=60)
    _sample_target(obs, rng, c, "food", rng.random(n) < 0.6, max_dist=80)
    obs[:, i("mean_energy")] = rng.uniform(0, 1, n)
    obs[:, i("group_density")] = rng.uniform(0, 1, n)
    obs[:, i("group_dispersion")] = rng.uniform(1, 15, n)
    obs[:, i("nearby_group_count_normalized")] = rng.uniform(0, 1, n)
    return obs


def _sample_target(obs, rng, c: PolicyContract, name: str, present: np.ndarray, max_dist: float) -> None:
    i = c.feature_index
    n = len(obs)
    dist = rng.uniform(0.5, max_dist, n)
    rel = _random_dirs(rng, n) * dist[:, None]
    start = i(f"nearest_{name}_relative_x")
    obs[:, start:start + 3] = np.where(present[:, None], rel, 0.0)
    obs[:, i(f"nearest_{name}_distance")] = np.where(present, dist, c.max_distance)
    obs[:, i(f"{name}_present")] = present.astype(np.float32)
