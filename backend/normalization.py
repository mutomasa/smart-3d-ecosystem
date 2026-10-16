"""Observation normalization shared by inference (backend) and training (training.env).

Clients send raw world-unit features in the order of ``shared/schemas/policy.json``.
This module is the single place where they are scaled, so training and inference match.
"""

from __future__ import annotations

import numpy as np

from shared.contract import PolicyContract, load_contract

CLIP = 5.0


def _scale_vector(contract: PolicyContract) -> np.ndarray:
    c = contract
    hx, hy, hz = c.world_half_extents
    scales = {
        "group_center_x": hx, "group_center_y": hy, "group_center_z": hz,
        "group_velocity_x": c.max_speed, "group_velocity_y": c.max_speed, "group_velocity_z": c.max_speed,
        "nearest_predator_relative_x": c.max_distance, "nearest_predator_relative_y": c.max_distance,
        "nearest_predator_relative_z": c.max_distance, "nearest_predator_distance": c.max_distance,
        "nearest_food_relative_x": c.max_distance, "nearest_food_relative_y": c.max_distance,
        "nearest_food_relative_z": c.max_distance, "nearest_food_distance": c.max_distance,
        "mean_energy": c.max_energy,
        "group_density": c.max_density,
        "group_dispersion": c.max_distance,
        "nearby_group_count_normalized": 1.0,
        "predator_present": 1.0,
        "food_present": 1.0,
    }
    missing = set(c.feature_names) - scales.keys()
    if missing:
        raise ValueError(f"no normalization defined for features: {sorted(missing)}")
    return np.array([scales[name] for name in c.feature_names], dtype=np.float32)


class ObservationNormalizer:
    """Divides each raw feature by its scale and clips to [-CLIP, CLIP]; non-finite -> 0."""

    def __init__(self, contract: PolicyContract | None = None) -> None:
        self.contract = contract or load_contract()
        self.scale = _scale_vector(self.contract)

    def __call__(self, raw: np.ndarray) -> np.ndarray:
        x = np.asarray(raw, dtype=np.float32)
        if x.shape[-1] != self.contract.obs_dim:
            raise ValueError(f"expected {self.contract.obs_dim} features, got {x.shape[-1]}")
        x = np.where(np.isfinite(x), x, 0.0) / self.scale
        return np.clip(x, -CLIP, CLIP).astype(np.float32)
