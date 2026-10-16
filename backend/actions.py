"""Action post-processing shared by the ML policy and the training environment."""

from __future__ import annotations

import numpy as np

EPS = 1e-6


def normalize_directions(directions: np.ndarray) -> np.ndarray:
    """Clamps rows to at most unit length (longer rows are normalized, shorter ones keep their
    magnitude as preference strength). Zero / non-finite rows become the zero vector, so a
    "no preference" output never turns into an arbitrary unit direction."""
    d = np.where(np.isfinite(directions), directions, 0.0).astype(np.float32)
    norms = np.linalg.norm(d, axis=-1, keepdims=True)
    return np.where(norms > 1.0, d / np.maximum(norms, EPS), d).astype(np.float32)


def sigmoid(x: np.ndarray) -> np.ndarray:
    return (1.0 / (1.0 + np.exp(-np.clip(x, -30.0, 30.0)))).astype(np.float32)


def postprocess(raw: np.ndarray) -> np.ndarray:
    """Raw network output (N, 6) -> [direction with |d| <= 1 (3), weights in (0, 1) (3)]."""
    raw = np.asarray(raw, dtype=np.float32)
    return np.concatenate([normalize_directions(raw[:, :3]), sigmoid(raw[:, 3:])], axis=1)


def clamp_actions(actions: np.ndarray) -> np.ndarray:
    """Final safety net for any policy: |direction| <= 1 and weights clamped to [0, 1]."""
    a = np.asarray(actions, dtype=np.float32)
    weights = np.clip(np.where(np.isfinite(a[:, 3:]), a[:, 3:], 0.0), 0.0, 1.0)
    return np.concatenate([normalize_directions(a[:, :3]), weights], axis=1).astype(np.float32)
