"""Policies usable inside FishSchoolEnv. Each returns the RAW action the env post-processes."""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

import numpy as np
import torch

from backend.actions import RAW_ACTION_LIMIT
from backend.model import load_checkpoint
from backend.policy import RulesPolicy
from training.env import FishSchoolEnv


class EnvAgent(Protocol):
    name: str

    def act(self, env: FishSchoolEnv, obs: np.ndarray) -> np.ndarray: ...


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-3, 1 - 1e-3)
    return np.log(p / (1 - p))


class RandomAgent:
    name = "random"

    def __init__(self, seed: int) -> None:
        self.rng = np.random.default_rng(seed)

    def act(self, env, obs):
        return self.rng.uniform(-RAW_ACTION_LIMIT, RAW_ACTION_LIMIT, env.action_space.shape).astype(np.float32)


class RulesAgent:
    """The handcrafted heuristic, encoded as raw outputs that post-process back to itself."""

    name = "rules"

    def __init__(self) -> None:
        self.policy = RulesPolicy()

    def act(self, env, obs):
        a = self.policy.act(env.heuristic_raw_obs()[None])[0]
        return np.clip(np.concatenate([a[:3], _logit(a[3:])]), -RAW_ACTION_LIMIT, RAW_ACTION_LIMIT).astype(np.float32)


class NetAgent:
    """A PolicyNet checkpoint; the env already provides normalized observations."""

    def __init__(self, name: str, checkpoint: Path) -> None:
        self.name = name
        self.net, self.meta = load_checkpoint(checkpoint, torch.device("cpu"))

    @torch.inference_mode()
    def act(self, env, obs):
        return self.net(torch.from_numpy(np.asarray(obs, dtype=np.float32))[None])[0].numpy()


def rollout(env: FishSchoolEnv, agent: EnvAgent, seed: int) -> dict:
    obs, _ = env.reset(seed=seed)
    total, done, info = 0.0, False, {}
    while not done:
        obs, reward, terminated, truncated, info = env.step(agent.act(env, obs))
        total += reward
        done = terminated or truncated
    return {"policy": agent.name, "seed": seed, "return": total, **info}
