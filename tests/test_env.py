import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env

from backend.normalization import ObservationNormalizer
from backend.policy import RulesPolicy
from training.config import EnvConfig, load_config
from training.env import FishSchoolEnv, raw_features

SMALL = EnvConfig(n_fish=16, episode_seconds=3.0)


def rollout(env: FishSchoolEnv, act, seed: int):
    obs, _ = env.reset(seed=seed)
    total, done, info = 0.0, False, {}
    while not done:
        obs, r, term, trunc, info = env.step(act(env, obs))
        total += r
        done = term or trunc
    return total, info


def test_passes_gymnasium_env_checker():
    check_env(FishSchoolEnv(SMALL), skip_render_check=True)


def test_episode_length_and_finite_values():
    env = FishSchoolEnv(SMALL)
    obs, _ = env.reset(seed=0)
    steps = 0
    done = False
    while not done:
        obs, r, term, trunc, info = env.step(env.action_space.sample())
        assert np.isfinite(obs).all() and np.isfinite(r)
        steps += 1
        done = term or trunc
    assert steps == SMALL.max_decisions
    assert 0 <= info["survival_rate"] <= 1


def test_same_seed_is_reproducible():
    zero = lambda env, obs: np.zeros(6, dtype=np.float32)
    assert rollout(FishSchoolEnv(SMALL), zero, 3) == rollout(FishSchoolEnv(SMALL), zero, 3)


def test_observation_uses_the_shared_normalizer():
    env = FishSchoolEnv(SMALL)
    obs, _ = env.reset(seed=1)
    np.testing.assert_array_equal(obs, ObservationNormalizer()(raw_features(env.sim)))
    assert obs[18] == 1.0  # at least one predator


def test_predators_eat_and_reward_penalizes_losses():
    cfg = EnvConfig(n_fish=16, episode_seconds=20.0, predators=(2, 2), food=(0, 0), predator_speed=(14, 14))
    no_flee = np.array([0, 0, 0, -5, 0, 0], dtype=np.float32)  # predator weight ~0
    _, info = rollout(FishSchoolEnv(cfg), lambda env, obs: no_flee, 0)
    assert info["caught"] > 0
    assert info["survival_rate"] < 1


def test_rules_policy_beats_passive_policy_on_food():
    cfg = EnvConfig(n_fish=16, episode_seconds=10.0, predators=(0, 0), food=(3, 3))

    def rules(env, obs):
        a = RulesPolicy().act(env.heuristic_raw_obs()[None])[0]
        weights = np.clip(a[3:], 1e-3, 1 - 1e-3)
        return np.concatenate([a[:3] * 5, np.log(weights / (1 - weights))]).astype(np.float32)

    passive = lambda env, obs: np.array([0, 0, 0, 0, -5, 0], dtype=np.float32)
    _, rules_info = rollout(FishSchoolEnv(cfg), rules, 2)
    _, passive_info = rollout(FishSchoolEnv(cfg), passive, 2)
    assert rules_info["food_collected"] > passive_info["food_collected"]


def test_load_config_files():
    assert load_config("default").n_fish == 48
    smoke = load_config("smoke")
    assert smoke.n_fish == 24 and smoke.reward.prey_lost == pytest.approx(2.0)
