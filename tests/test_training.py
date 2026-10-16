import json

import numpy as np
import pandas as pd
import pytest
import torch
from stable_baselines3.common.env_util import make_vec_env

from backend.ml_policy import MlPolicy
from backend.model import ModelMetadata, PolicyNet, save_checkpoint
from tests.helpers import CONTRACT
from training import evaluate as ev
from training.agents import NetAgent, RandomAgent, RulesAgent, rollout
from training.config import EnvConfig
from training.env import FishSchoolEnv
from training.train_ppo import build_model, export_actor, learning_curve, warm_start

TINY = EnvConfig(n_fish=12, episode_seconds=2.0)


@pytest.fixture
def bc_checkpoint(tmp_path):
    torch.manual_seed(1)
    net = PolicyNet(CONTRACT.obs_dim, CONTRACT.action_dim)
    path = tmp_path / "bc.pt"
    save_checkpoint(net, ModelMetadata("bc-test", "imitation", CONTRACT.obs_dim, CONTRACT.action_dim), path)
    return path


def tiny_model():
    env = make_vec_env(FishSchoolEnv, n_envs=1, seed=0, env_kwargs={"config": TINY})
    return build_model(env, seed=0, device="cpu")


def test_warm_start_then_export_reproduces_the_checkpoint(bc_checkpoint):
    model = tiny_model()
    warm_start(model, bc_checkpoint)
    exported = export_actor(model)
    original = NetAgent("bc", bc_checkpoint).net
    x = torch.randn(8, CONTRACT.obs_dim)
    torch.testing.assert_close(exported(x), original(x))


def test_exported_actor_matches_sb3_deterministic_action(tmp_path):
    """The served PolicyNet must equal SB3's deterministic actor (mean action, clipped)."""
    model = tiny_model()
    model.learn(256)
    path = tmp_path / "ppo.pt"
    save_checkpoint(export_actor(model), ModelMetadata("ppo-test", "ppo", CONTRACT.obs_dim, CONTRACT.action_dim), path)
    obs = np.random.default_rng(0).normal(size=(5, CONTRACT.obs_dim)).astype(np.float32)
    sb3, _ = model.predict(obs, deterministic=True)
    with torch.no_grad():
        ours = NetAgent("ppo", path).net(torch.from_numpy(obs)).numpy()
    np.testing.assert_allclose(np.clip(ours, -5, 5), sb3, atol=1e-5)
    assert MlPolicy.from_checkpoint(path, torch.device("cpu")).label == "ML (PPO) ppo-test"


def test_learning_curve_reads_monitor_files(tmp_path):
    (tmp_path / "0.monitor.csv").write_text('#{"t_start": 0}\nr,l,t\n1.0,10,0.5\n3.0,10,1.0\n')
    curve = learning_curve(tmp_path)
    assert curve["timesteps"].tolist() == [10, 20]
    assert curve["return_ma"].iloc[-1] == pytest.approx(2.0)


def test_rules_agent_reproduces_rules_through_postprocessing():
    env = FishSchoolEnv(TINY)
    obs, _ = env.reset(seed=0)
    from backend.actions import postprocess
    from backend.policy import RulesPolicy

    raw = RulesAgent().act(env, obs)
    expected = RulesPolicy().act(env.heuristic_raw_obs()[None])[0]
    # weights saturate at sigmoid(RAW_ACTION_LIMIT) = 0.9933 instead of exactly 1.0
    np.testing.assert_allclose(postprocess(raw[None])[0], expected, atol=0.007)


def test_evaluate_report(tmp_path, monkeypatch, bc_checkpoint):
    monkeypatch.setattr(ev, "OUTPUT_DIR", tmp_path)
    agents = [RandomAgent(0), RulesAgent(), NetAgent("bc", bc_checkpoint)]
    results = ev.evaluate(agents, FishSchoolEnv(TINY), seeds=[10_000, 10_001])
    assert len(results) == 6 and set(results["policy"]) == {"random", "rules", "bc"}
    summary = ev.summarize(results)
    assert {"return_mean", "survival_rate_std"} <= set(summary.columns)
    out = ev.write_report(results, summary, {"seeds": [10_000, 10_001]}, tmp_path / "run")
    assert pd.read_csv(out / "results.csv").shape[0] == 6
    assert json.loads((tmp_path / "latest.json").read_text())["meta"]["seeds"] == [10_000, 10_001]


def test_rollout_is_reproducible_for_a_seed():
    env = FishSchoolEnv(TINY)
    assert rollout(env, RulesAgent(), 5) == rollout(env, RulesAgent(), 5)
