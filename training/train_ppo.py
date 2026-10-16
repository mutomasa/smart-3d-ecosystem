"""PPO training of the group policy (Phase 4).

uv run python -m training.train_ppo --timesteps 300000 --init-from backend/models/policy.pt
The actor is exported to the same PolicyNet format the backend serves (state_dict + JSON).
"""

from __future__ import annotations

import os

# One BLAS thread per env worker: with N subprocesses the default thread pools oversubscribe
# the CPU (measured ~5x slower). Must be set before numpy / torch are imported.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import argparse
import json
import logging
import time
from pathlib import Path

import pandas as pd
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from torch import nn

from backend.model import ModelMetadata, PolicyNet, load_checkpoint, save_checkpoint
from shared.contract import load_contract
from training.config import load_config
from training.env import FishSchoolEnv

log = logging.getLogger(__name__)
OUTPUT_DIR = Path("outputs/ppo")
HIDDEN = (64, 64)


def build_model(env, seed: int, device: str) -> PPO:
    policy_kwargs = {"net_arch": {"pi": list(HIDDEN), "vf": list(HIDDEN)}, "activation_fn": nn.Tanh}
    return PPO("MlpPolicy", env, policy_kwargs=policy_kwargs, seed=seed, device=device,
               n_steps=512, batch_size=256, learning_rate=3e-4, gamma=0.99, ent_coef=0.0, verbose=0)


def actor_layers(model: PPO) -> list[nn.Linear]:
    linears = [m for m in model.policy.mlp_extractor.policy_net if isinstance(m, nn.Linear)]
    return [*linears, model.policy.action_net]


def warm_start(model: PPO, checkpoint: Path, log_std: float = -1.0) -> None:
    """Copies a PolicyNet (e.g. the behaviour-cloned bc-v1) into the PPO actor."""
    net, _ = load_checkpoint(checkpoint, torch.device("cpu"))
    src = [m for m in net.net if isinstance(m, nn.Linear)]
    with torch.no_grad():
        for dst, s in zip(actor_layers(model), src, strict=True):
            dst.weight.copy_(s.weight)
            dst.bias.copy_(s.bias)
        model.policy.log_std.fill_(log_std)


def export_actor(model: PPO) -> PolicyNet:
    c = load_contract()
    net = PolicyNet(c.obs_dim, c.action_dim, HIDDEN)
    dst = [m for m in net.net if isinstance(m, nn.Linear)]
    with torch.no_grad():
        for d, s in zip(dst, actor_layers(model), strict=True):
            d.weight.copy_(s.weight.cpu())
            d.bias.copy_(s.bias.cpu())
    return net.eval()


def learning_curve(monitor_dir: Path) -> pd.DataFrame:
    frames = [pd.read_csv(f, skiprows=1) for f in sorted(monitor_dir.glob("*.monitor.csv"))]
    if not frames:
        return pd.DataFrame(columns=["r", "l", "t"])
    df = pd.concat(frames).sort_values("t").reset_index(drop=True)
    df["timesteps"] = df["l"].cumsum()
    df["return_ma"] = df["r"].rolling(20, min_periods=1).mean()
    return df


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="default")
    p.add_argument("--timesteps", type=int, default=300_000)
    p.add_argument("--n-envs", type=int, default=16)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cpu", help="SB3 MLP training is usually fastest on CPU")
    p.add_argument("--init-from", type=Path, default=None)
    p.add_argument("--out", type=Path, default=Path("backend/models/ppo.pt"))
    p.add_argument("--version", default="ppo-v1")
    return p.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    args = parse_args()
    cfg = load_config(args.config)
    run_dir = OUTPUT_DIR / args.version
    monitor_dir = run_dir / "monitor"
    monitor_dir.mkdir(parents=True, exist_ok=True)
    vec_cls = SubprocVecEnv if args.n_envs > 1 else DummyVecEnv
    env = make_vec_env(FishSchoolEnv, n_envs=args.n_envs, seed=args.seed, monitor_dir=str(monitor_dir),
                       env_kwargs={"config": cfg}, vec_env_cls=vec_cls)
    model = build_model(env, args.seed, args.device)
    if args.init_from:
        warm_start(model, args.init_from)
    start = time.time()
    model.learn(total_timesteps=args.timesteps, progress_bar=False)
    env.close()
    seconds = round(time.time() - start, 1)
    curve = learning_curve(monitor_dir)
    curve.to_csv(run_dir / "learning_curve.csv", index=False)
    c = load_contract()
    meta = ModelMetadata(
        model_version=args.version, kind="ppo", obs_dim=c.obs_dim, action_dim=c.action_dim, hidden=list(HIDDEN),
        contract_version=c.version,
        training={**{k: str(v) for k, v in vars(args).items()}, "env_config": cfg.to_dict(), "seconds": seconds,
                  "episodes": len(curve), "final_return_ma": float(curve["return_ma"].iloc[-1]) if len(curve) else None},
    )
    save_checkpoint(export_actor(model), meta, args.out)
    (run_dir / "metadata.json").write_text(json.dumps(meta.training, indent=2, default=str))
    log.info("saved %s after %.0fs (%d episodes)", args.out, seconds, len(curve))


if __name__ == "__main__":
    main()
