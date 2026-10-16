"""Evaluate policies on held-out seeds in FishSchoolEnv and write CSV / JSON reports.

uv run python -m training.evaluate --checkpoint bc-v1=backend/models/policy.pt \
    --checkpoint ppo-v1=backend/models/ppo.pt --seeds 10
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import pandas as pd

from training.agents import EnvAgent, NetAgent, RandomAgent, RulesAgent, rollout
from training.config import load_config
from training.env import FishSchoolEnv

EVAL_SEED_BASE = 10_000  # disjoint from training seeds (0 .. n_envs)
METRICS = ["return", "survival_rate", "food_collected", "mean_energy", "dispersion", "collisions", "caught"]
OUTPUT_DIR = Path("outputs/eval")


def evaluate(agents: list[EnvAgent], env: FishSchoolEnv, seeds: list[int]) -> pd.DataFrame:
    return pd.DataFrame([rollout(env, agent, seed) for agent in agents for seed in seeds])


def summarize(results: pd.DataFrame) -> pd.DataFrame:
    summary = results.groupby("policy")[METRICS].agg(["mean", "std"])
    summary.columns = [f"{m}_{s}" for m, s in summary.columns]
    return summary.reset_index()


def parse_checkpoint(spec: str) -> NetAgent:
    name, _, path = spec.partition("=")
    return NetAgent(name, Path(path))


def write_report(results: pd.DataFrame, summary: pd.DataFrame, meta: dict, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    results.to_csv(out_dir / "results.csv", index=False)
    summary.to_csv(out_dir / "summary.csv", index=False)
    report = {"meta": meta, "summary": summary.to_dict(orient="records")}
    (out_dir / "summary.json").write_text(json.dumps(report, indent=2))
    latest = OUTPUT_DIR / "latest.json"
    latest.parent.mkdir(parents=True, exist_ok=True)
    latest.write_text(json.dumps(report, indent=2))
    return out_dir


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="default")
    p.add_argument("--seeds", type=int, default=10)
    p.add_argument("--checkpoint", action="append", default=[], help="name=path/to/checkpoint.pt")
    args = p.parse_args()
    cfg = load_config(args.config)
    seeds = [EVAL_SEED_BASE + k for k in range(args.seeds)]
    agents: list[EnvAgent] = [RandomAgent(seed=1), RulesAgent(), *map(parse_checkpoint, args.checkpoint)]
    results = evaluate(agents, FishSchoolEnv(cfg), seeds)
    summary = summarize(results)
    meta = {"config": args.config, "seeds": seeds, "policies": [a.name for a in agents], "created_at": time.time()}
    out = write_report(results, summary, meta, OUTPUT_DIR / time.strftime("%Y%m%d-%H%M%S"))
    with pd.option_context("display.width", 200, "display.max_columns", 30):
        print(summary.round(3).to_string(index=False))
    print(f"report: {out}")


if __name__ == "__main__":
    main()
