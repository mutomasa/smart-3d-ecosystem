"""Behaviour cloning of the rules heuristic -> first ML checkpoint (Phase 3).

uv run python -m training.imitation --steps 15000 --out backend/models/policy.pt
"""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

from backend.ml_policy import DEFAULT_CHECKPOINT, select_device
from backend.model import ModelMetadata, PolicyNet, save_checkpoint
from backend.normalization import ObservationNormalizer
from backend.policy import RulesPolicy
from shared.contract import load_contract
from training.observations import sample_observations

log = logging.getLogger(__name__)


def make_dataset(n: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    raw = sample_observations(n, np.random.default_rng(seed))
    return ObservationNormalizer()(raw), RulesPolicy().act(raw)


def bc_loss(out: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Direction regressed on the raw output, weights through the same sigmoid as inference."""
    return nn.functional.mse_loss(out[:, :3], target[:, :3]) + nn.functional.mse_loss(torch.sigmoid(out[:, 3:]), target[:, 3:])


def train(steps: int, batch: int, seed: int, device: torch.device) -> tuple[PolicyNet, dict]:
    torch.manual_seed(seed)
    contract = load_contract()
    x, y = (torch.from_numpy(a).to(device) for a in make_dataset(200_000, seed))
    xv, yv = (torch.from_numpy(a).to(device) for a in make_dataset(20_000, seed + 1))
    net = PolicyNet(contract.obs_dim, contract.action_dim).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=3e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    for step in range(steps):
        idx = torch.randint(0, len(x), (batch,), device=device)
        loss = bc_loss(net(x[idx]), y[idx])
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
        if step % 500 == 0:
            log.info("step %d loss %.5f", step, loss.item())
    with torch.no_grad():
        val = bc_loss(net(xv), yv).item()
    return net.eval(), {"val_loss": val, "train_loss": loss.item()}


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=15000)
    p.add_argument("--batch", type=int, default=1024)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=Path, default=DEFAULT_CHECKPOINT)
    p.add_argument("--version", default="bc-v1")
    args = p.parse_args()
    device = select_device()
    start = time.time()
    net, stats = train(args.steps, args.batch, args.seed, device)
    contract = load_contract()
    meta = ModelMetadata(
        model_version=args.version, kind="imitation", obs_dim=contract.obs_dim, action_dim=contract.action_dim,
        contract_version=contract.version,
        training={**vars(args), "out": str(args.out), "device": str(device), "seconds": round(time.time() - start, 1), **stats},
    )
    save_checkpoint(net.cpu(), meta, args.out)
    log.info("saved %s (val_loss %.5f)", args.out, stats["val_loss"])


if __name__ == "__main__":
    main()
