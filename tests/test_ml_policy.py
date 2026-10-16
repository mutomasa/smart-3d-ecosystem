import numpy as np
import pytest
import torch

from backend.ml_policy import MlPolicy, UntrainedHeuristicPolicy, build_registry, select_device
from backend.model import ModelMetadata, PolicyNet, save_checkpoint
from backend.policy import RulesPolicy
from tests.helpers import CONTRACT
from training.imitation import bc_loss, make_dataset, train
from training.observations import sample_observations

CPU = torch.device("cpu")


@pytest.fixture
def checkpoint(tmp_path):
    torch.manual_seed(0)
    net = PolicyNet(CONTRACT.obs_dim, CONTRACT.action_dim)
    meta = ModelMetadata(model_version="test-v0", kind="imitation", obs_dim=CONTRACT.obs_dim, action_dim=CONTRACT.action_dim)
    path = tmp_path / "policy.pt"
    save_checkpoint(net, meta, path)
    return path


def test_checkpoint_roundtrip_and_metadata(checkpoint):
    policy = MlPolicy.from_checkpoint(checkpoint, CPU)
    assert policy.model_version == "test-v0"
    assert policy.label == "ML (imitation of rules) test-v0"


def test_batched_inference_is_bounded_and_deterministic(checkpoint):
    policy = MlPolicy.from_checkpoint(checkpoint, CPU)
    obs = sample_observations(32, np.random.default_rng(0))
    a, b = policy.act(obs), policy.act(obs)
    assert a.shape == (32, 6)
    np.testing.assert_array_equal(a, b)
    assert (np.linalg.norm(a[:, :3], axis=1) <= 1 + 1e-5).all()
    assert ((a[:, 3:] >= 0) & (a[:, 3:] <= 1)).all()


def test_missing_checkpoint_serves_honestly_labelled_heuristic(tmp_path):
    registry = build_registry(tmp_path / "none.pt", force_cpu=True)
    policy = registry.get("ml")
    assert isinstance(policy, UntrainedHeuristicPolicy)
    info = {p.name: p for p in registry.describe()}["ml"]
    assert info.kind == "heuristic-untrained" and info.label == "Heuristic policy (untrained)"


def test_corrupt_checkpoint_does_not_crash(tmp_path, checkpoint):
    checkpoint.write_bytes(b"garbage")
    assert isinstance(build_registry(checkpoint, force_cpu=True).get("ml"), UntrainedHeuristicPolicy)


def test_force_cpu():
    assert select_device(force_cpu=True).type == "cpu"


def test_behaviour_cloning_learns_the_rules():
    _, stats = train(steps=1500, batch=512, seed=0, device=CPU)
    x, y = (torch.from_numpy(a) for a in make_dataset(2000, seed=5))
    untrained = bc_loss(PolicyNet(CONTRACT.obs_dim, CONTRACT.action_dim)(x), y).item()
    assert stats["val_loss"] < untrained * 0.5


def test_ml_policy_uses_the_shared_normalizer(checkpoint):
    """Feeding raw features must equal feeding pre-normalized features to the bare network."""
    policy = MlPolicy.from_checkpoint(checkpoint, CPU)
    raw = sample_observations(4, np.random.default_rng(1))
    with torch.no_grad():
        direct = policy.net(torch.from_numpy(policy.normalizer(raw))).numpy()
    from backend.actions import clamp_actions, postprocess
    np.testing.assert_allclose(policy.act(raw), clamp_actions(postprocess(direct)), atol=1e-6)


def test_rules_and_untrained_heuristic_agree():
    obs = sample_observations(64, np.random.default_rng(2))
    np.testing.assert_array_equal(RulesPolicy().act(obs), UntrainedHeuristicPolicy().act(obs))
