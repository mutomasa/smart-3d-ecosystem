import json
from pathlib import Path

import numpy as np
import pytest

from backend.actions import clamp_actions, normalize_directions, postprocess
from backend.policy import PolicyRegistry, RulesPolicy

CASES = json.loads((Path(__file__).parents[1] / "shared/fixtures/rules_cases.json").read_text())


@pytest.mark.parametrize("case", CASES, ids=[c["name"] for c in CASES])
def test_rules_policy_matches_shared_fixture(case):
    action = RulesPolicy().act(np.array([case["features"]]))[0]
    np.testing.assert_allclose(action, case["expected"], atol=1e-6)


def test_rules_policy_is_batched():
    obs = np.array([c["features"] for c in CASES])
    actions = RulesPolicy().act(obs)
    assert actions.shape == (len(CASES), 6)


def test_direction_normalization_handles_zero_and_nan():
    d = normalize_directions(np.array([[3, 4, 0], [0, 0, 0], [np.nan, 1, 0]], dtype=np.float32))
    np.testing.assert_allclose(d[0], [0.6, 0.8, 0], atol=1e-6)
    np.testing.assert_array_equal(d[1], [0, 0, 0])
    assert np.isfinite(d).all()


def test_postprocess_bounds_weights_with_sigmoid():
    out = postprocess(np.array([[1, 0, 0, -100, 0, 100]], dtype=np.float32))
    assert out[0, 3] == pytest.approx(0, abs=1e-6)
    assert out[0, 4] == pytest.approx(0.5)
    assert out[0, 5] == pytest.approx(1, abs=1e-6)


def test_clamp_actions_never_exceeds_bounds():
    out = clamp_actions(np.array([[10, 0, 0, 5, -3, np.inf]], dtype=np.float32))
    assert np.linalg.norm(out[0, :3]) == pytest.approx(1)
    assert (out[0, 3:] >= 0).all() and (out[0, 3:] <= 1).all()


def test_registry_reports_missing_ml():
    infos = {p.name: p for p in PolicyRegistry().describe()}
    assert infos["boids"].available and infos["rules"].available
    assert not infos["ml"].available
