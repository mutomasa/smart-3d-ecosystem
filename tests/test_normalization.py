import numpy as np
import pytest

from backend.normalization import CLIP, ObservationNormalizer
from tests.helpers import CONTRACT, features


def test_scales_positions_by_world_extents_and_keeps_flags():
    norm = ObservationNormalizer()
    raw = np.array([features(group_center_x=25.0, predator_present=1.0)])
    out = norm(raw)[0]
    assert out[CONTRACT.feature_index("group_center_x")] == pytest.approx(0.5)
    assert out[CONTRACT.feature_index("predator_present")] == 1.0
    assert out[CONTRACT.feature_index("nearest_predator_distance")] == pytest.approx(1.0)


def test_clips_and_sanitizes_non_finite_values():
    raw = np.array([features(group_center_x=1e9, group_velocity_x=np.nan)])
    out = ObservationNormalizer()(raw)[0]
    assert out[0] == CLIP
    assert out[CONTRACT.feature_index("group_velocity_x")] == 0.0


def test_rejects_wrong_dimension():
    with pytest.raises(ValueError):
        ObservationNormalizer()(np.zeros((1, 3)))


def test_same_instance_used_by_training_and_inference_is_deterministic():
    raw = np.random.default_rng(0).normal(size=(8, CONTRACT.obs_dim)) * 30
    np.testing.assert_array_equal(ObservationNormalizer()(raw), ObservationNormalizer()(raw))
