import pandas as pd
import pytest
import requests

from dashboard.api_client import ApiClient, ApiError
from dashboard.charts import (
    assign_runs,
    comparison_bar,
    mode_summary,
    samples_frame,
    time_series_figure,
)


def sample(tick, mode="rules", seed=1, survival=1.0):
    return {"t": 0, "tick": tick, "mode": mode, "seed": seed, "survivalRate": survival, "foodCollected": tick / 10}


METRICS = {
    "A": {"seed": 1, "samples": [sample(60), sample(120, survival=0.9), sample(60, mode="ml", survival=1.0), sample(120, mode="ml", survival=0.95)]},
    "B": {"seed": 2, "samples": [sample(60, seed=2), sample(120, seed=2, survival=0.8)]},
    "C": {"seed": 3, "samples": []},
}


def test_runs_split_on_reset_and_mode_change():
    df = pd.DataFrame([sample(60), sample(120), sample(30), sample(90, mode="ml"), sample(150, mode="ml", seed=5)])
    assert assign_runs(df).tolist() == [0, 0, 1, 2, 3]


def test_samples_frame_flattens_experiments():
    df = samples_frame(METRICS)
    assert set(df["experiment_id"]) == {"A", "B"}
    assert df["sim_time_s"].max() == pytest.approx(2.0)
    assert df[df["experiment_id"] == "A"]["run"].nunique() == 2


def test_mode_summary_uses_final_value_per_run():
    summary = mode_summary(samples_frame(METRICS), ["survivalRate"]).set_index("mode")
    assert summary.loc["rules", "survivalRate mean"] == pytest.approx(0.85)  # runs A#0 (0.9) and B#0 (0.8)
    assert summary.loc["rules", "survivalRate count"] == 2
    assert summary.loc["ml", "survivalRate mean"] == pytest.approx(0.95)


def test_empty_metrics():
    assert samples_frame({}).empty
    assert mode_summary(pd.DataFrame(), ["survivalRate"]).empty


def test_figures_build():
    df = samples_frame(METRICS)
    assert len(time_series_figure(df, "survivalRate").data) >= 3
    assert len(comparison_bar(mode_summary(df, ["survivalRate"]), "survivalRate").data) == 1


class FakeResponse:
    def __init__(self, payload, status=200):
        self.payload, self.status_code = payload, status

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(f"{self.status_code}")

    def json(self):
        return self.payload


class FakeSession:
    def __init__(self, response=None, exc=None):
        self.response, self.exc, self.calls = response, exc, []

    def request(self, method, url, **kwargs):
        self.calls.append((method, url, kwargs))
        if self.exc:
            raise self.exc
        return self.response


def test_api_client_builds_urls_and_bodies():
    session = FakeSession(FakeResponse({"version": 1}))
    client = ApiClient("http://api/", session=session)
    assert client.push_settings("exp", {"food": 2}) == {"version": 1}
    method, url, kwargs = session.calls[0]
    assert (method, url, kwargs["json"]) == ("POST", "http://api/api/experiments/exp/settings", {"food": 2})


def test_api_client_wraps_errors():
    with pytest.raises(ApiError):
        ApiClient("http://api", session=FakeSession(exc=requests.ConnectionError("down"))).health()
    with pytest.raises(ApiError):
        ApiClient("http://api", session=FakeSession(FakeResponse({}, 500))).health()


def test_api_client_against_real_app(client):
    """End-to-end through the FastAPI TestClient (it is a requests-compatible session)."""
    api = ApiClient("http://testserver", session=client)
    api.reset("exp-x", 3, "rules", "Food Rush")
    assert api.push_settings("exp-x", {"preset": "Food Rush"})["settings"] == {"preset": "Food Rush"}
    assert "exp-x" in api.metrics()


def test_eval_summary_and_learning_curve_figures():
    from dashboard.charts import eval_bar, eval_summary_frame, learning_curve_figure

    report = {"summary": [{"policy": "rules", "return_mean": 1.0, "return_std": 0.1},
                          {"policy": "ppo-v1", "return_mean": 2.0, "return_std": 0.2}]}
    summary = eval_summary_frame(report)
    assert list(summary["policy"]) == ["rules", "ppo-v1"]
    assert len(eval_bar(summary, "return").data) == 1
    curve = pd.DataFrame({"timesteps": [10, 20], "return_ma": [0.0, 1.0]})
    assert len(learning_curve_figure({"ppo-v1": curve, "bad": pd.DataFrame()}).data) == 1


def test_short_runs_are_excluded_from_the_comparison():
    from dashboard.charts import drop_short_runs

    metrics = {"cmp-ml-1": {"samples": [sample(60, mode="boids")] + [sample(60 * k, mode="ml") for k in range(1, 8)]}}
    df, dropped = drop_short_runs(samples_frame(metrics))
    assert dropped == 1
    assert set(df["mode"]) == {"ml"}
    assert mode_summary(df, ["survivalRate"])["survivalRate count"].tolist() == [1]


def test_compare_prefix_keeps_matching_and_live_experiments():
    from dashboard.charts import filter_experiments

    metrics = {"cmp-ml-1": {}, "cmp-rules-1": {}, "demo": {}, "other": {}}
    assert set(filter_experiments(metrics, "cmp", keep="demo")) == {"cmp-ml-1", "cmp-rules-1", "demo"}
    assert filter_experiments(metrics, "", keep="demo") == metrics
