"""Streamlit dashboard: embeds the 3D scene, pushes settings, compares Boids / Rules / ML.

uv run streamlit run dashboard/app.py --server.port 8501
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

# `streamlit run dashboard/app.py` puts dashboard/ (not the repo root) on sys.path.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dashboard.api_client import ApiClient, ApiError
from dashboard.charts import (
    METRICS,
    comparison_bar,
    drop_short_runs,
    eval_bar,
    eval_summary_frame,
    learning_curve_figure,
    mode_summary,
    samples_frame,
    time_series_figure,
)

API_URL = os.environ.get("API_URL", "http://localhost:8000")
SCENE_URL = os.environ.get("SCENE_URL", API_URL.rstrip("/") + "/")
PRESETS = ["Calm School", "Predator Attack", "Food Rush", "ML vs Rules"]
ROOT = Path(__file__).resolve().parent.parent
EVAL_REPORT = ROOT / "outputs" / "eval" / "latest.json"
PPO_RUNS = ROOT / "outputs" / "ppo"
KPI_METRICS = ["survivalRate", "foodCollected", "meanEnergy", "dispersion", "fps", "stepMs"]


def api() -> ApiClient:
    return ApiClient(API_URL)


def sidebar_controls() -> tuple[str, dict]:
    st.sidebar.header("Experiment")
    experiment_id = st.sidebar.text_input("Experiment ID", value="demo", max_chars=64)
    st.sidebar.header("Scenario")
    settings = {
        "preset": st.sidebar.selectbox("Preset", ["(custom)", *PRESETS]),
        "count": st.sidebar.slider("Population", 100, 5000, 1500, 100),
        "seed": st.sidebar.number_input("Seed", 0, 99_999, 42),
        "mode": st.sidebar.radio("Policy", ["boids", "rules", "ml"], index=1, horizontal=True),
        "predators": st.sidebar.slider("Predators", 0, 5, 2),
        "food": st.sidebar.slider("Food patches", 0, 10, 4),
    }
    with st.sidebar.expander("Boids gains"):
        settings["weights"] = {k: st.slider(k, 0.0, 5.0, v, 0.05) for k, v in
                               {"separation": 1.6, "alignment": 1.0, "cohesion": 0.8, "predator": 3.0, "food": 1.0}.items()}
    return experiment_id, settings


def settings_payload(settings: dict) -> dict:
    """A preset overrides everything else; otherwise send the explicit values."""
    if settings["preset"] != "(custom)":
        return {"preset": settings["preset"]}
    return {k: v for k, v in settings.items() if k != "preset"}


def action_buttons(client: ApiClient, experiment_id: str, settings: dict) -> None:
    col1, col2 = st.sidebar.columns(2)
    if col1.button("Apply to scene", width="stretch"):
        _call(lambda: client.push_settings(experiment_id, settings_payload(settings)), "settings pushed")
    if col2.button("New run", width="stretch"):
        _call(lambda: client.reset(experiment_id, int(settings["seed"]), settings["mode"], settings["preset"]), "run reset")
    paused = st.sidebar.toggle("Pause scene", value=False)
    if st.sidebar.button("Apply pause state"):
        _call(lambda: client.push_settings(experiment_id, {"paused": paused}), "pause updated")


def _call(fn, ok_message: str) -> None:
    try:
        result = fn()
        st.sidebar.success(f"{ok_message} ({result.get('clients', '-')} scene client(s))" if isinstance(result, dict) else ok_message)
    except ApiError as exc:
        st.sidebar.error(str(exc))


def status_bar(client: ApiClient) -> None:
    try:
        health = client.health()
        ml = next(p for p in client.policies() if p["name"] == "ml")
    except ApiError as exc:
        st.error(f"Backend unreachable at {API_URL}: {exc}")
        return
    cols = st.columns(3)
    cols[0].metric("Backend", health["status"])
    cols[1].metric("CUDA", "available" if health["cuda"] else "CPU only")
    cols[2].metric("ML policy", ml["label"])


def scene(experiment_id: str) -> None:
    st.iframe(f"{SCENE_URL}?experiment={experiment_id}", height=620)


@st.fragment(run_every=2)
def live_metrics(client: ApiClient, experiment_id: str) -> None:
    try:
        metrics = client.metrics()
    except ApiError as exc:
        st.warning(str(exc))
        return
    df = samples_frame(metrics)
    if df.empty:
        st.info("No telemetry yet. Open the scene above; it reports metrics once per second.")
        return
    kpis(df[df["experiment_id"] == experiment_id])
    comparison(df)


def kpis(df) -> None:
    if df.empty:
        return
    last = df.iloc[-1]
    cols = st.columns(len(KPI_METRICS))
    for col, key in zip(cols, KPI_METRICS):
        if key in last:
            col.metric(METRICS[key], f"{last[key]:.3g}")


def comparison(df) -> None:
    df, dropped = drop_short_runs(df)
    if df.empty:
        st.info("No run is long enough to compare yet.")
        return
    st.subheader("Time-series comparison: Boids / Rules / ML")
    if dropped:
        st.caption(f"{dropped} run(s) shorter than 5 s were excluded (e.g. a scene before a preset was applied).")
    metric = st.selectbox("Metric", [m for m in METRICS if m in df], format_func=METRICS.get)
    st.plotly_chart(time_series_figure(df, metric), width="stretch")
    summary = mode_summary(df, list(METRICS))
    st.subheader("Final value per run (mean ± std over runs / seeds)")
    st.plotly_chart(comparison_bar(summary, metric), width="stretch")
    st.dataframe(summary, width="stretch", hide_index=True)


def offline_evaluation() -> None:
    """Results of `training.evaluate` / `training.train_ppo` (held-out seeds, training env)."""
    st.header("Offline evaluation (training environment, held-out seeds)")
    if not EVAL_REPORT.exists():
        st.caption("Run `uv run python -m training.evaluate` to produce a report.")
        return
    report = json.loads(EVAL_REPORT.read_text())
    summary = eval_summary_frame(report)
    st.caption(f"policies {report['meta'].get('policies')} | seeds {report['meta'].get('seeds')}")
    cols = st.columns(3)
    for col, metric in zip(cols, ["return", "survival_rate", "food_collected"]):
        col.plotly_chart(eval_bar(summary, metric), width="stretch")
    st.dataframe(summary.round(3), width="stretch", hide_index=True)
    curves = {p.parent.name: pd.read_csv(p) for p in sorted(PPO_RUNS.glob("*/learning_curve.csv"))}
    if curves:
        st.plotly_chart(learning_curve_figure(curves), width="stretch")


def main() -> None:
    st.set_page_config(page_title="Smart 3D Ecosystem", layout="wide")
    st.title("SMART 3D ECOSYSTEM")
    client = api()
    experiment_id, settings = sidebar_controls()
    action_buttons(client, experiment_id, settings)
    status_bar(client)
    scene(experiment_id)
    live_metrics(client, experiment_id)
    offline_evaluation()


main()
