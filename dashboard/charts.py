"""Pure data shaping + Plotly figures for the dashboard (unit-testable without Streamlit)."""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

METRICS = {
    "survivalRate": "Survival Rate",
    "foodCollected": "Food Collected",
    "meanEnergy": "Mean Energy",
    "dispersion": "Group Dispersion",
    "collisions": "Collision Count",
    "fps": "Render FPS",
    "stepMs": "Simulation Step ms",
    "mlLatencyP50": "ML Latency p50 (ms)",
    "mlLatencyP95": "ML Latency p95 (ms)",
}
MODE_COLORS = {"boids": "#7f8c9a", "rules": "#2a9df4", "ml": "#f5a623"}
SIM_HZ = 60.0


def samples_frame(metrics: dict) -> pd.DataFrame:
    """Flattens `/api/metrics` into one row per sample with experiment metadata and run ids."""
    frames = [_experiment_frame(eid, body) for eid, body in metrics.items() if body.get("samples")]
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def _experiment_frame(experiment_id: str, body: dict) -> pd.DataFrame:
    df = pd.DataFrame(body["samples"])
    df["experiment_id"] = experiment_id
    df["seed"] = df["seed"].astype(int) if "seed" in df else body.get("seed", 0)
    df["run"] = assign_runs(df)
    df["sim_time_s"] = df["tick"] / SIM_HZ
    df["run_label"] = df.apply(lambda r: f"{experiment_id}#{r['run']} {r['mode']} seed={r['seed']}", axis=1)
    return df


def assign_runs(df: pd.DataFrame) -> pd.Series:
    """A new run starts when the tick goes backwards (reset) or the mode / seed changes."""
    tick_reset = df["tick"].diff().fillna(0) < 0
    changed = (df["mode"] != df["mode"].shift()) | (df["seed"] != df["seed"].shift())
    changed.iloc[0] = False
    return (tick_reset | changed).cumsum()


def final_per_run(df: pd.DataFrame) -> pd.DataFrame:
    """Last sample of every run (the value a run ended with)."""
    if df.empty:
        return df
    return df.sort_values("tick").groupby(["experiment_id", "run"], as_index=False).last()


def mode_summary(df: pd.DataFrame, metrics: list[str]) -> pd.DataFrame:
    """Mean and std of the final value of each run, per mode (runs = seeds / repetitions)."""
    finals = final_per_run(df)
    if finals.empty:
        return pd.DataFrame()
    present = [m for m in metrics if m in finals]
    agg = finals.groupby("mode")[present].agg(["mean", "std", "count"])
    agg.columns = [f"{m} {stat}" for m, stat in agg.columns]
    return agg.reset_index()


def time_series_figure(df: pd.DataFrame, metric: str) -> go.Figure:
    fig = px.line(
        df, x="sim_time_s", y=metric, color="mode", line_group="run_label", hover_name="run_label",
        color_discrete_map=MODE_COLORS, labels={"sim_time_s": "simulated time (s)", metric: METRICS.get(metric, metric)},
    )
    fig.update_layout(height=320, margin={"l": 10, "r": 10, "t": 30, "b": 10}, legend_title_text="mode")
    return fig


def comparison_bar(summary: pd.DataFrame, metric: str) -> go.Figure:
    fig = go.Figure()
    if f"{metric} mean" in summary:
        fig.add_bar(
            x=summary["mode"], y=summary[f"{metric} mean"],
            error_y={"type": "data", "array": summary[f"{metric} std"].fillna(0)},
            marker_color=[MODE_COLORS.get(m, "#999") for m in summary["mode"]],
        )
    fig.update_layout(height=280, margin={"l": 10, "r": 10, "t": 30, "b": 10}, title=METRICS.get(metric, metric))
    return fig


def eval_summary_frame(report: dict) -> pd.DataFrame:
    """`outputs/eval/latest.json` -> one row per policy (mean / std columns)."""
    return pd.DataFrame(report.get("summary", []))


def eval_bar(summary: pd.DataFrame, metric: str) -> go.Figure:
    fig = go.Figure()
    if f"{metric}_mean" in summary:
        fig.add_bar(x=summary["policy"], y=summary[f"{metric}_mean"],
                    error_y={"type": "data", "array": summary[f"{metric}_std"].fillna(0)})
    fig.update_layout(height=280, margin={"l": 10, "r": 10, "t": 30, "b": 10}, title=metric)
    return fig


def learning_curve_figure(curves: dict[str, pd.DataFrame]) -> go.Figure:
    fig = go.Figure()
    for name, df in curves.items():
        if {"timesteps", "return_ma"} <= set(df.columns):
            fig.add_scatter(x=df["timesteps"], y=df["return_ma"], mode="lines", name=name)
    fig.update_layout(height=300, margin={"l": 10, "r": 10, "t": 30, "b": 10},
                      title="PPO learning curve (episode return, moving average)",
                      xaxis_title="timesteps", yaxis_title="return")
    return fig
