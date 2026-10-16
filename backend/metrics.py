"""Per-experiment telemetry, latency statistics and settings (isolated by experiment ID)."""

from __future__ import annotations

import threading
import time
import uuid
from collections import deque
from dataclasses import dataclass, field

import numpy as np

from backend.schemas import ExperimentInfo, SimulationSettings

MAX_SAMPLES = 3600
MAX_LATENCIES = 2000


@dataclass
class Experiment:
    info: ExperimentInfo
    samples: deque = field(default_factory=lambda: deque(maxlen=MAX_SAMPLES))
    latencies_ms: deque = field(default_factory=lambda: deque(maxlen=MAX_LATENCIES))
    fallbacks: int = 0
    settings: SimulationSettings = field(default_factory=SimulationSettings)
    settings_version: int = 0


def percentile(values: list[float], q: float) -> float | None:
    return float(np.percentile(values, q)) if values else None


class ExperimentStore:
    """Thread-safe in-memory store. Each experiment ID has its own, non-shared state."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._experiments: dict[str, Experiment] = {}

    def reset(self, experiment_id: str | None, seed: int, mode: str, scenario: str) -> ExperimentInfo:
        info = ExperimentInfo(
            experiment_id=experiment_id or uuid.uuid4().hex[:12],
            seed=seed, mode=mode, scenario=scenario, created_at=time.time(),
        )
        with self._lock:
            previous = self._experiments.get(info.experiment_id)
            exp = Experiment(info=info)
            if previous:  # keep the settings version monotonic across resets
                exp.settings_version = previous.settings_version
            self._experiments[info.experiment_id] = exp
        return info

    def ensure(self, experiment_id: str) -> Experiment:
        with self._lock:
            exp = self._experiments.get(experiment_id)
        if exp is None:
            self.reset(experiment_id, seed=0, mode="rules", scenario="ad-hoc")
            with self._lock:
                exp = self._experiments[experiment_id]
        return exp

    def record_telemetry(self, experiment_id: str, tick: int, mode: str, metrics: dict[str, float]) -> None:
        exp = self.ensure(experiment_id)
        with self._lock:
            exp.samples.append({"t": time.time(), "tick": tick, "mode": mode, **metrics})

    def record_latency(self, experiment_id: str, latency_ms: float) -> None:
        exp = self.ensure(experiment_id)
        with self._lock:
            exp.latencies_ms.append(latency_ms)

    def record_fallback(self, experiment_id: str) -> None:
        exp = self.ensure(experiment_id)
        with self._lock:
            exp.fallbacks += 1

    def update_settings(self, experiment_id: str, patch: SimulationSettings) -> tuple[int, SimulationSettings]:
        """Merges `patch` into the stored settings (last writer wins) and bumps the version."""
        exp = self.ensure(experiment_id)
        with self._lock:
            merged = exp.settings.model_dump(exclude_none=True) | patch.model_dump(exclude_none=True)
            exp.settings = SimulationSettings(**merged)
            exp.settings_version += 1
            return exp.settings_version, exp.settings

    def settings(self, experiment_id: str) -> tuple[int, SimulationSettings]:
        exp = self.ensure(experiment_id)
        with self._lock:
            return exp.settings_version, exp.settings

    def summary(self, experiment_id: str) -> dict:
        exp = self.ensure(experiment_id)
        with self._lock:
            latencies = list(exp.latencies_ms)
            return {
                **exp.info.model_dump(),
                "samples": list(exp.samples),
                "latency_ms": {"p50": percentile(latencies, 50), "p95": percentile(latencies, 95), "n": len(latencies)},
                "fallbacks": exp.fallbacks,
            }

    def list(self) -> list[ExperimentInfo]:
        with self._lock:
            return [e.info for e in self._experiments.values()]
