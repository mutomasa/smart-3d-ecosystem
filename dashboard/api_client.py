"""Thin HTTP client for the FastAPI backend (used by the Streamlit dashboard)."""

from __future__ import annotations

from dataclasses import dataclass, field

import requests


class ApiError(RuntimeError):
    pass


@dataclass
class ApiClient:
    base_url: str
    timeout_s: float = 2.0
    session: requests.Session = field(default_factory=requests.Session)

    def health(self) -> dict:
        return self._get("/health")

    def policies(self) -> list[dict]:
        return self._get("/api/policies")

    def experiments(self) -> list[dict]:
        return self._get("/api/experiments")

    def metrics(self, experiment_id: str | None = None) -> dict:
        params = {"experiment_id": experiment_id} if experiment_id else None
        return self._get("/api/metrics", params=params)

    def reset(self, experiment_id: str, seed: int, mode: str, scenario: str) -> dict:
        body = {"experiment_id": experiment_id, "seed": seed, "mode": mode, "scenario": scenario}
        return self._post("/api/experiments/reset", body)

    def push_settings(self, experiment_id: str, settings: dict) -> dict:
        return self._post(f"/api/experiments/{experiment_id}/settings", settings)

    def _get(self, path: str, params: dict | None = None):
        return self._request("GET", path, params=params)

    def _post(self, path: str, body: dict):
        return self._request("POST", path, json=body)

    def _request(self, method: str, path: str, **kwargs):
        url = self.base_url.rstrip("/") + path
        try:
            resp = self.session.request(method, url, timeout=self.timeout_s, **kwargs)
            resp.raise_for_status()
            return resp.json()
        except (requests.RequestException, ValueError) as exc:
            raise ApiError(f"{method} {path} failed: {exc}") from exc
