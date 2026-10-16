"""FastAPI application: REST API, WebSocket and static hosting of the built 3D scene."""

from __future__ import annotations

import logging
from pathlib import Path

import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from backend.metrics import ExperimentStore
from backend.ml_policy import build_registry
from backend.policy import PolicyRegistry
from backend.schemas import CONTRACT, ResetRequest, SettingsEnvelope, SimulationSettings
from backend.ws import ConnectionManager, PolicyService, build_router

FRONTEND_DIST = Path(__file__).resolve().parent.parent / "frontend" / "dist"
SIM_DEFAULTS = {
    "count": 1500, "seed": 42, "dt": 1 / 60, "half_extents": list(CONTRACT.world_half_extents),
    "group_count": 16, "policy_hz": 10, "max_groups": CONTRACT.max_groups,
    "contract_version": CONTRACT.version,
}

logging.basicConfig(level=logging.INFO)


def create_app(registry: PolicyRegistry | None = None, store: ExperimentStore | None = None) -> FastAPI:
    registry = registry or build_registry()
    store = store or ExperimentStore()
    manager = ConnectionManager()
    service = PolicyService(registry, timeout_s=CONTRACT.request_timeout_ms / 1000)
    app = FastAPI(title="Smart 3D Ecosystem API")
    app.state.registry, app.state.store, app.state.manager = registry, store, manager
    app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
    _add_routes(app, registry, store, manager)
    app.include_router(build_router(store, service, manager))
    _mount_frontend(app)
    return app


def _add_routes(app: FastAPI, registry: PolicyRegistry, store: ExperimentStore, manager: ConnectionManager) -> None:
    @app.get("/health")
    def health() -> dict:
        ml = next(p for p in registry.describe() if p.name == "ml")
        return {"status": "ok", "cuda": torch.cuda.is_available(), "ml_loaded": ml.available, "ml_kind": ml.kind}

    @app.get("/api/config")
    def config() -> dict:
        return SIM_DEFAULTS

    @app.get("/api/policies")
    def policies() -> list[dict]:
        return [p.__dict__ for p in registry.describe()]

    @app.get("/api/experiments")
    def experiments() -> list[dict]:
        return [e.model_dump() for e in store.list()]

    @app.get("/api/metrics")
    def metrics(experiment_id: str | None = None) -> dict:
        """Telemetry history (last 3,600 samples per experiment), latency p50/p95, fallbacks."""
        ids = [experiment_id] if experiment_id else [e.experiment_id for e in store.list()]
        return {eid: store.summary(eid) for eid in ids}

    @app.post("/api/experiments/reset")
    def reset(req: ResetRequest) -> dict:
        return store.reset(req.experiment_id, req.seed, req.mode, req.scenario).model_dump()

    @app.post("/api/experiments/{experiment_id}/settings")
    async def push_settings(experiment_id: str, patch: SimulationSettings) -> dict:
        if len(experiment_id) > 64:
            raise HTTPException(422, "experiment_id too long")
        version, settings = store.update_settings(experiment_id, patch)
        envelope = SettingsEnvelope(version=version, settings=settings).model_dump(exclude_none=True)
        await manager.broadcast(experiment_id, envelope)
        return {**envelope, "clients": manager.count(experiment_id)}


def _mount_frontend(app: FastAPI) -> None:
    if FRONTEND_DIST.exists():
        app.mount("/", StaticFiles(directory=FRONTEND_DIST, html=True), name="frontend")


app = create_app()
