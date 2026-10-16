"""WebSocket endpoint ``/ws/simulation/{experiment_id}``.

Handles batched policy requests (with size limit, schema validation and timeout), low-rate
telemetry, and pushes versioned settings to every client of the same experiment.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from collections import defaultdict

import numpy as np
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from pydantic import ValidationError

from backend.metrics import ExperimentStore
from backend.policy import PolicyRegistry
from backend.schemas import (
    CONTRACT,
    ErrorMessage,
    GroupAction,
    PolicyRequest,
    PolicyResponse,
    SettingsEnvelope,
    TelemetryMessage,
)

log = logging.getLogger(__name__)


class ConnectionManager:
    """Tracks sockets per experiment so that broadcasts never cross experiments."""

    def __init__(self) -> None:
        self._clients: dict[str, set[WebSocket]] = defaultdict(set)

    def add(self, experiment_id: str, ws: WebSocket) -> None:
        self._clients[experiment_id].add(ws)

    def remove(self, experiment_id: str, ws: WebSocket) -> None:
        self._clients[experiment_id].discard(ws)

    def count(self, experiment_id: str) -> int:
        return len(self._clients[experiment_id])

    async def broadcast(self, experiment_id: str, message: dict) -> None:
        for ws in list(self._clients[experiment_id]):
            try:
                await ws.send_json(message)
            except (RuntimeError, WebSocketDisconnect, OSError):  # a dead socket must not break the others
                self.remove(experiment_id, ws)


class PolicyService:
    """Runs a policy on a batch with a timeout (inference happens in a worker thread)."""

    def __init__(self, registry: PolicyRegistry, timeout_s: float) -> None:
        self.registry = registry
        self.timeout_s = timeout_s

    async def respond(self, req: PolicyRequest) -> PolicyResponse | ErrorMessage:
        policy = self.registry.get(req.policy)
        if policy is None:
            return ErrorMessage(request_id=req.request_id, code="policy_unavailable", message=f"{req.policy} not loaded")
        obs = np.array([o.features for o in req.observations], dtype=np.float32).reshape(-1, CONTRACT.obs_dim)
        start = time.perf_counter()
        try:
            actions = await asyncio.wait_for(asyncio.to_thread(policy.act, obs), self.timeout_s)
        except TimeoutError:
            return ErrorMessage(request_id=req.request_id, code="timeout", message="inference timed out")
        except Exception as exc:  # any inference failure => client falls back to rules
            log.exception("policy failure")
            return ErrorMessage(request_id=req.request_id, code="inference_error", message=str(exc))
        latency = (time.perf_counter() - start) * 1000
        return self._response(req, policy, actions, latency)

    @staticmethod
    def _response(req: PolicyRequest, policy, actions: np.ndarray, latency: float) -> PolicyResponse:
        items = [
            GroupAction(group_id=o.group_id, direction=a[:3].tolist(), weights=a[3:].tolist())
            for o, a in zip(req.observations, actions)
        ]
        return PolicyResponse(
            request_id=req.request_id, tick=req.tick, policy=policy.name,
            model_version=policy.model_version, latency_ms=latency, actions=items,
        )


class SimulationSocket:
    """Message loop of one connection."""

    def __init__(self, ws: WebSocket, experiment_id: str, store: ExperimentStore, service: PolicyService) -> None:
        self.ws, self.experiment_id, self.store, self.service = ws, experiment_id, store, service

    async def run(self) -> None:
        await self._send_current_settings()
        while True:
            text = await self.ws.receive_text()
            reply = await self.handle(text)
            if reply is not None:
                await self.ws.send_json(reply)

    async def handle(self, text: str) -> dict | None:
        if len(text.encode("utf-8")) > CONTRACT.max_message_bytes:
            return ErrorMessage(code="too_large", message="message exceeds max_message_bytes").model_dump()
        try:
            data = json.loads(text)
            kind = data.get("type") if isinstance(data, dict) else None
        except json.JSONDecodeError:
            return ErrorMessage(code="bad_json", message="invalid JSON").model_dump()
        if kind == "policy_request":
            return await self._policy(data)
        if kind == "telemetry":
            return self._telemetry(data)
        return ErrorMessage(code="unknown_type", message=f"unknown message type {kind!r}").model_dump()

    async def _policy(self, data: dict) -> dict:
        try:
            req = PolicyRequest.model_validate(data)
        except ValidationError as exc:
            self.store.record_fallback(self.experiment_id)
            return ErrorMessage(request_id=data.get("request_id"), code="invalid_schema", message=str(exc)[:500]).model_dump()
        result = await self.service.respond(req)
        if isinstance(result, ErrorMessage):
            self.store.record_fallback(self.experiment_id)
        else:
            self.store.record_latency(self.experiment_id, result.latency_ms)
        return result.model_dump()

    def _telemetry(self, data: dict) -> dict | None:
        try:
            msg = TelemetryMessage.model_validate(data)
        except ValidationError as exc:
            return ErrorMessage(code="invalid_schema", message=str(exc)[:500]).model_dump()
        self.store.record_telemetry(self.experiment_id, msg.tick, msg.mode, msg.metrics)
        return None

    async def _send_current_settings(self) -> None:
        version, settings = self.store.settings(self.experiment_id)
        if version > 0:
            await self.ws.send_json(SettingsEnvelope(version=version, settings=settings).model_dump(exclude_none=True))


def build_router(store: ExperimentStore, service: PolicyService, manager: ConnectionManager) -> APIRouter:
    router = APIRouter()

    @router.websocket("/ws/simulation/{experiment_id}")
    async def simulation_ws(ws: WebSocket, experiment_id: str) -> None:
        await ws.accept()
        manager.add(experiment_id, ws)
        store.ensure(experiment_id)
        try:
            await SimulationSocket(ws, experiment_id, store, service).run()
        except WebSocketDisconnect:
            pass
        finally:
            manager.remove(experiment_id, ws)

    return router
