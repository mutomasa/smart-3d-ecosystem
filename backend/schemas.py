"""Pydantic models for the HTTP API and the WebSocket protocol (section 8)."""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, Field, field_validator

from shared.contract import load_contract

CONTRACT = load_contract()
PolicyName = Literal["boids", "rules", "ml"]


class GroupObservation(BaseModel):
    group_id: Annotated[int, Field(ge=0, lt=CONTRACT.max_groups)]
    features: list[float]

    @field_validator("features")
    @classmethod
    def _fixed_length(cls, v: list[float]) -> list[float]:
        if len(v) != CONTRACT.obs_dim:
            raise ValueError(f"features must have length {CONTRACT.obs_dim}, got {len(v)}")
        return v


class PolicyRequest(BaseModel):
    type: Literal["policy_request"]
    request_id: str = Field(max_length=64)
    tick: int = Field(ge=0)
    policy: Literal["rules", "ml"] = "ml"
    observations: list[GroupObservation] = Field(max_length=CONTRACT.max_groups)


class GroupAction(BaseModel):
    group_id: int
    direction: list[float]
    weights: list[float]


class PolicyResponse(BaseModel):
    type: Literal["policy_response"] = "policy_response"
    request_id: str
    tick: int
    policy: str
    model_version: str
    latency_ms: float
    actions: list[GroupAction]


class ErrorMessage(BaseModel):
    type: Literal["error"] = "error"
    request_id: str | None = None
    code: str
    message: str
    fallback: Literal["rules"] = "rules"


class TelemetryMessage(BaseModel):
    """Low-rate (~1 Hz) metrics sample sent by the browser."""

    type: Literal["telemetry"]
    tick: int = Field(ge=0)
    mode: PolicyName
    metrics: dict[str, float]


class SimulationSettings(BaseModel):
    """Settings that the dashboard can push to the 3D scene."""

    preset: str | None = None
    count: Annotated[int, Field(ge=0, le=5000)] | None = None
    seed: Annotated[int, Field(ge=0)] | None = None
    mode: PolicyName | None = None
    predators: Annotated[int, Field(ge=0, le=5)] | None = None
    food: Annotated[int, Field(ge=0, le=10)] | None = None
    paused: bool | None = None
    weights: dict[str, Annotated[float, Field(ge=0, le=5)]] | None = None


class SettingsEnvelope(BaseModel):
    """Versioned settings push. Clients apply only versions newer than the last one seen."""

    type: Literal["settings"] = "settings"
    version: int
    settings: SimulationSettings


class ResetRequest(BaseModel):
    experiment_id: str | None = Field(default=None, max_length=64, pattern=r"^[A-Za-z0-9_.-]+$")
    seed: int = Field(default=42, ge=0)
    mode: PolicyName = "rules"
    scenario: str = "custom"


class ExperimentInfo(BaseModel):
    experiment_id: str
    seed: int
    mode: PolicyName
    scenario: str
    created_at: float
