import json
import time

import numpy as np

from backend.policy import PolicyRegistry
from tests.helpers import CONTRACT, policy_request


class SlowPolicy:
    name, kind, model_version = "ml", "test", "slow"

    def act(self, raw_obs):
        time.sleep(1.0)
        return np.zeros((len(raw_obs), 6), dtype=np.float32)


class BrokenPolicy:
    name, kind, model_version = "ml", "test", "broken"

    def act(self, raw_obs):
        raise RuntimeError("boom")


def test_policy_request_returns_one_action_per_group(client):
    with client.websocket_connect("/ws/simulation/exp1") as ws:
        ws.send_json(policy_request(3, predator_present=1, nearest_predator_relative_x=2, nearest_predator_distance=2))
        msg = ws.receive_json()
    assert msg["type"] == "policy_response" and msg["request_id"] == "req-1" and msg["tick"] == 1
    assert [a["group_id"] for a in msg["actions"]] == [0, 1, 2]
    assert msg["actions"][0]["direction"] == [-1.0, 0.0, 0.0]
    assert msg["latency_ms"] >= 0


def test_invalid_schema_falls_back_to_rules(client, store):
    bad = policy_request(1)
    bad["observations"][0]["features"] = [0.0] * 3
    with client.websocket_connect("/ws/simulation/exp1") as ws:
        ws.send_json(bad)
        msg = ws.receive_json()
    assert msg["type"] == "error" and msg["code"] == "invalid_schema" and msg["fallback"] == "rules"
    assert store.summary("exp1")["fallbacks"] == 1


def test_too_many_groups_and_too_large_messages_are_rejected(client):
    with client.websocket_connect("/ws/simulation/exp1") as ws:
        ws.send_json(policy_request(CONTRACT.max_groups + 1))
        assert ws.receive_json()["code"] == "invalid_schema"
        ws.send_text("x" * (CONTRACT.max_message_bytes + 1))
        assert ws.receive_json()["code"] == "too_large"
        ws.send_text("{not json")
        assert ws.receive_json()["code"] == "bad_json"


def test_unavailable_policy_reports_error(client):
    with client.websocket_connect("/ws/simulation/exp1") as ws:
        ws.send_json(policy_request(1, policy="ml"))
        msg = ws.receive_json()
    assert msg["code"] == "policy_unavailable"


def test_inference_timeout_falls_back(make_client):
    client = make_client(PolicyRegistry({"ml": SlowPolicy()}))
    with client.websocket_connect("/ws/simulation/exp1") as ws:
        ws.send_json(policy_request(1, policy="ml"))
        msg = ws.receive_json()
    assert msg["code"] == "timeout" and msg["fallback"] == "rules"


def test_inference_error_falls_back(make_client):
    client = make_client(PolicyRegistry({"ml": BrokenPolicy()}))
    with client.websocket_connect("/ws/simulation/exp1") as ws:
        ws.send_json(policy_request(1, policy="ml"))
        assert ws.receive_json()["code"] == "inference_error"


def test_telemetry_is_stored_per_experiment_without_crosstalk(client):
    with client.websocket_connect("/ws/simulation/A") as a, client.websocket_connect("/ws/simulation/B") as b:
        a.send_json({"type": "telemetry", "tick": 10, "mode": "rules", "metrics": {"survivalRate": 0.9}})
        b.send_json({"type": "telemetry", "tick": 20, "mode": "boids", "metrics": {"survivalRate": 0.5}})
        a.send_json(policy_request(1))  # round-trip to make sure telemetry was processed
        a.receive_json()
        b.send_json(policy_request(1))
        b.receive_json()
    metrics = client.get("/api/metrics").json()
    assert [s["survivalRate"] for s in metrics["A"]["samples"]] == [0.9]
    assert [s["survivalRate"] for s in metrics["B"]["samples"]] == [0.5]
    assert metrics["A"]["latency_ms"]["n"] == 1


def test_settings_are_broadcast_only_to_the_same_experiment(client):
    with client.websocket_connect("/ws/simulation/A") as a, client.websocket_connect("/ws/simulation/B") as b:
        resp = client.post("/api/experiments/A/settings", json={"mode": "ml", "predators": 3}).json()
        assert resp["clients"] == 1
        msg = a.receive_json()
        assert msg == {"type": "settings", "version": 1, "settings": {"mode": "ml", "predators": 3}}
        b.send_json(policy_request(1))
        assert b.receive_json()["type"] == "policy_response"  # B received no settings push


def test_settings_are_replayed_on_connect_with_increasing_versions(client):
    client.post("/api/experiments/C/settings", json={"seed": 5})
    client.post("/api/experiments/C/settings", json={"food": 4})
    with client.websocket_connect("/ws/simulation/C") as ws:
        msg = ws.receive_json()
    assert msg["version"] == 2 and msg["settings"] == {"seed": 5, "food": 4}


def test_unknown_message_type(client):
    with client.websocket_connect("/ws/simulation/x") as ws:
        ws.send_text(json.dumps({"type": "hello"}))
        assert ws.receive_json()["code"] == "unknown_type"
