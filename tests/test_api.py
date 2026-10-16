def test_health_reports_cuda_and_model_state(client):
    body = client.get("/health").json()
    assert body["status"] == "ok"
    assert isinstance(body["cuda"], bool)
    assert body["ml_loaded"] is False


def test_config_matches_contract(client):
    body = client.get("/api/config").json()
    assert body["half_extents"] == [50.0, 25.0, 50.0]
    assert body["max_groups"] == 32


def test_policies_lists_boids_rules_ml(client):
    names = {p["name"]: p for p in client.get("/api/policies").json()}
    assert set(names) == {"boids", "rules", "ml"}
    assert names["ml"]["available"] is False


def test_reset_creates_experiment_with_seed(client):
    info = client.post("/api/experiments/reset", json={"experiment_id": "exp-a", "seed": 7, "mode": "rules"}).json()
    assert info["experiment_id"] == "exp-a" and info["seed"] == 7
    generated = client.post("/api/experiments/reset", json={"seed": 1}).json()
    assert len(generated["experiment_id"]) == 12
    ids = {e["experiment_id"] for e in client.get("/api/experiments").json()}
    assert {"exp-a", generated["experiment_id"]} <= ids


def test_reset_rejects_bad_experiment_id(client):
    assert client.post("/api/experiments/reset", json={"experiment_id": "../x"}).status_code == 422


def test_metrics_for_unknown_experiment_is_empty(client):
    body = client.get("/api/metrics", params={"experiment_id": "nope"}).json()
    assert body["nope"]["samples"] == []
    assert body["nope"]["latency_ms"]["p50"] is None


def test_settings_validation(client):
    assert client.post("/api/experiments/e1/settings", json={"predators": 9}).status_code == 422
    ok = client.post("/api/experiments/e1/settings", json={"predators": 2}).json()
    assert ok["version"] == 1 and ok["settings"] == {"predators": 2}
