import pytest
from fastapi.testclient import TestClient

from backend.main import create_app
from backend.metrics import ExperimentStore
from backend.policy import PolicyRegistry, RulesPolicy


@pytest.fixture(autouse=True)
def _no_real_checkpoint(monkeypatch, tmp_path):
    """Tests never pick up a locally trained backend/models/policy.pt."""
    monkeypatch.setenv("POLICY_CHECKPOINT", str(tmp_path / "missing.pt"))
    monkeypatch.setenv("FORCE_CPU", "1")


@pytest.fixture
def store() -> ExperimentStore:
    return ExperimentStore()


@pytest.fixture
def make_client(store):
    def _make(registry: PolicyRegistry | None = None) -> TestClient:
        return TestClient(create_app(registry or PolicyRegistry({"rules": RulesPolicy()}), store))

    return _make


@pytest.fixture
def client(make_client) -> TestClient:
    return make_client()
