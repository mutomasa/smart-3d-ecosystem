from shared.contract import load_contract

CONTRACT = load_contract()


def features(**overrides: float) -> list[float]:
    f = [0.0] * CONTRACT.obs_dim
    f[CONTRACT.feature_index("nearest_predator_distance")] = CONTRACT.max_distance
    f[CONTRACT.feature_index("nearest_food_distance")] = CONTRACT.max_distance
    f[CONTRACT.feature_index("mean_energy")] = 1.0
    for k, v in overrides.items():
        f[CONTRACT.feature_index(k)] = v
    return f


def policy_request(n_groups: int = 2, request_id: str = "req-1", tick: int = 1, policy: str = "rules", **kw) -> dict:
    return {
        "type": "policy_request", "request_id": request_id, "tick": tick, "policy": policy,
        "observations": [{"group_id": g, "features": features(**kw)} for g in range(n_groups)],
    }
