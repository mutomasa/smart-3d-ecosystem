from shared.contract import load_contract


def test_contract_dimensions():
    contract = load_contract()
    assert contract.obs_dim == 20
    assert contract.action_dim == 6


def test_feature_index_lookup():
    contract = load_contract()
    assert contract.feature_index("group_center_x") == 0
    assert contract.feature_index("food_present") == contract.obs_dim - 1
