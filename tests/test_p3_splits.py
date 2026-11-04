import pytest

from wsp.predictor.splits import make_splits


@pytest.mark.parametrize("mode", ["family", "seed", "trajectory"])
def test_group_splits_do_not_leak(mode):
    rows = [
        {"id": "a", "family": "x", "seed": 1, "parent_id": None},
        {"id": "b", "family": "x", "seed": 1, "parent_id": "a"},
        {"id": "c", "family": "y", "seed": 2, "parent_id": None},
        {"id": "d", "family": "y", "seed": 2, "parent_id": "c"},
    ]
    for train, test in make_splits(rows, mode, n_splits=2):
        assert set(train).isdisjoint(test)
        key = {"family": "family", "seed": "seed"}.get(mode)
        if key:
            assert {rows[i][key] for i in train}.isdisjoint(rows[i][key] for i in test)
        else:
            assert (0 in train) == (1 in train)
            assert (2 in train) == (3 in train)


def test_missing_parent_still_groups_siblings():
    rows = [{"id": x, "parent_id": "external"} for x in ("a", "b")]
    rows += [{"id": "c", "parent_id": None}]
    for train, _ in make_splits(rows, "trajectory", n_splits=2):
        assert (0 in train) == (1 in train)
