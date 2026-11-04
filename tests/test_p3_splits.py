import pytest

from wsp.predictor.splits import make_splits, trajectory_groups


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


def _ancestry_rows():
    # a -> m -> z (m has no target), a -> m2 -> y, independent b..e
    ancestry = {"a": None, "m": "a", "z": "m", "m2": "a", "y": "m2"}
    ancestry |= {k: None for k in "bcde"}
    return ancestry


def test_trajectory_groups_survive_removed_intermediate_entries():
    ancestry = _ancestry_rows()
    kept = [k for k in ancestry if k not in ("m", "m2")]  # intermediates have no eval.json
    rows = [{"id": k, "parent_id": ancestry[k]} for k in kept]
    for n_splits in (2, 3, 4):
        for _train, test in make_splits(rows, "trajectory", n_splits, ancestry=ancestry):
            test_ids = {rows[i]["id"] for i in test}
            chain = {"a", "z", "y"}
            assert chain <= test_ids or chain.isdisjoint(test_ids)
    # without the zoo-wide ancestry the removed intermediates disconnect the chain (the old leak)
    groups = trajectory_groups(rows)
    assert len({groups[kept.index(k)] for k in ("a", "z")}) == 2
    groups = trajectory_groups(rows, ancestry)
    assert len({groups[kept.index(k)] for k in ("a", "z", "y")}) == 1


def test_ancestry_conflict_and_cycle_are_rejected():
    with pytest.raises(ValueError, match="conflicting parent"):
        trajectory_groups([{"id": "a", "parent_id": "p"}], {"a": "q"})
    with pytest.raises(ValueError, match="cycle"):
        trajectory_groups([{"id": "a", "parent_id": "x"}], {"x": "y", "y": "x"})
