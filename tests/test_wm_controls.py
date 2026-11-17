"""wm.controls: the SYNTHESIS-r9 C2 red-team controls to the Board A contract (conditions 5, 6, 10).

* fixed-endpoint control: consistent alternate decompositions of the same realised final edit, static invariant;
* corruption diagnostic (descriptive): inconsistent shuffles; a static cumulative predictor is exactly invariant;
* inert-edit generators (dead vocabulary rows; post-norm-cancelled k_proj scales) and the |yhat| <= 2 sigma check;
* trust-region stratification at WikiText ΔNLL 0.1;
* direct best-of-N with a two-stage screen at the equal real-evaluation budget;
* unamortised cost accounting.
"""

import numpy as np
import pytest
import torch

from wsp.wm import access as X
from wsp.wm import actions as A
from wsp.wm import controls as C
from wsp.wm import model as M
from wsp.wm import synthetic as S
from wsp.wm.plan import BudgetExceeded, BudgetLedger

CFG = M.WMConfig(epochs=30, hidden=16, latent=8)


def _retoken(world):
    return lambda actions: (S.action_tokens(world, actions), S.bookkeeping(world, actions))


# ------------------------------------------------------------------ shuffled bookkeeping


def test_shuffle_holds_cumulative_edit_and_changes_order_and_local_inputs():
    world = S.SynthWorld(kind="saturating", seed=0)
    traj = world.trajectories(n=1, horizon=4, seed=1)[0]
    sh = C.shuffle_bookkeeping(traj, J=world.J, seed=5, retoken=_retoken(world))
    T, J = 4, world.J
    np.testing.assert_array_equal(sh["x"][T, :J], traj["x"][T, :J])
    np.testing.assert_array_equal(sh["y"][T], traj["y"][T])
    assert np.isnan(sh["y"][1:T]).all()  # intermediate labels are not those of the shuffled path
    assert sh["actions"] != traj["actions"] and sorted(sh["actions"]) == sorted(traj["actions"])
    # step-local inputs are decoupled from the (permuted) step tokens
    D_rows = sh["x"][1:, J : 2 * J]
    e_rows = sh["e"]
    assert not np.array_equal(D_rows, e_rows)
    assert sorted(map(tuple, D_rows)) == sorted(map(tuple, e_rows))
    np.testing.assert_array_equal(sh["obs"][0], traj["obs"][0])
    assert (
        C.shuffle_bookkeeping(traj, J=J, seed=5, retoken=_retoken(world))["actions"]
        == sh["actions"]
    )


def test_shuffle_refuses_horizon_one():
    world = S.SynthWorld(kind="saturating", seed=0)
    traj = world.trajectories(n=1, horizon=1, seed=1)[0]
    with pytest.raises(ValueError):
        C.shuffle_bookkeeping(traj, J=world.J, seed=0, retoken=_retoken(world))


@pytest.fixture(scope="module")
def sat_models():
    world = S.SynthWorld(kind="saturating", seed=0)
    train = world.trajectories(n=60, horizon=2, seed=1)
    test = world.trajectories(n=16, horizon=4, seed=2) + world.trajectories(n=16, horizon=3, seed=3)
    st = M.StaticPredictor(CFG, world.field(), n_seeds=2).fit(train)
    return world, test, st


def test_static_predictor_is_exactly_invariant_under_corruption(sat_models):
    world, test, st = sat_models
    out = C.corruption_diagnostic(
        st, world.base_observation(), test, J=world.J, retoken=_retoken(world)
    )
    assert out["mean_diff"] == 0.0 and out["sensitive"] is False and out["n"] == 32
    assert out["gates"] is None  # descriptive only
    assert out["horizons"] == [3, 4]


class _OrderModel:
    """A predictor that genuinely uses step order: yhat_T = sum_t t * <w, D_t>."""

    def __init__(self, J, K):
        self.guard = X.R0Guard(S.BASE_HASH)
        self.w = np.linspace(-1, 1, J)
        self.K = K

    def _f(self, traj):
        J = len(self.w)
        D = traj["x"][1:, J : 2 * J]
        v = np.cumsum([(t + 1) * D[t] @ self.w for t in range(len(D))])
        out = np.zeros((len(D) + 1, self.K))
        out[1:] = v[:, None]
        return out

    @X.r0_only
    def predict(self, base_obs, traj):
        f = self._f(traj)
        return f, np.ones_like(f)


def test_corruption_diagnostic_detects_a_model_that_uses_order():
    world = S.SynthWorld(kind="saturating", seed=0)
    m = _OrderModel(world.J, world.K)
    test = world.trajectories(n=40, horizon=4, seed=4)
    test = [dict(t, y=m._f(t)) for t in test]  # the order model is exact on the true paths
    out = C.corruption_diagnostic(
        m, world.base_observation(), test, J=world.J, retoken=_retoken(world)
    )
    assert out["mean_orig"] == 0.0 and out["mean_shuf"] > 0 and out["p"] < 0.05 and out["sensitive"]


def test_null_clusters_by_component():
    world = S.SynthWorld(kind="saturating", seed=0)
    m = _OrderModel(world.J, world.K)
    test = [dict(t, y=m._f(t)) for t in world.trajectories(n=12, horizon=3, seed=6)]
    comps = [i // 4 for i in range(12)]
    out = C.corruption_diagnostic(
        m, world.base_observation(), test, J=world.J, retoken=_retoken(world), components=comps
    )
    assert out["n_clusters"] == 3


# ------------------------------------------------------------------ fixed-endpoint control


def _realise(world):
    def f(actions):
        coef = np.zeros(world.J)
        for j, d in actions:
            coef[j] += d * world.step
        return coef

    return f


def test_consistent_decompositions_hold_the_realised_endpoint_and_bookkeeping_is_consistent():
    world = S.SynthWorld(kind="saturating", seed=0)
    traj = world.trajectories(n=1, horizon=3, seed=1)[0]
    alts = C.consistent_decompositions(traj, _retoken(world), _realise(world), n=8, seed=0)
    assert 1 <= len(alts) <= 8
    assert any(a["horizon"] == 4 for a in alts)  # a split step
    for a in alts:
        np.testing.assert_array_equal(
            _realise(world)(a["actions"]), _realise(world)(traj["actions"])
        )
        np.testing.assert_array_equal(
            a["x"], S.bookkeeping(world, a["actions"])
        )  # consistent inputs
        np.testing.assert_array_equal(a["y"][a["horizon"]], traj["y"][3])
        assert a["actions"] != traj["actions"]


def test_noncommuting_steps_have_no_alternate():
    world = S.SynthWorld(kind="saturating", seed=0)
    traj = world.trajectories(n=1, horizon=3, seed=1)[0]
    order_dependent = lambda acts: np.array([hash(tuple(acts))], dtype=float)
    assert C.consistent_decompositions(traj, _retoken(world), order_dependent) == []


def test_fixed_endpoint_control_static_is_invariant_and_wm_equal_static_fails(sat_models):
    world, test, st = sat_models
    out = C.fixed_endpoint_control(
        st, st, world.base_observation(), test, _retoken(world), _realise(world)
    )
    assert out["static_invariant"] and out["static_max_deviation"] <= 1e-6
    assert out["ct_alternates"]["ct"] == pytest.approx(0.0, abs=1e-6)
    assert out["pass"] is False and out["n_states"] == 32


def test_fixed_endpoint_control_exposes_a_path_memorising_model(sat_models):
    world, _, st = sat_models
    m = _OrderModel(world.J, world.K)
    test = [dict(t, y=m._f(t)) for t in world.trajectories(n=30, horizon=4, seed=8)]
    out = C.fixed_endpoint_control(
        m, st, world.base_observation(), test, _retoken(world), _realise(world)
    )
    assert out["ct_original"]["ct"] == pytest.approx(1.0)
    assert out["ct_alternates"]["ct"] < out["ct_original"]["ct"]
    assert out["wm_path_shift"] > 0


# ------------------------------------------------------------------ inert edits


def test_synthetic_inert_directions_do_not_move_outcomes():
    world = C.synthetic_inert_world(seed=0, n_dead=2, n_cancel=1)
    assert world.inert == {world.J - 3: "dead", world.J - 2: "dead", world.J - 1: "cancel"}
    for j in world.inert:
        for dose in (-2.0, 1.0, 2.0):
            traj = world.trajectory_from_actions([(j, dose)])
            assert np.abs(traj["y"][1]).max() < 1e-5
        g = world.field().payload["g"]
        assert np.abs(g[:, j]).max() < 1e-6


def test_inert_check_needs_both_rules_and_the_coverage_check_first():
    tol = [0.5, 0.5]
    preds = [(np.full((3, 2), 0.1), np.ones((3, 2)))]
    ok = C.inert_check(preds, abs_tol=tol, coverage_passed=True)
    assert ok["pass"] and ok["frac_within"] == 1.0
    ghost = [(np.full((3, 2), 3.0), np.ones((3, 2)))]
    bad = C.inert_check(ghost, abs_tol=tol, coverage_passed=True)
    assert not bad["pass"] and bad["max_ratio"] == pytest.approx(3.0)
    inflated = [(np.full((3, 2), 3.0), np.full((3, 2), 100.0))]  # big sigma cannot rescue a ghost
    out = C.inert_check(inflated, abs_tol=tol, coverage_passed=True)
    assert out["sigma_rule"] and not out["abs_rule"] and not out["pass"]
    assert C.inert_check([(np.zeros((2, 2)), np.zeros((2, 2)))], tol, True)["n"] == 0
    with pytest.raises(C.OrderError):
        C.inert_check(preds, abs_tol=tol, coverage_passed=False)


def test_verify_inert_behavioural_and_mathematical():
    base = {"a": [1.0, 2.0], "b": [0.0, 0.0]}
    near = {"a": [1.0001, 2.0], "b": [0.0, 0.0]}
    assert C.verify_inert(base, near, tol={"a": 0.001, "b": 0.001})["pass"]
    assert not C.verify_inert(base, near, tol={"a": 1e-6, "b": 1e-6})["pass"]
    assert not C.verify_inert(base, near, tol={}, mathematical=True)["pass"]
    assert C.verify_inert(base, dict(base), tol={}, mathematical=True)["pass"]


def test_inert_spec_names_exact_tensors():
    spec = C.INERT_SPEC
    assert spec["dead_rows"]["tensor"] == "model.language_model.embed_tokens.weight"
    assert spec["dead_rows"]["rows_from"] == 248077 and spec["dead_rows"]["rows_to"] == 248320
    assert spec["compensated_pairs"]["mlp_layers"] == (12, 13, 14, 15) and spec["abs_tol_se"] > 0
    assert "k_norm_cancelled" in spec["excluded"]


def test_compensated_pairs_are_bitwise_inert_for_a_gated_mlp_in_bf16():
    torch.manual_seed(0)
    rm = A.RoleMap(2, 16, 32, 10, ("full_attention", "linear_attention"), tied=False)
    (up, down), (v, _o) = C.compensated_pairs(rm, mlp_layers=(0,), attention_layers=(0,))
    assert (
        up.param("factor") == 2.0
        and down.param("factor") == 0.5
        and v.tensor.endswith("v_proj.weight")
    )
    with pytest.raises(C.NotInertError):
        C.compensated_pairs(rm, attention_layers=(1,))
    x = torch.randn(7, 16).to(torch.bfloat16)
    Wg, Wu, Wd = (torch.randn(*s).to(torch.bfloat16) / 4 for s in ((32, 16), (32, 16), (16, 32)))
    Wu2, Wd2 = A.realise(Wu, up.intended_delta(Wu)), A.realise(Wd, down.intended_delta(Wd))
    mlp = lambda g, u, d: (torch.nn.functional.silu(x @ g.T) * (x @ u.T)) @ d.T
    assert torch.equal(mlp(Wg, Wu, Wd), mlp(Wg, Wu2, Wd2))


def test_dead_vocab_rows_touch_only_never_produced_rows_and_refuse_tied_embeddings():
    rm = A.RoleMap(4, 8, 16, 50, ("full_attention",) * 4, tied=False)
    used = set(range(40))
    act = C.dead_vocab_rows("model.embed_tokens.weight", rm, tokenizer_len=40, n_rows=6, seed=1)
    rows = set(act.param("rows"))
    assert len(rows) == 6 and min(rows) >= 40 and max(rows) < 50
    with pytest.raises(C.NotInertError):
        C.dead_vocab_rows("e", rm, tokenizer_len=46, n_rows=6, seed=1)
    w = torch.randn(50, 8)
    d = act.intended_delta(w)
    assert (d[sorted(used)] == 0).all() and (d[sorted(rows)] != 0).any()
    tied = A.RoleMap(4, 8, 16, 50, ("full_attention",) * 4, tied=True)
    with pytest.raises(C.NotInertError):
        C.dead_vocab_rows("model.embed_tokens.weight", tied, tokenizer_len=40, n_rows=6, seed=1)


def test_dead_rows_are_exactly_inert_for_an_embedding_lookup():
    rm = A.RoleMap(2, 8, 16, 30, ("full_attention",) * 2, tied=False)
    ids = torch.tensor([[1, 4, 5, 1, 9]])
    act = C.dead_vocab_rows("e", rm, tokenizer_len=10, n_rows=10, seed=2)
    W = torch.randn(30, 8).to(torch.bfloat16)
    W2 = A.realise(W, act.intended_delta(W))
    assert torch.equal(
        torch.nn.functional.embedding(ids, W), torch.nn.functional.embedding(ids, W2)
    )


def test_post_norm_cancelled_scale_is_k_proj_of_full_attention_only():
    rm = A.RoleMap(4, 8, 16, 50, ("linear_attention", "full_attention") * 2, tied=False)
    acts = C.post_norm_cancelled_scales(rm, layers=(0, 1, 2, 3))
    assert [a.tensor for a in acts] == [rm.name(1, "k"), rm.name(3, "k")]
    assert all(a.family == "scale" and a.param("factor") == 2.0 for a in acts)
    with pytest.raises(C.NotInertError):
        C.post_norm_cancelled_scales(rm, layers=(0,))


def test_power_of_two_k_scale_is_cancelled_by_rms_norm_in_bf16():
    torch.manual_seed(0)
    x = torch.randn(5, 16)
    Wk = (torch.randn(8, 16) / 4).to(torch.bfloat16)
    rm = A.RoleMap(2, 16, 32, 10, ("full_attention",) * 2, tied=False)
    act = C.post_norm_cancelled_scales(rm, layers=(0,))[0]
    Wk2 = A.realise(Wk, act.intended_delta(Wk))
    assert torch.equal(Wk2.float(), 2 * Wk.float())  # exact in bf16
    rms = lambda k: k / torch.sqrt((k**2).mean(-1, keepdim=True) + 1e-6)
    k1, k2 = rms(x @ Wk.float().T), rms(x @ Wk2.float().T)
    assert (k1 - k2).abs().max() < 1e-5


# ------------------------------------------------------------------ trust region


def test_trust_region_strata_split_at_point_one():
    rows = [{"dnll": v, "i": i} for i, v in enumerate([0.0, 0.05, 0.1, 0.1001, 0.8])]
    s = C.trust_region_strata(rows)
    assert [r["i"] for r in s["inside"]] == [0, 1, 2] and [r["i"] for r in s["outside"]] == [3, 4]
    out = C.by_region(rows, lambda rs: len(rs))
    assert out == {"inside": 3, "outside": 2, "all": 5}
    assert C.assign_stratum(0.1) == "inside" and C.assign_stratum(0.127) == "outside"


def test_stratum_report_marks_sparse_strata_descriptive():
    rows = [{"dnll": 0.01 * (i % 3), "component": i} for i in range(12)]
    rows += [{"dnll": 0.5, "component": 100 + i} for i in range(3)]
    rep = C.stratum_report(rows)
    assert rep["inside"]["inference"] == "registered" and rep["inside"]["n_components"] == 12
    assert rep["outside"]["inference"].startswith("descriptive")
    assert rep["nondestructive"]["n_rows"] == 8


# ------------------------------------------------------------------ direct best-of-N


def _bon_world(n=64, seed=0):
    rng = np.random.default_rng(seed)
    gains = rng.normal(0, 1, n)
    wiki = rng.normal(0, 0.005, n)
    return gains, wiki


def test_two_stage_best_of_n_respects_the_equal_budget_and_charges_screens():
    gains, wiki = _bon_world()
    led = BudgetLedger(32, ["bon", "wm"])
    screen = lambda c: {"gain": gains[c] + 0.3, "wikitext_dnll": wiki[c]}
    full = lambda c: {"gain": gains[c], "se": 0.1, "breaches": []}
    out = C.two_stage_best_of_n(
        list(range(64)), screen, full, led, "bon", screen_items=4, full_items=128, n_full=16
    )
    rep = led.report()
    assert rep["units"]["bon"] <= 32 + 1e-9 and rep["units"]["wm"] == 0
    assert out["n_screened"] == 64 and out["n_full"] <= 16
    assert out["winner"] in out["full_evaluated"]
    best = max(out["full_evaluated"], key=lambda c: gains[c] - 1.96 * 0.1)
    assert out["winner"] == best
    kinds = {e["kind"] for e in led.entries("bon")}
    assert kinds == {"screen", "evaluated"}


def test_two_stage_screen_drops_negative_gain_and_wikitext_degradation():
    led = BudgetLedger(32, ["bon"])
    screen = lambda c: {"gain": [1.0, -0.1, 2.0, 0.5][c], "wikitext_dnll": [0.0, 0.0, 0.05, 0.0][c]}
    full = lambda c: {"gain": 1.0, "se": 0.01, "breaches": []}
    out = C.two_stage_best_of_n(
        [0, 1, 2, 3], screen, full, led, "bon", screen_items=4, full_items=128, n_full=4
    )
    assert sorted(out["full_evaluated"]) == [0, 3]


def test_two_stage_returns_base_when_no_candidate_qualifies():
    led = BudgetLedger(32, ["bon"])
    screen = lambda c: {"gain": 1.0, "wikitext_dnll": 0.0}
    full = lambda c: {"gain": 1.0, "se": 0.01, "breaches": ["wikitext_nll"]}
    out = C.two_stage_best_of_n(
        [0, 1], screen, full, led, "bon", screen_items=4, full_items=128, n_full=2
    )
    assert out["winner"] is None


def test_ledger_units_overdraw_raises():
    led = BudgetLedger(2, ["a"])
    led.charge("a", "x", 1.0)
    led.charge("a", "y", 1.0, kind="screen", units=0.75)
    with pytest.raises(BudgetExceeded):
        led.charge("a", "z", 1.0)
    assert led.remaining("a") == pytest.approx(0.25)
    assert led.report()["used"] == {"a": 2}


# ------------------------------------------------------------------ unamortised accounting


def test_cost_report_covers_every_category_amortised_and_unamortised():
    shared = {"bank_gpu_s": 6 * 3600.0, "grounding_gpu_s": 1.5 * 3600, "feature_gpu_s": 0.5 * 3600}
    arms = {
        "wm": {"gain": 0.4, "eval_gpu_s": 3600.0, "planner_cpu_s": 7200.0, "uses_bank": True},
        "statnn": {"gain": 0.2, "eval_gpu_s": 3600.0, "uses_bank": True},
        "bon": {"gain": 0.3, "eval_gpu_s": 3600.0, "uses_bank": False},
        "lora": {"gain": 0.5, "eval_gpu_s": 1800.0, "train_gpu_s": 1800.0, "uses_bank": False},
        "reader": {"gain": 0.1, "eval_gpu_s": 3600.0, "llm_tokens": 12345},
    }
    rep = C.cost_report(arms, shared, n_episodes=4)
    assert rep["wm"]["unamortised_gpu_h"] == pytest.approx(9.0)
    assert rep["wm"]["amortised_gpu_h"] == pytest.approx(3.0)
    assert rep["wm"]["cpu_h"] == pytest.approx(2.0)
    assert rep["bon"]["unamortised_gpu_h"] == pytest.approx(1.0)
    assert rep["wm"]["gain_per_gpu_h_unamortised"] == pytest.approx(0.4 / 9.0)
    assert rep["lora"]["unamortised_gpu_h"] == pytest.approx(1.0)
    assert rep["reader"]["llm_tokens"] == 12345
    assert rep["_meta"]["gated"] is False and "generation_gpu_s" in rep["_meta"]["categories"]
    with pytest.raises(KeyError):
        C.cost_report({"x": {"gain": 0, "mystery_s": 1.0}}, shared, 1)
