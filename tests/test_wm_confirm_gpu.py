"""GPU smoke tests for the WM-E confirmatory engineering (Qwen3.5-0.8B, wme-p2 CONSTRUCTION items only; tiny).

* compensated power-of-two pairs are bitwise inert on the real model; the dead-vocabulary-row generator refuses the
  tied 0.8B embedding; the k_norm-cancelled k_proj x2 edit is measurably not inert (why it is excluded);
* the H4 preflight runs each objective's worst batch first and reads device memory under the 27 GB cap.
"""

from pathlib import Path

import pytest

pytestmark = pytest.mark.gpu
ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def m08():
    import torch

    if not torch.cuda.is_available():
        pytest.skip("no CUDA")
    from huggingface_hub import constants

    from wsp.wm import hf

    snaps = Path(constants.HF_HUB_CACHE) / "models--Qwen--Qwen3.5-0.8B" / "snapshots"
    if not snaps.exists():
        pytest.skip("0.8B not cached")
    hf.TARGETS["Qwen/Qwen3.5-0.8B"] = min(snaps.iterdir()).name
    model, tok, cfg, base_hash, _ = hf.load_target("Qwen/Qwen3.5-0.8B", "cuda")
    yield model, tok, cfg, base_hash
    del model
    torch.cuda.empty_cache()


def _items(n=4, objs=("wikitext_nll", "sycophancy_real", "gsm8k_gold_lp")):
    from wsp.wm import panel2 as P2

    its = P2.read_open(ROOT / "data/wme/panels/wme-p2", "construction")
    return {o: sorted(its[o], key=lambda r: r["id"])[:n] for o in objs}


def _rolemap(model, cfg):
    from wsp.wm import actions as A

    names = [n for n, _ in model.named_parameters()]
    prefix = (
        "model.language_model.layers."
        if any(".language_model." in n for n in names)
        else "model.layers."
    )
    return A.RoleMap.from_config(cfg, prefix=prefix)


def test_compensated_pairs_are_bitwise_inert_on_08b_and_dead_rows_refuse_tied(m08):
    from wsp.wm import actions as A
    from wsp.wm import controls as C
    from wsp.wm import panel as P
    from wsp.wm import panel2 as P2

    model, tok, cfg, base_hash = m08
    rm = _rolemap(model, cfg)
    assert rm.tied
    with pytest.raises(C.NotInertError):
        C.dead_vocab_rows("model.language_model.embed_tokens.weight", rm, len(tok), 4, 0)
    items = _items()
    ev = lambda: P2.evaluate(P.LMAdapter(model), tok, items, P.PanelConfig())
    vals = lambda r: {o: [x["value"] for x in v] for o, v in r.items()}
    base = vals(ev())
    full = [L for L, t in enumerate(rm.layer_types) if t == "full_attention"]
    for bundle in C.compensated_pairs(rm, mlp_layers=(12, 15), attention_layers=(full[-1],)):
        sess = A.EditSession(dict(model.named_parameters()), base_hash=base_hash, rolemap=rm)
        sess.apply_bundle(bundle)
        assert sess.theta_hash() != base_hash
        edited = vals(ev())
        sess.restore()
        assert sess.theta_hash() == base_hash
        assert C.verify_inert(base, edited, tol={}, mathematical=True)["pass"]
    assert vals(ev()) == base  # bitwise restore


def test_k_norm_cancelled_scale_is_measurably_not_inert_on_08b(m08):
    """Records why k_proj x2 is excluded: in bf16 it moves teacher-forced values beyond a 0.01-nat tolerance."""
    from wsp.wm import actions as A
    from wsp.wm import controls as C
    from wsp.wm import panel as P
    from wsp.wm import panel2 as P2

    model, tok, cfg, base_hash = m08
    rm = _rolemap(model, cfg)
    items = _items()
    ev = lambda: P2.evaluate(P.LMAdapter(model), tok, items, P.PanelConfig())
    vals = lambda r: {o: [x["value"] for x in v] for o, v in r.items()}
    base = vals(ev())
    sess = A.EditSession(dict(model.named_parameters()), base_hash=base_hash, rolemap=rm)
    sess.apply_bundle(C.post_norm_cancelled_scales(rm, [3]))
    edited = vals(ev())
    sess.restore()
    v = C.verify_inert(base, edited, tol={o: 0.01 for o in items})
    assert not v["pass"]
    assert v["objectives"]["wikitext_nll"]["max_abs_change"] < 0.05  # small but nonzero


def test_preflight_runs_worst_batches_under_the_cap(m08):
    import torch

    from wsp.wm import harness as H
    from wsp.wm import panel as P
    from wsp.wm import panel2 as P2

    model, tok, _cfg, _ = m08
    items = _items(n=6)
    cfg = P.PanelConfig()
    rows = {o: [r for it in its for r in P2.rows_for(o, it, tok, cfg)] for o, its in items.items()}
    lengths = {o: [len(r.ids) for r in rs] for o, rs in rows.items()}
    plan = H.worst_batches(lengths, cfg.max_batch_tokens)
    pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id

    def run(p):
        rs = [rows[p["objective"]][i] for i in p["rows"]]
        P._score_rows(P.LMAdapter(model), rs, cfg, pad)
        torch.cuda.synchronize()

    out = H.preflight(plan, run)
    assert out["pass"] and out["peak_bytes"] < H.DEVICE_CAP_BYTES and len(out["rows"]) == len(items)
