"""GPU smoke for track B (Qwen3.5-0.8B, wme-p2 CONSTRUCTION items only; tiny; engineering, no outcome reported).

* the vocabulary-row field refuses the tied 0.8B embedding (the 9B target is untied);
* the LoRA-SFT arm trains a rank-8 adapter on construction items, snapshots 2 checkpoints, evaluates each as one
  ``checkpoint`` bundle on the shared ledger, charges training seconds, and restores the base bitwise.
"""

import json
from pathlib import Path

import pytest

pytestmark = pytest.mark.gpu
ROOT = Path(__file__).resolve().parents[1]
P2DIR = ROOT / "data/wme/panels/wme-p2"
BASE = "b" * 64


@pytest.fixture(scope="module")
def m08():
    import torch

    if not torch.cuda.is_available():
        pytest.skip("no CUDA")
    from huggingface_hub import constants

    from wsp.wm import hf

    snaps = Path(constants.HF_HUB_CACHE) / "models--Qwen--Qwen3.5-0.8B" / "snapshots"
    if not snaps.exists() or not (P2DIR / "manifest.json").exists():
        pytest.skip("0.8B or wme-p2 not available")
    hf.TARGETS["Qwen/Qwen3.5-0.8B"] = min(snaps.iterdir()).name
    model, tok, cfg, base_hash, _ = hf.load_target("Qwen/Qwen3.5-0.8B", "cuda")
    yield model, tok, cfg, base_hash
    del model
    torch.cuda.empty_cache()


def _rolemap(model, cfg):
    from wsp.wm import actions as A

    names = [n for n, _ in model.named_parameters()]
    prefix = (
        "model.language_model.layers."
        if any(".language_model." in n for n in names)
        else "model.layers."
    )
    return A.RoleMap.from_config(cfg, prefix=prefix)


def test_vocab_field_refuses_tied_08b_embedding(m08, tmp_path):
    from wsp.wm import vocabfield as VF

    model, tok, _cfg, base_hash = m08
    vocab = model.get_output_embeddings().out_features
    decl = VF.declare_rows({"lm_head": {"r": [10, 11]}, "embed": {"r": [10, 11]}}, vocab)
    with pytest.raises(VF.TiedEmbeddingError):
        VF.collect_vocab_rows(model, tok, [], {}, decl, "cuda", tmp_path, theta_hash=base_hash)


def test_lora_sft_arm_checkpoints_on_08b_construction_items(m08):
    import torch

    from wsp.wm import access as AC
    from wsp.wm import actions as A
    from wsp.wm import lora as LO
    from wsp.wm import panel as P
    from wsp.wm import panel2 as P2
    from wsp.wm.arms import direct as D
    from wsp.wm.arms import spec as S
    from wsp.wm.plan import BudgetLedger

    model, tok, cfg, base_hash = m08
    rm = _rolemap(model, cfg)
    man = json.loads((P2DIR / "manifest.json").read_text())
    obj = "gsm8k_gold_lp"
    cons = sorted(P2.read_open(P2DIR, "construction")[obj], key=lambda r: r["id"])[:4]
    shard = set(man["objectives"][obj]["construction"]["hashes"])
    forbidden = {
        h
        for s in ("dev", "validation")
        for h in man["objectives"][obj].get(s, {}).get("hashes", [])
    }
    rows = LO.sft_rows(obj, cons, tok, P.PanelConfig(), man)
    pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    targets = [rm.writers()[(L, r)] for L in (12, 13) for r in ("down", "o")]
    ad = LO.LoRA(model, targets, rank=8, alpha=16, seed=1)
    before = {n: p.detach().clone() for n, p in model.named_parameters() if n in targets}
    snaps = {}

    def train_fn(items, steps_at, seed):
        out = D.train_with_checkpoints(
            model, ad, rows, pad, steps_at, lr=1e-3, batch_rows=2, seed=seed, device="cuda"
        )
        snaps.update(out["checkpoints"])
        return {
            "checkpoints": {s: ("lora_sft", s) for s in steps_at},
            "gpu_seconds": out["seconds"],
        }

    ev_items = {obj: cons}
    score = lambda: [
        x["value"] for x in P2.evaluate(P.LMAdapter(model), tok, ev_items, P.PanelConfig())[obj]
    ]
    base_vals = score()

    def evaluate(c):
        sess = A.EditSession(dict(model.named_parameters()), base_hash=base_hash, rolemap=rm)
        LO.apply_lean(sess, LO.lowrank_actions(snaps[c[1]], 1.0))
        th = sess.theta_hash()
        vals = score()
        sess.restore()
        assert sess.theta_hash() == base_hash and th != base_hash
        d = torch.tensor(vals) - torch.tensor(base_vals)
        return {
            "gain": float(d.mean()),
            "se": float(d.std() / len(d) ** 0.5),
            "guards": [],
            "rel_norm": 0.0,
            "seconds": 0.0,
        }

    led = BudgetLedger(2, ["lora_sft"])
    ctx = S.ArmContext(
        S.ARM_SPECS["lora_sft"],
        led,
        "smoke",
        evaluate,
        theta_of=repr,
        resources={
            "construction_e2": {"items": cons, "shard_hashes": shard, "forbidden_hashes": forbidden}
        },
        stores=S.StoreRegistry(),
        guard=AC.R0Guard(BASE),
    )
    out = D.run_lora_sft(ctx, train_fn, B=2, seed=1, total_steps=4)
    rep = led.verify_parity()
    assert rep["by_kind"]["lora_sft"] == {"checkpoint": 2.0} and out["steps_at"] == [2, 4]
    assert out["costs"]["train_gpu_s"] > 0
    assert sorted(snaps) == [2, 4]
    for n, p in model.named_parameters():
        if n in before:
            assert torch.equal(p, before[n]), n
    assert score() == base_vals  # bitwise restore
