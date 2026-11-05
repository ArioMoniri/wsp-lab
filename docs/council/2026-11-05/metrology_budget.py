#!/usr/bin/env python3
"""Council R1 (metrology): signal-to-noise budget, effect sizes and cost tables.

Reproduces every table in round1-metrology.md from canonical artefacts (read-only):
  data/zoo/qwen3.5-0.8b/<id>/{eval.json,records.jsonl,edit.json}     389 evaluated entries (300 D3 subset + 89 chain predecessors)
  data/evals/qwen3.5-0.8b/base/devices/<dev>/<run>/records.jsonl     device-matched base references
  data/runs/p2-batch/20261105-050543-aad6c310/<id>/repeats/seed1/    seed-1 panel (10 controls + base, device H0)
  data/hf/hub/datasets--TIGER-Lab--MMLU-Pro, datasets--Rowan--hellaswag  gold labels for the likelihood items

Run:  . scripts/env.sh && python docs/council/2026-11-05/metrology_budget.py > /tmp/metrology_tables.txt
Notes: no GPU, no model forward passes. Targets are the fast-tier fields already stored.
"""
import glob
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from scipy import stats

ROOT = Path(__file__).resolve().parents[3]  # worktree root; data/ is a symlink to canonical data
ZOO = ROOT / "data/zoo/qwen3.5-0.8b"
BASES = ROOT / "data/evals/qwen3.5-0.8b/base/devices"
PANEL = ROOT / "data/runs/p2-batch/20261105-050543-aad6c310"
HF = ROOT / "data/hf/hub"
BASE_RUN = {"H0": "20261104-233637-7c43da7c", "H1": "20261104-112635-8cee6d4a", "H2": "20261104-112739-7cbc5bbc"}
CONTROLS = ["0a841d4082532e8b", "1caef8eb4c640c3c", "3db97b535c996265", "3ddc3b4bf47ed994", "439dbfa62a14d55f",
            "7b8247d8d7ab7db9", "af6d608f8651fb46", "b6fe7f59016cdda2", "db5c3a7bfc19178c", "dd90d99e2ddc0e77"]
BINARY = ["gsm8k", "math500", "ifeval", "refusal_rate", "over_refusal", "mmlu_pro", "hellaswag"]
SAMPLED = ["gsm8k", "math500", "ifeval", "refusal_rate", "over_refusal"]  # seed-dependent; deterministic otherwise
TARGETS = BINARY + ["mmlu_pro_lp", "mmlu_pro_margin", "hellaswag_lp", "hellaswag_margin", "kl_to_base", "wikitext_ppl"]


def jl(path):
    with open(path) as f:
        return [json.loads(x) for x in f]


def index(recs):
    out = {}
    for r in recs:
        out.setdefault(r["benchmark"], {})[str(r["item_id"])] = r
    return out


# ---------- gold labels (validated: argmax(scores)==gold reproduces stored `correct` for all audited items)
mmlu_gold = pq.read_table(glob.glob(str(HF / "datasets--TIGER-Lab--MMLU-Pro/snapshots/*/data/test-*.parquet"))[0]).to_pandas()["answer_index"].to_numpy()
hs_gold = pq.read_table(glob.glob(str(HF / "datasets--Rowan--hellaswag/snapshots/*/data/validation-*.parquet"))[0]).to_pandas()["label"].astype(int).to_numpy()


def gold_lp_margin(scores, g):
    s = np.asarray(scores, float)
    lse = np.log(np.exp(s - s.max()).sum()) + s.max()
    return s[g] - lse, s[g] - np.delete(s, g).max()


# gold-label validation: argmax of stored choice scores must reproduce the stored `correct` flag
_chk = {"mmlu_pro": [0, 0], "hellaswag": [0, 0]}
for _eid in ["0a841d4082532e8b", "92e43fc1875443da"]:
    for _r in jl(ZOO / _eid / "records.jsonl"):
        if _r["benchmark"] in _chk:
            _gold = mmlu_gold if _r["benchmark"] == "mmlu_pro" else hs_gold
            _pred = int(np.argmax(_r["extra"]["scores"]))
            _chk[_r["benchmark"]][0] += int(int(_pred == int(_gold[int(_r["item_id"])])) == int(_r["correct"]))
            _chk[_r["benchmark"]][1] += 1
print("gold-label check (argmax agrees with stored correct):", {k: f"{v[0]}/{v[1]}" for k, v in _chk.items()})


# ---------- entries and matched bases
rows = []
for f in sorted(glob.glob(str(ZOO / "*/eval.json"))):
    d = json.load(open(f))
    eid = os.path.basename(os.path.dirname(f))
    edit = json.load(open(os.path.join(os.path.dirname(f), "edit.json")))
    rows.append(dict(id=eid, fam=d["family"], dev=d["eval_device_id"], base_run=d.get("base_run_id") or BASE_RUN[d["eval_device_id"]],
                     subset300=d.get("base_run_id") is None, dnorm=edit.get("delta_fro_norm"), eval=d))
E = pd.DataFrame(rows)
print(f"entries: {len(E)}  D3 subset (base_run_id None): {int(E.subset300.sum())}")
print(E.groupby(["fam", "dev"]).size().to_string())

base_cache = {}


def base_idx(run, dev):
    key = (run, dev)
    if key not in base_cache:
        base_cache[key] = index(jl(BASES / dev / run / "records.jsonl"))
    return base_cache[key]


def item_table(recs, rec_index):
    """per-entry paired item differences for every target, aligned to base items."""
    out = {}
    for bm in BINARY + ["kl_to_base", "wikitext_ppl"]:
        if bm not in recs or bm not in rec_index:
            continue
        common = [k for k in recs[bm] if k in rec_index[bm]]
        e = np.array([recs[bm][k]["correct"] for k in common], float)
        b = np.array([rec_index[bm][k]["correct"] for k in common], float)
        if bm == "wikitext_ppl":
            e, b = np.log(e), np.log(b)
        out[bm] = (e - b, e, b)
    return out


# ---------- item-level Δ for every entry, with an aggregate consistency check
X = []
mismatch = 0
for _, z in E.iterrows():
    recs = index(jl(ZOO / z.id / "records.jsonl"))
    bidx = base_idx(z.base_run, z.dev)
    items = item_table(recs, bidx)
    for bm in BINARY + ["kl_to_base", "wikitext_ppl"]:
        if bm not in items:
            continue
        d, e, b = items[bm]
        if bm in BINARY and abs(e.mean() - z.eval[bm]) > 1e-6:  # item means must reproduce eval.json aggregates
            mismatch += 1
        X.append(dict(id=z.id, fam=z.fam, dev=z.dev, target=bm, dy=d.mean(), se_item=d.std(ddof=1) / np.sqrt(len(d)), n=len(d), sd_item=d.std(ddof=1)))
    for bm, gold, nm in [("mmlu_pro", mmlu_gold, "mmlu"), ("hellaswag", hs_gold, "hs")]:
        if bm not in recs or bm not in bidx:
            continue
        ks = [k for k in recs[bm] if k in bidx[bm]]
        lpe, lpb, mge, mgb = [], [], [], []
        for k in ks:
            g = int(gold[int(k)])
            a1, a2 = gold_lp_margin(recs[bm][k]["extra"]["scores"], g)
            c1, c2 = gold_lp_margin(bidx[bm][k]["extra"]["scores"], g)
            lpe.append(a1); mge.append(a2); lpb.append(c1); mgb.append(c2)
        for tg, ve, vb in [(bm + "_lp", lpe, lpb), (bm + "_margin", mge, mgb)]:
            dd = np.asarray(ve) - np.asarray(vb)
            X.append(dict(id=z.id, fam=z.fam, dev=z.dev, target=tg, dy=dd.mean(), se_item=dd.std(ddof=1) / np.sqrt(len(dd)), n=len(dd), sd_item=dd.std(ddof=1)))
X = pd.DataFrame(X)
print(f"\naggregate check: item means vs eval.json aggregates differ in {mismatch} cells (expect 0)")

# ---------- seed panel: 10 controls, seed 0 (zoo, device H0) vs seed 1 (panel), each against its own seed base
b0 = json.load(open(BASES / "H0" / BASE_RUN["H0"] / "eval.json"))
b1 = json.load(open(PANEL / "base/repeats/seed1/eval.json"))
Z = E.set_index("id")
pan = []
for cid in CONTROLS:
    e1 = json.load(open(PANEL / cid / "repeats/seed1/eval.json"))
    r = dict(id=cid, fam=Z.loc[cid, "fam"])
    e0 = Z.loc[cid, "eval"]
    for m in SAMPLED + ["mmlu_pro", "hellaswag", "kl_to_base", "wikitext_ppl"]:
        f = (lambda v: np.log(v)) if m == "wikitext_ppl" else (lambda v: v)
        r[m] = f(e1[m]) - f(b1[m]) - (f(e0[m]) - f(b0[m]))  # Δ(seed1) - Δ(seed0)
    pan.append(r)
P = pd.DataFrame(pan)
seed_sd = {}
print("\nseed-panel noise on Δy (n=10 controls, seed1-seed0 difference / sqrt 2; 95% CI from chi-square, df=9):")
for m in SAMPLED + ["mmlu_pro", "hellaswag", "kl_to_base", "wikitext_ppl"]:
    s = P[m].std(ddof=1) / np.sqrt(2)
    seed_sd[m] = s
    lo = s * np.sqrt(9 / stats.chi2.ppf(0.975, 9))
    hi = s * np.sqrt(9 / stats.chi2.ppf(0.025, 9))
    print(f"  {m:14s} sd_seed={s:.4f}  CI=[{lo:.4f},{hi:.4f}]")

# seed-invariant targets: exact zero seed noise
for m in ["mmlu_pro", "hellaswag", "kl_to_base", "wikitext_ppl"]:
    seed_sd.setdefault(m, 0.0)
for m in ["mmlu_pro_lp", "mmlu_pro_margin", "hellaswag_lp", "hellaswag_margin"]:
    seed_sd[m] = 0.0

# ---------- decomposition
def budget(df_t, tg):
    """device-centred (per-device base offset removed) and family x device residual variances."""
    g = df_t.copy()
    g["c_dev"] = g.dy - g.groupby("dev").dy.transform("mean")
    g["c_fd"] = g.dy - g.groupby(["fam", "dev"]).dy.transform("mean")
    out = []
    for scope, col in [("pooled (device-centred)", "c_dev"), ("within family x device", "c_fd")]:
        tot = g[col].var(ddof=1)
        se2 = (g.se_item ** 2).mean()
        sv = seed_sd.get(tg, 0.0) ** 2
        sig = max(tot - se2, 0.0)
        out.append(dict(target=tg, scope=scope, n=len(g), sd_dy=np.sqrt(tot), sd_seed=np.sqrt(sv), sd_itemSE=np.sqrt(se2),
                        sd_signal_pop=np.sqrt(sig),
                        ceil_fixed=np.sqrt(max(1 - sv / tot, 0)) if tot > 0 else np.nan,
                        ceil_pop=np.sqrt(sig / tot) if tot > 0 else np.nan))
    return out


B = []
for tg in TARGETS:
    B += budget(X[X.target == tg], tg)
B = pd.DataFrame(B)
Bs = X[X.id.isin(E[E.subset300].id)]
Bsub = []
for tg in TARGETS:
    Bsub += budget(Bs[Bs.target == tg], tg)
Bsub = pd.DataFrame(Bsub)
pd.set_option("display.width", 250)
pd.set_option("display.max_columns", 30)
print("\nS/N budget, D3 subset (300 entries), per target:")
print(Bsub.round(4).to_string(index=False))
print("\nS/N budget, all 389 entries (for reference):")
print(B[B.scope.str.startswith("pooled")].round(4).to_string(index=False))

# ---------- per-family budget (device-centred within family)
F = []
for tg in TARGETS:
    for fam, g in Bs[Bs.target == tg].groupby("fam"):
        g = g.copy()
        g["c"] = g.dy - g.groupby("dev").dy.transform("mean") if g.dev.nunique() > 1 else g.dy - g.dy.mean()
        tot = g.c.var(ddof=1)
        se2 = (g.se_item ** 2).mean()
        sv = seed_sd.get(tg, 0.0) ** 2
        F.append(dict(target=tg, family=fam, n=len(g), sd_dy=np.sqrt(tot), sd_itemSE=np.sqrt(se2), sd_seed=np.sqrt(sv),
                      ceil_fixed=np.sqrt(max(1 - sv / tot, 0)) if tot > 0 else np.nan,
                      ceil_pop=np.sqrt(max(tot - se2, 0) / tot) if tot > 0 else np.nan))
F = pd.DataFrame(F)
print("\nper-family budget (device-centred within family, D3 subset):")
print(F.round(3).to_string(index=False))

# ---------- raw levels by device within family (base-draw check on sampled targets)
L = E[E.subset300].copy()
for m in ["gsm8k", "math500", "ifeval", "over_refusal"]:
    L[m] = L['eval'].map(lambda d, m=m: d[m])
lev = L.groupby(["fam", "dev"])[["gsm8k", "math500", "ifeval", "over_refusal"]].mean().round(3)
print("\nraw entry means by family x device (D3 subset):")
print(lev.to_string())
print("base runs (same seed 0):", {d: round(json.load(open(BASES / d / BASE_RUN[d] / 'eval.json'))['gsm8k'], 3) for d in BASE_RUN},
      "gsm8k; H0 seed-1 base:", round(b1["gsm8k"], 3))

# ---------- effect sizes per family (matched Δ, D3 subset)
print("\neffect sizes per family, matched-reference Δ on D3 subset (median / 90th pct |Δ|, fraction |Δ|>0.02 for binaries):")
ES = []
for fam, g in Bs.groupby("fam"):
    for tg in ["kl_to_base", "refusal_rate", "mmlu_pro", "mmlu_pro_lp", "hellaswag", "wikitext_ppl"]:
        h = g[g.target == tg]
        if h.empty:
            continue
        ES.append(dict(family=fam, target=tg, n=len(h), median_abs=h.dy.abs().median(), p90_abs=h.dy.abs().quantile(0.9),
                       frac_gt_002=(h.dy.abs() > 0.02).mean() if tg in BINARY else np.nan))
print(pd.DataFrame(ES).round(4).to_string(index=False))
print("\nmean matched Δ per family (D3 subset) and share of entries with |Δ|<=0.02 (binaries):")
pv = Bs.groupby(["target", "fam"]).dy.mean().unstack().round(4)
print(pv.to_string())
for tg in ["mmlu_pro", "hellaswag", "gsm8k", "math500", "ifeval", "over_refusal", "refusal_rate"]:
    h = Bs[Bs.target == tg]
    print(f"  {tg:13s} |Δ|<=0.02: {(h.dy.abs() <= 0.02 + 1e-9).mean():.3f}  (n={len(h)})  sd(Δ) {h.dy.std():.4f}")

# ---------- resolved entries (|Δ| > 2 total SE) and the edit magnitudes that clear it
kl = dict(zip(E.id, E['eval'].map(lambda d: d["kl_to_base"])))
dn = dict(zip(E.id, E.dnorm))
print("\nentries with |Δ| > 2 total SE (seed+item), and minimum KL / delta-norm among them:")
for tg in ["mmlu_pro", "mmlu_pro_lp", "hellaswag", "refusal_rate", "gsm8k", "over_refusal", "wikitext_ppl"]:
    g = Bs[Bs.target == tg].copy()
    g["thr"] = 2 * np.sqrt(seed_sd.get(tg, 0.0) ** 2 + g.se_item ** 2)
    r = g[g.dy.abs() > g.thr]
    print(f"  {tg:13s} {len(r):3d}/{len(g)}  min KL {min([kl[i] for i in r.id] or [np.nan]):.4f}  median KL {np.median([kl[i] for i in r.id]) if len(r) else np.nan:.4f}  "
          f"min |delta| {min([dn[i] for i in r.id] or [np.nan]):.3f}  median |delta| {np.median([dn[i] for i in r.id]) if len(r) else np.nan:.3f}")

# ---------- items needed for a paired SE target (per-item SD of paired differences)
print("\nitems needed for paired SE <= 0.02 / 0.01 (per-item SD of paired diffs, chain entries of D3 subset):")
for tg in ["mmlu_pro", "hellaswag", "gsm8k", "math500", "ifeval", "over_refusal", "mmlu_pro_lp", "hellaswag_lp"]:
    g = X[(X.target == tg) & X.id.isin(E[E.subset300].id)]
    sd_med = g.sd_item.median()
    sd_chain = g[g.fam == "chain"].sd_item.median()
    print(f"  {tg:13s} sd_item median {sd_med:.4f} chain {sd_chain:.4f}  n(SE .02) median {np.ceil((sd_med/0.02)**2):.0f} chain {np.ceil((sd_chain/0.02)**2):.0f}  "
          f"n(SE .01) chain {np.ceil((sd_chain/0.01)**2):.0f}")

# ---------- how much of the device-centred variance is family alone; within-family dose response on ||Delta||
print("\nfamily eta^2 (device-centred Δy), and within-family Spearman(delta_fro_norm, Δy):")
for tg in TARGETS:
    g = Bs[Bs.target == tg].copy()
    g["c"] = g.dy - g.groupby("dev").dy.transform("mean")
    fm = g.groupby("fam").c.transform("mean")
    ss_tot = ((g.c - g.c.mean()) ** 2).sum()
    eta2 = 1 - ((g.c - fm) ** 2).sum() / ss_tot if ss_tot > 0 else np.nan
    rho = {}
    for fam, h in g.groupby("fam"):
        h = h.assign(dn=h.id.map(dict(zip(E.id, E.dnorm))))
        rho[fam] = round(stats.spearmanr(h.dn, h.dy).statistic, 2) if h.dy.std() > 0 and h.dn.std() > 0 else None
    print(f"  {tg:15s} eta2_family={eta2:.3f}  rho(norm,Δy) within family: {rho}")

# ---------- non-null entries (KL > 0.01 is the non-trivial-edit line) and zoo composition
Sx = E[E.subset300]
print("\nD3 subset entries with KL>0.01 (non-null edits):", {f: f"{int((g['eval'].map(lambda d: d['kl_to_base']) > 0.01).sum())}/{len(g)}" for f, g in Sx.groupby("fam")})
print("zoo composition (edit.json, all dirs):", pd.Series([json.load(open(p)).get("family") for p in glob.glob(str(ZOO / "*/edit.json"))]).value_counts().to_dict())

# ---------- compute cost (from recorded token counts; decode steps = batch max, capped at 1024)
steps = []
for _, z in E.iterrows():
    recs = jl(ZOO / z.id / "records.jsonl")
    s = 0
    for bm in ["gsm8k", "math500", "ifeval", "refusal_rate", "over_refusal"]:
        rr = sorted([r for r in recs if r["benchmark"] == bm], key=lambda r: int(r["item_id"]))
        for i in range(0, len(rr), 32):
            s += max(r["n_tokens"] for r in rr[i:i + 32])
    s += 2 * 128  # KL base continuations are regenerated per entry (2 batches of <=32, 128 tokens)
    fw = sum(len(r["extra"]["scores"]) for r in recs if r["benchmark"] in ("mmlu_pro", "hellaswag"))
    steps.append(dict(id=z.id, dev=z.dev, decode_steps=s, choice_forwards=fw, minutes=z.eval["minutes"]))
S = pd.DataFrame(steps)
print("\ncost of current fast tier (records):")
print(S.groupby("dev")[["decode_steps", "choice_forwards", "minutes"]].mean().round(1).to_string())
ms = S[S.dev != "H2"].minutes.mean() * 60 / S[S.dev != "H2"].decode_steps.mean()
print(f"  implied wall time per decode step if generation dominates: {ms*1000:.0f} ms (H0/H1 mean; derived, not profiled)")

# ---------- power (Fisher z)
print("\npower, Fisher-z (two-sided alpha .05, 80% power):")
for n in [300, 1000, 246]:
    se = 1 / np.sqrt(n - 3)
    z = np.arctanh
    print(f"  n={n:4d}  SE(z)={se:.4f}  95% CI rho=.6: [{np.tanh(z(.6)-1.96*se):.3f},{np.tanh(z(.6)+1.96*se):.3f}]  "
          f"rho=.4: [{np.tanh(z(.4)-1.96*se):.3f},{np.tanh(z(.4)+1.96*se):.3f}]")
    print(f"         P(detect .6 vs 0)={stats.norm.cdf(z(.6)/se-1.96):.4f}  P(detect .4 vs 0)={stats.norm.cdf(z(.4)/se-1.96):.4f}  "
          f"P(0.6 vs 0.4, two independent samples)={stats.norm.cdf((z(.6)-z(.4))/(np.sqrt(2)/np.sqrt(n-3))-1.96):.4f}  "
          f"P(single sample truth .6, H0 .4)={stats.norm.cdf((z(.6)-z(.4))/se-1.96):.4f}")
    print(f"         smallest rho detectable vs 0 at 80% power: {np.tanh(2.8016*se):.3f}")
    for rel in [0.5, 0.75, 0.9]:
        print(f"         reliability {rel}: attenuated rho at true .6 = {.6*np.sqrt(rel):.3f}, at true .4 = {.4*np.sqrt(rel):.3f}")
# non-null subset (KL>0.01) and extrapolated non-null count for the full zoo (family rates from the D3 subset)
rate = {f: float((g["eval"].map(lambda d: d["kl_to_base"]) > 0.01).mean()) for f, g in Sx.groupby("fam")}
zoo_counts = pd.Series([json.load(open(p)).get("family") for p in glob.glob(str(ZOO / "*/edit.json"))]).value_counts()
n_full_nonnull = int(round(sum(zoo_counts.get(f, 0) * rate.get(f, 0) for f in rate)))
print(f"\nnon-null subset: n=94 (KL>0.01 in D3 subset); extrapolated full-zoo non-null count: ~{n_full_nonnull} of {int(zoo_counts.sum())}")
for n in [94, n_full_nonnull]:
    se = 1 / np.sqrt(n - 3)
    print(f"  n={n}: 95% CI rho=.6 [{np.tanh(z(.6)-1.96*se):.3f},{np.tanh(z(.6)+1.96*se):.3f}]; P(0.6 vs 0.4, independent) = "
          f"{stats.norm.cdf((z(.6)-z(.4))/(np.sqrt(2)/np.sqrt(n-3))-1.96):.3f}")

# ---------- Board A fast-tier screening rule applied to our panels (approximate: R_b ~ 1.96*SD(d) and max|d|;
# d = second-minus-first execution of the same entry; delta_b = .10 binaries, log 1.05 log-ppl, .01 nats KL; need R_b <= delta/2)
print("\nBoard A repeat criterion (R_b <= delta/2), seed panel d = Δ(seed1) - Δ(seed0), 10 controls:")
DELTA = {m: 0.10 for m in SAMPLED + ["mmlu_pro", "hellaswag"]}
DELTA.update({"kl_to_base": 0.01, "wikitext_ppl": float(np.log(1.05))})
for m in SAMPLED + ["mmlu_pro", "hellaswag", "kl_to_base", "wikitext_ppl"]:
    d = P[m].to_numpy()
    r_sd = 1.96 * d.std(ddof=1) if len(d) > 1 else np.nan
    r_max = np.max(np.abs(d))
    half_item = 1.96 * np.median(X[(X.target == m) & X.id.isin(E[E.subset300].id)].se_item) if m in X.target.values else np.nan
    print(f"  {m:14s} delta/2={DELTA[m]/2:.4f}  R_b(1.96 SD)={r_sd:.4f}  R_b(max|d|)={r_max:.4f}  median item CI half-width={half_item:.4f}  "
          f"qualifies(repeat)={'yes' if max(r_sd, 0) <= DELTA[m]/2 else 'no'}")

# ---------- cross-device check on aggregates (device copies carry eval.json only, no item records)
CTRL_X = []
for cid in CONTROLS:
    h0 = json.load(open(ZOO / cid / "eval.json"))
    for dev_dir in sorted(glob.glob(str(ZOO / cid / "devices/H[12]/*/eval.json"))):
        h = json.load(open(dev_dir))
        CTRL_X.append((cid, h["eval_device_id"], h0, h))
keys = SAMPLED + ["mmlu_pro", "hellaswag", "kl_to_base", "wikitext_ppl"]
print("\ncross-device aggregate differences, same entry: H0 (zoo) vs H1/H2 copies (n pairs shown per device):")
for dev in ["H1", "H2"]:
    pairs = [(c, a, b) for c, d, a, b in CTRL_X if d == dev]
    for m in keys:
        diffs = [((np.log(b[m]) - np.log(a[m])) if m == "wikitext_ppl" else (b[m] - a[m])) for c, a, b in pairs]
        print(f"  {dev} n={len(pairs)} {m:14s} mean diff {np.mean(diffs):+.4f}  max|diff| {np.max(np.abs(diffs)):.4f}  sd {np.std(diffs, ddof=1):.4f}")
