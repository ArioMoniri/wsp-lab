"""Round 2: measured pilot of the proposed primary estimand on EXISTING P4 rows (CPU, read-only).

Estimand (see round2-metrology.md section 4): residual r = y - family effect - b * log||Delta||_F,
with family effect and b fitted on training folds only. Model arm: StatNN on the pooled stats fitted to r.
Comparator arm: isotonic function of ||Delta||_F fitted to r (the norm comparator, monotone, on the residual).
theta = 1 - sum_c sum_i |r_i - model_i| / sum_c sum_i |r_i - comparator_i|, summed over ancestry clusters c.
Positive theta: stats features explain held-out residual error beyond family + norm.

Targets here are the existing rate endpoints (MMLU-Pro, HellaSwag predecessor-relative deltas), not TF-v1;
this is a pilot of the estimand's behaviour on data already used by P4, so it is exploratory.

Run: python docs/council/2026-11-05/round2_residual_estimand.py  (after . scripts/env.sh, from the worktree root)
"""

import json
import math
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr
from sklearn.isotonic import IsotonicRegression

from wsp.predictor.baselines import _make_model
from wsp.predictor.data import FeatureDataset

ROOT = Path(".")
ZOO = ROOT / "data/zoo/qwen3.5-0.8b"
RNG = np.random.default_rng(7)
Z = 1.959964  # two-sided 0.05

man = {}
for line in (ZOO / "manifest.jsonl").read_text().splitlines():
    r = json.loads(line)
    if "id" in r:
        man[r["id"]] = r


def root(i):
    cur, seen = i, {i}
    while man[cur]["parent_id"] is not None and man[cur]["parent_id"] not in seen:
        cur = man[cur]["parent_id"]
        seen.add(cur)
    return cur


ds = FeatureDataset(
    "data/features/qwen3.5-0.8b", "data/evals/qwen3.5-0.8b", targets=["mmlu_pro", "hellaswag"]
)
rows = ds.rows
ids = [r["id"] for r in rows]
clusters = np.array([root(i) for i in ids])
fam = np.array([r["family"] for r in rows])
X = np.stack([ds.entry(i)["stats"].numpy() for i in ids])
N = np.array([ds.metadata[i]["norms"]["fro_effective"] for i in ids], dtype=float)
logN = np.log(np.clip(N, 1e-12, None))
families = sorted(set(fam))
devs = np.array([r["eval_device_id"] for r in rows])
DEVICES = sorted(set(devs))
F1 = np.stack([(fam == f).astype(float) for f in families], axis=1)  # family one-hot
F2 = np.stack([(devs == d).astype(float) for d in DEVICES], axis=1)  # device one-hot (base-draw offset)
print("devices", DEVICES, {d: int((devs == d).sum()) for d in DEVICES})
folds = ds.folds("trajectory", 5)
print("rows", len(ids), "clusters", len(set(clusters)), "families", families)

out = {}
for t_name in ["mmlu_pro", "hellaswag"]:
    j = ds.names.index(t_name)
    y = ds.targets.numpy()[:, j].astype(float)
    model_pred = np.full(len(y), np.nan)
    comp_pred = np.full(len(y), np.nan)
    resid = np.full(len(y), np.nan)
    base_pred = np.full(len(y), np.nan)
    for tr, te in folds:
        tr = tr[np.isfinite(y[tr])]
        te = te[np.isfinite(y[te])]
        if len(tr) < 10 or len(te) == 0:
            continue
        # additive nuisance: family + device (base draw) + log-norm, fitted on training rows only
        D = np.concatenate([F1, F2, logN[:, None]], axis=1)
        coef, *_ = np.linalg.lstsq(D[tr], y[tr], rcond=None)  # family effects + log-norm slope
        r_tr = y[tr] - D[tr] @ coef
        base_pred[te] = D[te] @ coef
        resid[te] = y[te] - base_pred[te]
        est = _make_model("statnn", 0).fit(X[tr], r_tr)
        model_pred[te] = est.predict(X[te])
        iso = IsotonicRegression(increasing="auto", out_of_bounds="clip").fit(N[tr], r_tr)
        comp_pred[te] = iso.predict(N[te])
    ok = np.isfinite(resid) & np.isfinite(model_pred) & np.isfinite(comp_pred)
    r, m, c, cl = resid[ok], model_pred[ok], comp_pred[ok], clusters[ok]
    em, ec = np.abs(r - m), np.abs(r - c)
    theta = 1 - em.sum() / ec.sum()
    # cluster bootstrap of the ratio
    uniq = np.unique(cl)
    idx_by = {u: np.where(cl == u)[0] for u in uniq}
    boots = []
    for _ in range(4000):
        pick = RNG.choice(uniq, len(uniq))
        sel = np.concatenate([idx_by[u] for u in pick])
        boots.append(1 - em[sel].sum() / ec[sel].sum())
    lo, hi = np.percentile(boots, [2.5, 97.5])
    rho_m = spearmanr(r, m).statistic
    rho_c = spearmanr(r, c).statistic
    # zero residual prediction (= family + device + norm baseline only)
    e0 = np.abs(r)
    dv = devs[ok]
    out[t_name] = dict(
        n=int(ok.sum()),
        clusters=int(len(uniq)),
        theta=round(float(theta), 4),
        boot_ci=[round(float(lo), 3), round(float(hi), 3)],
        spearman_residual_statnn=round(float(rho_m), 3),
        spearman_residual_norm_comparator=round(float(rho_c), 3),
        mae_residual_zero_pred=round(float(e0.mean()), 4),
        mae_residual_statnn=round(float(em.mean()), 4),
        mae_residual_norm=round(float(ec.mean()), 4),
    )
    # theta restricted to the families that carry signal on the fast tier (non-null proxy; KL is not in these rows)
    nn_sel = np.isin(fam[ok], ["lora_sft", "chain", "task_vector"])
    em_nn, ec_nn, cl_nn = em[nn_sel], ec[nn_sel], cl[nn_sel]
    u_nn = np.unique(cl_nn)
    idx_nn = {u: np.where(cl_nn == u)[0] for u in u_nn}
    bn = []
    for _ in range(4000):
        pick = RNG.choice(u_nn, len(u_nn))
        sel = np.concatenate([idx_nn[u] for u in pick])
        bn.append(1 - em_nn[sel].sum() / ec_nn[sel].sum())
    out[t_name]["nonnull_proxy_theta"] = dict(
        rows=int(nn_sel.sum()),
        clusters=int(len(u_nn)),
        theta=round(float(1 - em_nn.sum() / ec_nn.sum()), 4),
        boot_ci=[round(float(x), 3) for x in np.percentile(bn, [2.5, 97.5])],
    )
    # cluster-level SD implied by the bootstrap SE, for power planning
    lo_b, hi_b = np.percentile(boots, [2.5, 97.5])
    se_theta = (hi_b - lo_b) / (2 * Z)
    out[t_name]["implied_cluster_sd_relative"] = round(float(se_theta * math.sqrt(len(uniq))), 3)
    # within-family residual Spearman for the model arm
    per_fam = {}
    for f in families:
        sel = fam[ok] == f
        if sel.sum() >= 30 and np.std(r[sel]) > 0 and np.std(m[sel]) > 0:
            per_fam[f] = round(float(spearmanr(r[sel], m[sel]).statistic), 3)
    out[t_name]["within_family_spearman_statnn"] = per_fam
    per_dev = {}
    for d in DEVICES:
        sel = dv == d
        if sel.sum() >= 30 and np.std(r[sel]) > 0 and np.std(m[sel]) > 0:
            per_dev[str(d)] = round(float(spearmanr(r[sel], m[sel]).statistic), 3)
    out[t_name]["within_device_spearman_statnn"] = per_dev

print("Residual nuisance: family + device + log||Delta||_F (additive, train folds only).")
print(json.dumps(out, indent=1))
# power planning from the bootstrap-implied cluster SD: clusters needed for 80% power, two-sided 0.05
print("\nclusters needed for 80% power (two-sided 0.05) to detect theta* with the implied cluster SD:")
for t_name, v in out.items():
    for label, sd in [("all families", v["implied_cluster_sd_relative"]),
                      ("non-null proxy (lora, chain, task)", None)]:
        if sd is None:
            ci = v["nonnull_proxy_theta"]["boot_ci"]
            sd = ((ci[1] - ci[0]) / (2 * Z)) * math.sqrt(v["nonnull_proxy_theta"]["clusters"])
        req = {th: int(math.ceil(((Z + 0.841621) * sd / th) ** 2)) for th in [0.10, 0.15, 0.20]}
        print(f"  {t_name:10s} {label:36s} SD={sd:.3f}  theta*=0.10: {req[0.10]}  0.15: {req[0.15]}  0.20: {req[0.20]}")


# ---------------------------------------------------------------- minimum detectable theta by panel size
print("\nminimum detectable theta (80% power, two-sided 0.05) = 2.8016 * cluster SD / sqrt(clusters)")
for t_name, v in out.items():
    sd_all = v["implied_cluster_sd_relative"]
    ci = v["nonnull_proxy_theta"]["boot_ci"]
    sd_nn = ((ci[1] - ci[0]) / (2 * Z)) * math.sqrt(v["nonnull_proxy_theta"]["clusters"])
    for label, C, sd in [
        ("all families, existing (278 clusters)", 278, sd_all),
        ("non-null proxy, existing (128 clusters)", v["nonnull_proxy_theta"]["clusters"], sd_nn),
        ("full zoo non-null, projected (~492 clusters)", 492, sd_nn),
        ("full zoo, all families (646 clusters)", 646, sd_all),
        ("full zoo + 700 new independent edits (1,346)", 1346, sd_all),
    ]:
        print(f"  {t_name:10s} {label:44s} MDE={(Z + 0.841621) * sd / math.sqrt(C):.3f}")
