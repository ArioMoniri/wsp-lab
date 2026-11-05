"""Round 2 metrology checks (CPU, read-only over canonical data/).

Run from the worktree root after `. scripts/env.sh`:
    python docs/council/2026-11-05/round2_metrology.py > docs/council/2026-11-05/round2_metrology_output.txt

Sections
  A  lineage and ancestry-cluster census of the zoo (manifest + edit.json + eval.json)
  B  non-null subset of the 300-entry D3 panel and its ancestry clusters
  C  measured fast-tier cost (eval.json `minutes`) by device
  D  stored-field audit: which per-item fields exist for the TF-v1 endpoints
  E  history (M2 vs M2-matched stateless) on length>=4 rows: effect, cluster SD, power, cost
  F  paired-MAE power for feature-addition effects (StatNN vs descriptor-only, vs delta-norm)
  G  Fisher-z power for rho 0.6 vs 0.4 with entries vs ancestry clusters as n

Every estimator is the ratio of cluster sums: theta = sum_c D_c / sum_c S_c, where S_c is the
stateless absolute error summed over cluster c's rows (seed-averaged) and D_c the excess over the
comparator. Power uses the cluster-level SD of D_c - theta* S_c (two-sided alpha 0.05, 80% power).
"""

import glob
import json
import math
import os
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

Z_POWER = 1.959964 + 0.841621  # two-sided alpha 0.05, power 0.80
ROOT = Path(".")
ZOO = ROOT / "data/zoo/qwen3.5-0.8b"
P4 = ROOT / "data/runs/p4"
RNG = np.random.default_rng(20261105)


def hdr(s):
    print("\n" + "=" * 8 + " " + s + " " + "=" * 8)


# ---------------------------------------------------------------- A lineage census
man = {}
for line in (ZOO / "manifest.jsonl").read_text().splitlines():
    r = json.loads(line)
    if "id" in r:
        man[r["id"]] = r
ed = {}
for f in glob.glob(str(ZOO / "*/edit.json")):
    e = json.load(open(f))
    ed[e.get("id", Path(f).parent.name)] = e
evald = {i for i in ed if (ZOO / i / "eval.json").exists()}


def lineage(i):
    out, cur, seen = [i], i, {i}
    while man[cur]["parent_id"] is not None and man[cur]["parent_id"] not in seen:
        cur = man[cur]["parent_id"]
        seen.add(cur)
        out.append(cur)
    return out[::-1]  # root .. entry


hdr("A lineage census (full zoo)")
print("entries", len(ed), "evaluated", len(evald))
print("distinct lineage roots, all entries:", len({lineage(i)[0] for i in ed}))
fam = Counter(e["family"] for e in ed.values())
print("family counts", dict(fam))
prod = {i for i in ed if ed[i].get("eligibility") == "production"}
print("production entries", len(prod), "distinct roots among production", len({lineage(i)[0] for i in prod}))
L = {i: len(lineage(i)) for i in ed}
for k in [2, 3, 4, 5]:
    ids = [i for i in ed if ed[i]["family"] == "chain" and L[i] >= k]
    full_prod_eval = [
        i
        for i in ids
        if all(a in prod for a in lineage(i)) and all(a in evald for a in lineage(i))
    ]
    print(
        f"chain lineage nodes>={k}: entries {len(ids)}, roots {len({lineage(i)[0] for i in ids})};"
        f" all-ancestors production+evaluated: entries {len(full_prod_eval)},"
        f" roots {len({lineage(i)[0] for i in full_prod_eval})}"
    )

# ---------------------------------------------------------------- B non-null D3 subset
hdr("B non-null D3 subset (300 entries without base_run_id; KL threshold 0.01)")
d3 = []
for i in sorted(evald):
    ev = json.load(open(ZOO / i / "eval.json"))
    if "base_run_id" not in ev:
        d3.append((i, ed[i]["family"], ev.get("kl_to_base"), ev.get("minutes"), ev.get("eval_device_id")))
d3 = pd.DataFrame(d3, columns=["id", "family", "kl", "minutes", "device"])
d3["nonnull"] = d3.kl > 0.01
d3["root"] = [lineage(i)[0] for i in d3.id]
print("D3 entries", len(d3), "ancestry clusters (all D3)", d3.root.nunique())
print(d3.groupby("family").agg(n=("id", "size"), nonnull=("nonnull", "sum")).to_string())
nn = d3[d3.nonnull]
print("non-null entries", len(nn), "non-null ancestry clusters", nn.root.nunique())
print(
    nn.groupby("family").agg(entries=("id", "size"), clusters=("root", "nunique")).to_string()
)
ratio_clusters_per_entry = nn.root.nunique() / len(nn)
print(f"clusters per non-null entry {ratio_clusters_per_entry:.3f}")
# Full-zoo extrapolation with family rates from D3 (as in round1) and family counts from edit.json
rates = d3.groupby("family").nonnull.mean().to_dict()
proj = {f: fam[f] * rates.get(f, 0.0) for f in fam}
print("family non-null rates (D3)", {k: round(v, 3) for k, v in rates.items()})
print("projected non-null full-zoo entries", round(sum(proj.values())), {k: round(v) for k, v in proj.items()})
print(
    "projected non-null full-zoo clusters (D3 clusters/entry ratio applied, approx)",
    round(sum(proj.values()) * ratio_clusters_per_entry),
)

# ---------------------------------------------------------------- C measured cost
hdr("C measured fast-tier wall time (eval.json minutes)")
print(d3.groupby("device").minutes.agg(["size", "mean", "median", "min", "max"]).round(2).to_string())
allmin = []
for i in evald:
    ev = json.load(open(ZOO / i / "eval.json"))
    allmin.append((ev.get("eval_device_id"), ev.get("minutes")))
am = pd.DataFrame(allmin, columns=["device", "minutes"])
print("all 389 evaluated entries by device")
print(am.groupby("device").minutes.agg(["size", "mean", "median"]).round(2).to_string())

# ---------------------------------------------------------------- D stored-field audit
hdr("D stored per-item fields in records.jsonl (D3 sample, all 300 entries)")
audit = Counter()
for i in d3.id:
    recs = [json.loads(l) for l in open(ZOO / i / "records.jsonl")]
    for r in recs:
        b = r["benchmark"]
        audit[(b, "items")] += 1
        if b == "mmlu_pro":
            audit[(b, "scores_len_%d" % len(r["extra"]["scores"]))] += 1
        elif b == "hellaswag":
            audit[(b, "scores_len_%d" % len(r["extra"]["scores"]))] += 1
        elif b == "kl_to_base":
            audit[(b, "first_token_kl_and_full_kl")] += int("first_token_kl" in r["extra"])
        elif b == "wikitext_ppl":
            audit[(b, "mean_nll_and_n_tokens")] += int("mean_nll" in r["extra"] and r["n_tokens"] > 0)
        elif b in ("refusal_rate", "over_refusal"):
            audit[(b, "output_text_only")] += int("output" in r["extra"])
        elif b in ("gsm8k", "math500", "ifeval"):
            audit[(b, "output_text_hit_max")] += int("output" in r["extra"])
for k in sorted(audit):
    print(k, audit[k])

# ---------------------------------------------------------------- E history effect
hdr("E history: M2 (sequence) vs M2-matched (stateless), P4 trajectory split, length>=4")


def load_arm(tag, seed):
    return pd.concat(
        [pd.read_parquet(f) for f in sorted(glob.glob(str(P4 / tag / f"seed-{seed}/fold-*.parquet")))]
    )


frames = []
for s in [0, 1, 2]:
    h = load_arm("d3v2-m2-trajectory", s)
    st = load_arm("d3v2-m2_matched-trajectory", s)
    h = h[["id", "target", "observed", "predicted", "trajectory_length", "ancestry_cluster", "family"]]
    st = st[["id", "target", "predicted"]].rename(columns={"predicted": "stateless"})
    m = h.merge(st, on=["id", "target"])
    m["seed"] = s
    frames.append(m)
M = pd.concat(frames)
M["eh"] = (M.observed - M.predicted).abs()
M["es"] = (M.observed - M.stateless).abs()


def ratio_power(sub, theta_star=0.15, label=""):
    """sub: rows with columns ancestry_cluster, eh, es, seed. Returns dict of stats."""
    per_seed = sub.groupby("seed").apply(lambda g: 1 - g.eh.sum() / g.es.sum())
    avg = sub.groupby(["ancestry_cluster", "id"]).agg(eh=("eh", "mean"), es=("es", "mean")).reset_index()
    cl = pd.DataFrame(
        {
            "D": avg.assign(d=avg.es - avg.eh).groupby("ancestry_cluster").d.sum(),
            "S": avg.groupby("ancestry_cluster").es.sum(),
        }
    )
    theta = cl.D.sum() / cl.S.sum()
    C = len(cl)
    sd_resid = (cl.D - theta_star * cl.S).std(ddof=1)
    mean_S = cl.S.mean()
    n_needed = (Z_POWER * sd_resid / (theta_star * mean_S)) ** 2 if theta_star != 0 else float("nan")
    boots = []
    idx = np.arange(C)
    for _ in range(4000):
        b = RNG.choice(idx, C)
        boots.append(cl.D.values[b].sum() / cl.S.values[b].sum())
    lo, hi = np.percentile(boots, [2.5, 97.5])
    # approximate MDE at the realised number of clusters (SD at theta=0)
    sd0 = cl.D.std(ddof=1)
    mde = Z_POWER * sd0 / (math.sqrt(C) * mean_S)
    return dict(
        label=label,
        rows=int(len(avg)),
        clusters=int(C),
        theta=round(float(theta), 4),
        boot_ci=(round(float(lo), 3), round(float(hi), 3)),
        seed_reductions=[round(float(x), 3) for x in per_seed.values],
        seed_sd=round(float(np.std(per_seed.values, ddof=1)), 3),
        cluster_sd_resid_at_theta_star=round(float(sd_resid / mean_S), 3),
        clusters_for_80pct_at_15pct=int(math.ceil(n_needed)),
        approx_mde_at_realised_clusters=round(float(mde), 3),
    )


E_rows = []
for tgt in ["mmlu_pro", "hellaswag"]:
    sub = M[(M.target == tgt) & (M.trajectory_length >= 4)]
    E_rows.append(ratio_power(sub, 0.15, f"{tgt} len>=4"))
    allr = M[(M.target == tgt)]
    E_rows.append(ratio_power(allr, 0.15, f"{tgt} all rows (sanity)"))
for r in E_rows:
    print(json.dumps(r))
# clusters needed for other effect sizes, pooled over the two benchmarks' length>=4 rows (mmlu_pro used for SD)
sub = M[(M.target == "mmlu_pro") & (M.trajectory_length >= 4)]
print("\nclusters needed for 80% power, mmlu_pro len>=4 (cluster SD at each theta*):")
for th in [0.05, 0.10, 0.15, 0.20, 0.30]:
    r = ratio_power(sub, th, "")
    print(f"  theta*={th:.2f}: clusters {r['clusters_for_80pct_at_15pct'] if th == 0.15 else None}")
    print(f"     -> {int(math.ceil((Z_POWER * r['cluster_sd_resid_at_theta_star'] / th) ** 2))}")
# Placebo: rows with no history (trajectory_length 1). M2 and M2-matched should agree in expectation here;
# a nonzero reduction is architecture/training-seed noise, not history.
for tgt in ["mmlu_pro", "hellaswag"]:
    for lab, cond in [("length 1 (no history)", M.trajectory_length == 1), ("length 2-3", M.trajectory_length.between(2, 3))]:
        r = ratio_power(M[(M.target == tgt) & cond], 0.15, f"{tgt} {lab}")
        print("PLACEBO " + json.dumps(r))
sub_h = M[(M.target == "hellaswag") & (M.trajectory_length >= 4)]
print("hellaswag len>=4 cluster-SD-based clusters needed:")
for th in [0.05, 0.10, 0.15, 0.20, 0.30]:
    r = ratio_power(sub_h, th, "")
    print(f"  theta*={th:.2f}: {int(math.ceil((Z_POWER * r['cluster_sd_resid_at_theta_star'] / th) ** 2))}")

# ---------------------------------------------------------------- F feature-addition power
hdr("F paired-MAE power for feature-addition effects on full D3 trajectory split (3 seeds)")
rows = []
for s in [0, 1, 2]:
    d = load_arm("d3v2-m2-trajectory", s)  # has statnn / descriptor_only / delta_norm columns
    d = d[d.target.isin(["mmlu_pro", "hellaswag"])]
    d = d.assign(seed=s)
    rows.append(d)
F = pd.concat(rows)
for comp in ["descriptor_only", "delta_norm"]:
    for tgt in ["mmlu_pro", "hellaswag"]:
        g = F[F.target == tgt].copy()
        g["eh"] = (g.observed - g.statnn).abs()
        g["es"] = (g.observed - g[comp]).abs()
        r = ratio_power(g, 0.05, f"StatNN vs {comp}, {tgt}")
        print(json.dumps(r))
        print(
            f"   clusters for 80% power at 5% MAE gain: {int(math.ceil((Z_POWER * r['cluster_sd_resid_at_theta_star'] / 0.05) ** 2))};"
            f" at 10%: {int(math.ceil((Z_POWER * r['cluster_sd_resid_at_theta_star'] / 0.10) ** 2))}"
        )

# ---------------------------------------------------------------- G rho power
hdr("G Fisher-z power: rho 0.6 vs rho 0.4 (one-sided alpha 0.05, H0: rho <= 0.4)")
Phi = lambda x: 0.5 * (1 + math.erf(x / math.sqrt(2)))  # noqa: E731
delta = math.atanh(0.6) - math.atanh(0.4)
print("  one-sample test (H0: rho<=0.4 vs rho=0.6), single predictor:")
for label, n in [
    ("D3 non-null, entries", len(nn)),
    ("D3 non-null, ancestry clusters", nn.root.nunique()),
    ("full-zoo non-null (projected), entries", round(sum(proj.values()))),
    ("full-zoo non-null (projected), clusters", round(sum(proj.values()) * ratio_clusters_per_entry)),
    ("D3 all, entries", len(d3)),
    ("D3 all, clusters", d3.root.nunique()),
]:
    power = Phi(delta * math.sqrt(n - 3) - 1.644854)
    print(f"  {label:44s} n={n:5d}  power={power:.2f}")
print("  two-sample, independent (round-1 definition: P(0.6 vs 0.4) for two separate samples of size n):")
for label, n in [("D3 non-null, entries", len(nn)), ("full-zoo non-null (projected), entries", round(sum(proj.values())))]:
    se2 = math.sqrt(2) / math.sqrt(n - 3)
    print(f"  {label:44s} n={n:5d}  power={Phi(delta / se2 - 1.959964):.2f}")
# SD uncertainty for the cluster-level H2 SD from 10 clusters (chi-square, df 9, 95%)
print("  note: the H2 cluster SD rests on 10 clusters; its 95% range is about x0.69 to x1.83 (chi-square, df=9).")
print("  Note: the non-null count and the cluster count, not total entries, set the power.")

# ---------------------------------------------------------------- H cluster budget
hdr("H minimum detectable MAE effect vs ancestry clusters (80% power, two-sided 0.05)")
ev_roots = {lineage(i)[0] for i in evald}
print("ancestry clusters among the 389 evaluated entries:", len(ev_roots))
print("ancestry clusters among all 1,247 zoo entries:", len({lineage(i)[0] for i in ed}))
SD_FEAT = 0.866  # measured cluster SD of StatNN-vs-descriptor MAE contrast (section F, mmlu_pro, 278 clusters)
for label, C in [
    ("D3 non-null clusters", nn.root.nunique()),
    ("D3 all clusters", d3.root.nunique()),
    ("evaluated clusters (389)", len(ev_roots)),
    ("zoo clusters, all entries", len({lineage(i)[0] for i in ed})),
    ("zoo clusters + 700 new independent edits", len({lineage(i)[0] for i in ed}) + 700),
    ("2,355 clusters (5% target)", 2355),
]:
    mde = Z_POWER * SD_FEAT / math.sqrt(C)
    print(f"  {label:44s} C={C:5d}  MDE={mde:.3f}")
print("  Required clusters for MDE 5%:", int(math.ceil((Z_POWER * SD_FEAT / 0.05) ** 2)),
      "; 10%:", int(math.ceil((Z_POWER * SD_FEAT / 0.10) ** 2)),
      "; 15%:", int(math.ceil((Z_POWER * SD_FEAT / 0.15) ** 2)))

# ---------------------------------------------------------------- I reliability gate at TF-v1 item counts
hdr("I reliability of each endpoint after expansion (signal = residual SD^2 - SE^2; reliability = signal/(signal+SE_new^2))")
# Inputs copied from round1-metrology.md section 1a (residual within family x device, 300-entry D3 panel).
# (name, residual SD, residual item SE at current items, current items, TF-v1 items)
R1_RESIDUALS = [
    ("mmlu_pro gold log-prob (nats)", 0.285, 0.058, 100, 1000),
    ("mmlu_pro gold margin (nats)", 0.439, 0.076, 100, 1000),
    ("mmlu_pro accuracy", 0.038, 0.025, 100, 1000),
    ("hellaswag gold log-prob (per char)", 0.0043, 0.0018, 100, 500),
    ("hellaswag accuracy", 0.013, 0.014, 100, 500),
    ("KL to base (nats)", 0.059, 0.0019, 50, 200),
    ("wikitext log-ppl", 0.029, 0.0074, 8, 64),
    ("gsm8k sampled accuracy", 0.059, 0.058, 50, 500),
]
for name, sd, se_now, n_now, n_new in R1_RESIDUALS:
    var_signal = sd ** 2 - se_now ** 2
    se_new = se_now * math.sqrt(n_now / n_new)
    if var_signal <= 0:
        print(f"  {name:36s} signal var < 0 (no residual signal beyond item noise); reliability undefined -> FAIL")
        continue
    rel = var_signal / (var_signal + se_new ** 2)
    print(f"  {name:36s} signal SD={math.sqrt(var_signal):.4f}  SE_now={se_now:.4f}@{n_now}  SE_new={se_new:.4f}@{n_new}  reliability={rel:.3f}  {'PASS' if rel >= 0.9 else 'FAIL'}")
print("  gate: reliability >= 0.90 (equivalently SD_signal/SE >= 3.2 at the TF-v1 item count)")
print("  note: SE scales as 1/sqrt(items) for item sampling; the 100-item SE is R1's paired item SE.")

