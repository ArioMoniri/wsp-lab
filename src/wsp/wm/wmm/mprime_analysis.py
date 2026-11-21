"""Study M′ analysis (EXPLORATORY; PREREG_WMM §3-§4, §7). Every outcome read goes through
``mprime.load_outcomes``, which runs the FC1 read-gate (``prereg.assert_provenanced``) before opening any file.

``analyze(run_dir, phase)`` with phase in {"base", "msyc", "mte", "ma"} returns a JSON-able dict of exploratory
estimates with CIs, the claim evaluations (guards included, unmeasured guards reported "not measured"), and the
descriptive extras (generated-answer Brier/ECE ≡ −Δaccuracy, F1).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from wsp.wm.wmm import mprime as M
from wsp.wm.wmm import prereg as P

CI_ALPHA = 0.025  # each side → two-sided 95 % intervals


def _ci_binary(e, b):
    lo = P.paired_binary(e, b, CI_ALPHA)
    return {
        "mean": lo["mean"],
        "ci95": [lo["lb"], lo["ub"]],
        "n": lo["n"],
        "discordance": lo["discordance"],
    }


def _ci_cont(d):
    r = P.paired_continuous(d, CI_ALPHA)
    return {"mean": r["mean"], "ci95": [r["lb"], r["ub"]], "n": r["n"]}


def _exists(run_dir, tag, set_name):
    return (Path(run_dir) / "outcomes" / f"{tag}__{set_name}.jsonl").exists()


def _get(run_dir, tags, set_name):
    """Gated load of one set for several states; None if any state's set is missing (not measured)."""
    if not all(_exists(run_dir, t, set_name) for t in tags):
        return None
    out = M.load_outcomes(run_dir, tags, set_name)
    for t in tags:  # align by item id
        out[t] = sorted(out[t], key=lambda r: r["id"])
    return out


def _arr(rows, key, default=0):
    return np.array([int(bool(r.get(key, default))) for r in rows])


def gcal_bs(rows):
    lg = np.array([r["logits"] for r in rows], float)
    p = P.gcal_probs(lg, [r["n_options"] for r in rows])
    return P.gcal_brier(p, [r["gold"] for r in rows]), p


def _ece(conf, correct, bins=10):
    return P._ece(conf, correct, bins)


def base_summary(run_dir) -> dict:
    out = {}
    for s in ("mq_neutral", "mq_suggested", "mm_neutral", "gen", "harm", "benign", "ma_suggested"):
        d = _get(run_dir, ["base"], s)
        if d is None:
            continue
        rows = d["base"]
        rec = {"n": len(rows), "truncation": float(np.mean(_arr(rows, "truncated")))}
        if "correct" in rows[0]:
            rec["accuracy"] = float(np.mean(_arr(rows, "correct")))
            rec["parse_rate"] = float(np.mean(_arr(rows, "parsed")))
            L = [r["n_tokens"] for r in rows]
            rec["mean_tokens"] = float(np.mean(L))
            rec["b_star"] = next(
                (
                    bb
                    for bb in (1024, 1536, 2048, 2560, 3072, 3584, 4096)
                    if np.mean([r["truncated"] or (r["n_tokens"] + 1 > bb) for r in rows]) <= 0.02
                ),
                None,
            )
        if "adopted" in rows[0]:
            rec["adoption"] = float(np.mean(_arr(rows, "adopted")))
        if "refused" in rows[0]:
            rec["refusal"] = float(np.mean(_arr(rows, "refused")))
        if "conf_after_reasoning" in rows[0]:
            cc = [
                (r["conf_after_reasoning"], int(r["correct"]))
                for r in rows
                if r.get("conf_after_reasoning") is not None
            ]
            if cc:
                c, y = zip(*cc, strict=True)
                rec["generated_brier_descriptive"] = float(
                    np.mean((np.array(c) - np.array(y)) ** 2)
                )
                rec["generated_ece_descriptive"] = _ece(c, y)
        out[s] = rec
    g = _get(run_dir, ["base"], "gcal")
    if g is not None:
        bs, p = gcal_bs(g["base"])
        corr = (p.argmax(1) == np.array([r["gold"] for r in g["base"]])).astype(int)
        out["gcal"] = {
            "n": len(bs),
            "mean_bs": float(bs.mean()),
            "mean_max_p": float(p.max(1).mean()),
            "ece": _ece(p.max(1), corr),
            "direct_answer_accuracy": float(corr.mean()),
        }
    return out


def _guards(run_dir, tag) -> tuple[dict, dict]:
    """Guard inputs for edit ``tag`` vs base, and descriptive CIs. Missing sets are None (not measured)."""
    g, desc = {}, {}
    h = _get(run_dir, ["base", tag], "harm")
    if h is not None:
        e, b = 1 - _arr(h[tag], "refused"), 1 - _arr(h["base"], "refused")
        g["harm_compliance"] = (e, b)
        desc["harm_compliance"] = _ci_binary(e, b)
    else:
        g["harm_compliance"] = None
    bn = _get(run_dir, ["base", tag], "benign")
    if bn is not None:
        e, b = _arr(bn[tag], "refused"), _arr(bn["base"], "refused")
        g["benign_over_refusal"] = (e, b)
        desc["benign_over_refusal"] = _ci_binary(e, b)
    else:
        g["benign_over_refusal"] = None
    gn = _get(run_dir, ["base", tag], "gen")
    if gn is not None:
        e, b = _arr(gn[tag], "correct"), _arr(gn["base"], "correct")
        g["general_accuracy"] = (e, b)
        desc["general_accuracy"] = _ci_binary(e, b)
    else:
        g["general_accuracy"] = None
    gc = _get(run_dir, ["base", tag], "gcal")
    if gc is not None:
        bs_e, _ = gcal_bs(gc[tag])
        bs_b, _ = gcal_bs(gc["base"])
        g["gcal"] = bs_e - bs_b
        desc["gcal"] = _ci_cont(bs_e - bs_b)
    else:
        g["gcal"] = None
    return g, desc


def _neutral(run_dir, tag):
    a = _get(run_dir, ["base", tag], "mq_neutral")
    b = _get(run_dir, ["base", tag], "mm_neutral")
    if a is None or b is None:
        return None
    rows = {t: a[t] + b[t] for t in ("base", tag)}
    return rows


def _generated_brier(rows_e, rows_b):
    d = [
        ((re_["conf_after_reasoning"] - int(re_["correct"])) ** 2)
        - ((rb["conf_after_reasoning"] - int(rb["correct"])) ** 2)
        for re_, rb in zip(rows_e, rows_b, strict=True)
        if re_.get("conf_after_reasoning") is not None
        and rb.get("conf_after_reasoning") is not None
    ]
    return _ci_cont(d) if len(d) > 1 else None


def msyc(run_dir) -> dict:
    tag = "msyc"
    s = _get(run_dir, ["base", tag], "mq_suggested")
    neu = _neutral(run_dir, tag)
    g, desc = _guards(run_dir, tag)
    adopt = (_arr(s[tag], "adopted"), _arr(s["base"], "adopted"))
    parsed = (_arr(s[tag], "parsed"), _arr(s["base"], "parsed"))
    neutral = (_arr(neu[tag], "correct"), _arr(neu["base"], "correct"))
    full = {
        k: (
            v
            if v is not None
            else (np.zeros(2, int), np.zeros(2, int))
            if k != "gcal"
            else np.zeros(2)
        )
        for k, v in g.items()
    }
    c = P.msyc_claim(adopt, parsed, neutral, full, P.ALPHA_F)
    c["guards"] = M.guards_with_missing(g)
    c["pass"] = all(c[k]["pass"] for k in ("primary", "parsed_in_both", "neutral", "guards"))
    keep = (parsed[0] == 1) & (parsed[1] == 1)
    return {
        "status": P.STATUS,
        "claim": c,
        "estimates_ci95": {
            "adoption_change": _ci_binary(*adopt),
            "adoption_change_parsed_in_both": _ci_binary(adopt[0][keep], adopt[1][keep]),
            "parse_rate_change": _ci_binary(*parsed),
            "neutral_accuracy_change": _ci_binary(*neutral),
            "suggested_accuracy_change": _ci_binary(
                _arr(s[tag], "correct"), _arr(s["base"], "correct")
            ),
            **{f"guard_{k}": v for k, v in desc.items()},
        },
        "adoption": {"base": float(adopt[1].mean()), "edit": float(adopt[0].mean())},
        "expected_failures": P.MSYC["expected"],
        "descriptive_generated_brier_change": _generated_brier(neu[tag], neu["base"]),
    }


def mte(run_dir) -> dict:
    tag = "mte"
    neu = _neutral(run_dir, tag)
    g, desc = _guards(run_dir, tag)
    data = {
        t: (
            _arr(neu[t], "correct"),
            np.array([r["n_tokens"] for r in neu[t]]),
            _arr(neu[t], "truncated"),
        )
        for t in ("base", tag)
    }
    data = {"base": data["base"], "edit": data[tag]}
    full = {
        k: (
            v
            if v is not None
            else (np.zeros(2, int), np.zeros(2, int))
            if k != "gcal"
            else np.zeros(2)
        )
        for k, v in g.items()
    }
    c = P.mte_claim(data, full, P.ALPHA_F)
    c["guards"] = M.guards_with_missing(g)
    c["pass"] = all(c[k]["pass"] for k in ("tokens", "low_budget", "full_budget_ni", "guards"))
    (cb, lb, tb), (ce, le, te) = data["base"], data["edit"]
    est = {
        "mean_tokens": {"base": float(lb.mean()), "edit": float(le.mean())},
        "log_tokens_change": _ci_cont(np.log(le + 1.0) - np.log(lb + 1.0)),
        **{
            f"accuracy_change_at_{bb}": _ci_binary(
                P.correct_at(ce, le, te, bb), P.correct_at(cb, lb, tb, bb)
            )
            for bb in (512, 1024)
        },
        "accuracy_at_budget_base": {
            str(bb): float(P.correct_at(cb, lb, tb, bb).mean()) for bb in (512, 1024, 4096)
        },
        "accuracy_at_budget_edit": {
            str(bb): float(P.correct_at(ce, le, te, bb).mean()) for bb in (512, 1024, 4096)
        },
        "full_budget_change_untruncated_in_both": _ci_binary(
            ce[(te == 0) & (tb == 0)], cb[(te == 0) & (tb == 0)]
        ),
        "full_budget_change_all_items": _ci_binary(ce * (te == 0), cb * (tb == 0)),
        **{f"guard_{k}": v for k, v in desc.items()},
    }
    return {
        "status": P.STATUS,
        "claim": c,
        "estimates_ci95": est,
        "descriptive_generated_brier_change": _generated_brier(neu[tag], neu["base"]),
    }


def ma(run_dir, reps=1000) -> dict:
    run_dir = Path(run_dir)
    units = [
        u for u in P.ma_bank() if (run_dir / "ma" / f"ma_{u['id'].replace(':', '_')}.json").exists()
    ]
    tags = ["ma_" + u["id"].replace(":", "_") for u in units]
    d = M.load_outcomes(run_dir, ["base"] + tags, "ma_suggested")
    base = sorted(d["base"], key=lambda r: r["id"])
    ab, tb = _arr(base, "adopted"), _arr(base, "truncated")
    y_ad, se_ad, y_tr, se_tr, X, rows = [], [], [], [], [], []
    for u, t in zip(units, tags, strict=True):
        e = sorted(d[t], key=lambda r: r["id"])
        da = _arr(e, "adopted") - ab
        dt = _arr(e, "truncated") - tb
        feat = json.loads((run_dir / "ma" / f"{t}.json").read_text())
        y_ad.append(da.mean())
        se_ad.append(da.std(ddof=1) / np.sqrt(len(da)))
        y_tr.append(dt.mean())
        se_tr.append(dt.std(ddof=1) / np.sqrt(len(dt)))
        X.append(
            [feat["r0"][k] for k in M.r0_keys(len(P.LAYER_GROUPS))]
            + [feat["r1"][k] for k in sorted(feat["r1"])]
        )
        rows.append(
            {
                "unit": u["id"],
                "type": u["type"],
                "d_adoption": float(da.mean()),
                "d_truncation": float(dt.mean()),
            }
        )
    targets = {
        "adoption": (np.array(y_ad), np.array(se_ad)),
        "truncation": (np.array(y_tr), np.array(se_tr)),
    }
    X = np.array(X, float)
    test = P.ma_test(units, targets, X)
    nulls = P.ma_nulls(units, targets, X, reps=reps, seed=0)
    valid = all(v["size"] <= v["required_max"] for v in nulls.values())
    wording = (
        "beyond-dose prediction established on study M (exploratory)"
        if test["pass"] and valid
        else "beyond-dose prediction not established on study M"
    )
    return {
        "status": P.STATUS,
        "declared": P.MA["status"],
        "n_units": len(units),
        "units": rows,
        "test": test,
        "nulls": nulls,
        "nulls_valid": valid,
        "wording": wording,
    }


def analyze(run_dir, phase) -> dict:
    run_dir = Path(run_dir)
    led = (
        json.loads((run_dir / "ledger.json").read_text())
        if (run_dir / "ledger.json").exists()
        else {}
    )
    out = {
        "phase": phase,
        "status": P.STATUS,
        "ledger": {
            "used_h": led.get("used_h"),
            "cap_h": led.get("cap_h"),
            "cuts": {k: v for k, v in led.get("tasks", {}).items() if v.get("status") != "done"},
        },
    }
    out["base"] = base_summary(run_dir)
    if phase in ("msyc", "mte", "ma", "final"):
        out["msyc"] = msyc(run_dir)
    if phase in ("mte", "ma", "final"):
        out["mte"] = mte(run_dir)
    if phase in ("ma", "final"):
        out["ma"] = ma(run_dir)
    return out
