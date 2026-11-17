"""v2 full-refit pipeline invariants (RULING-wme-v2 A1, A2, A6); tiny epochs, behaviour only."""

import numpy as np

from wsp.wm import powersim_refit as R1
from wsp.wm import powersim_refit_v2 as R2
from wsp.wm import powersim_v1 as PV
from wsp.wm import synthetic as S
from wsp.wm import wmcp as CP


def test_bank_is_v1_bank_and_chart_column_is_seen_in_training():
    g = PV.graph(8, 6)
    w = S.SynthWorld(kind="saturating", seed=3, n_dirs=8, n_obj=6)
    a, b = R1.build_bank(w, g, 5), R2.build_bank_v2(w, g, 5)
    assert all(np.array_equal(x["x"], y["x"]) for x, y in zip(a, b, strict=True))
    st = R2.states_of(w, b)
    spec = R2.spec_with_dead(w)
    chart = [s["r"][spec.names.index("dead")] for s in st]
    assert any(c != 0 for c in chart) and any(c == 0 for c in chart)


def test_replicate_runs_and_is_path_consistent():
    r = R2.replicate(
        0, {"r_f": 8, "n_blk": 6, "n_cal": 12}, null=True, cfg=CP.CPConfig(epochs=3), n_flip=199
    )
    assert r["decomposition_gap_max"] <= 1e-12 and not r["reject"]
    assert r["m"] >= 2 and set(r["static_chosen"]) <= {"S1", "S2", "S3"}


def test_refit_cli_passes_the_version_to_workers(tmp_path, monkeypatch):
    """Regression: the runner's task tuples must carry the version (v2 crashed on unpacking)."""
    import importlib.util
    import json
    from pathlib import Path

    p = Path(__file__).resolve().parents[1] / "scripts/wme_powersim_refit.py"
    spec = importlib.util.spec_from_file_location("wme_powersim_refit_cli", p)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    seen = []

    class _Pool:
        def __init__(self, *a, **k):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def imap_unordered(self, fn, todo, chunksize=1):
            for t in todo:
                rep, _cell, null, _kappa, _gamma, version = t
                seen.append(version)
                yield {"rep": rep, "null": null, "seconds": 0.0}

    class _Ctx:
        Pool = _Pool

    monkeypatch.setattr(mod.mp, "get_context", lambda *_: _Ctx)
    a = type(
        "A",
        (),
        {
            "out": str(tmp_path),
            "cell": '{"r_f": 8, "n_blk": 6, "n_cal": 12}',
            "reps_null": 2,
            "reps_alt": 1,
            "kappa": 1.0,
            "gamma": 3.0,
            "procs": 1,
            "version": "v2",
        },
    )
    mod.cmd_run(a)
    assert seen == ["v2"] * 3
    assert json.loads((tmp_path / "config.json").read_text())["version"] == "v2"


def test_refit_report_futility_rule(tmp_path):
    import importlib.util
    import json
    from pathlib import Path

    p = Path(__file__).resolve().parents[1] / "scripts/wme_refit_v2_report.py"
    spec = importlib.util.spec_from_file_location("wme_refit_v2_report", p)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert mod.k_required(600) == 497 and mod.k_required(60) == 54
    d = tmp_path / "c"
    d.mkdir()
    (d / "config.json").write_text(
        json.dumps({"cell": {"r_f": 8}, "gamma": 3.0, "reps_null": 60, "reps_alt": 60})
    )
    base = {
        "p": 1.0,
        "lb": 0.0,
        "c2i": True,
        "decomposition_gap_max": 0.0,
        "static_chosen": ["S2"],
        "seconds": 1.0,
    }
    rows = [dict(base, rep=i, null=False, reject=False, stat=-1.0, ct=-0.1) for i in range(7)]
    rows += [dict(base, rep=i, null=True, reject=False, stat=0.0, ct=0.0) for i in range(7)]
    (d / "results.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    c = mod.cell(d)
    assert (
        c["futility"]["cannot_pass"]
        and not c["alt"]["pass"]
        and c["completed"] == {"null": 7, "alt": 7}
    )
    assert c["null"]["exact_zero_share"] == 1.0 and c["alt"]["share_stat_negative"] == 1.0
