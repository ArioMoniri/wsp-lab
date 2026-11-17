"""scripts/wme_v2b_sim.py: the frozen selection and calibration rules (RULING-wme-v2b A7)."""

import importlib.util
import json
from pathlib import Path

import pytest

P = Path(__file__).resolve().parents[1] / "scripts/wme_v2b_sim.py"
spec = importlib.util.spec_from_file_location("wme_v2b_sim", P)
SIM = importlib.util.module_from_spec(spec)
spec.loader.exec_module(SIM)


def test_cpsel_pick_prefers_the_earlier_config_within_the_tie():
    rows = [
        {"name": "C0", "errors": [1.00] * 5},
        {"name": "C1", "errors": [0.99] * 5},
        {"name": "C2", "errors": [0.90] * 5},
        {"name": "C3", "errors": [1.5] * 5},
    ]
    assert SIM.cpsel_pick(rows)[0] == "C2"
    rows[2]["errors"] = [0.995] * 5
    assert SIM.cpsel_pick(rows)[0] == "C0"  # 0.99 and 0.995 are within 2 % of C0's 1.00


def test_cal_interpolates_kappa_to_the_target(tmp_path):
    rows = [
        {"kappa": k, "rep": 0, "errors": [m] * 3, "seconds": 1.0}
        for k, m in ((1, 1.0), (2, 2.0), (4, 3.0))
    ]
    (tmp_path / "results.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))

    class A:
        out = str(tmp_path)

    SIM.cmd_cal_summary(A)
    s = json.loads((tmp_path / "summary.json").read_text())
    assert s["kappa_star"] == pytest.approx(2.0 + 0.17 * 2.0)


def test_json_args_inline_or_from_file(tmp_path):
    f = tmp_path / "c.json"
    f.write_text('{"r_f": 16}')
    assert (
        SIM._arg('{"r_f": 8}') == {"r_f": 8}
        and SIM._arg(f"@{f}") == {"r_f": 16}
        and SIM._arg("") == {}
    )
    assert SIM.k_required(600) == 497
