"""The wme-c1 controller: stages in the registered order on one instance at a time (PREREG_WME §II.18, §II.28).

Stages read only ELIGIBLE upstream payloads (``guard.resolve``) and files whose sha256 those payloads record.
Each stage constructs its phase's ``ProductionGuard`` (verified fresh replay, new owner epoch), publishes the epoch
to the worker fence, opens (or resumes) the instance, and runs its units canary-bracketed. Nothing here reads a
sealed split before ``designate`` has written and hashed the winners manifest.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import time
from pathlib import Path

from wsp.wm import e2 as E2
from wsp.wm import harness as H
from wsp.wm import runguard as RG
from wsp.wm.c1 import ctl as C
from wsp.wm.c1 import study as SD
from wsp.wm.c1.schema import ARMS_PRIMARY, ARMS_SECONDARY

LLM_ARMS = ("wsr_reader", "fugu_team")


def sha_file(p) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


class Controller:
    def __init__(
        self,
        study: SD.Study,
        remote: C.Remote,
        transport,
        instance: str,
        gpus: int = 1,
        log=print,
        client_factory=None,
        chunk: int = 16,
        max_chunk_s: float = 7200.0,
        canary_args=None,
        panels=None,
    ):
        self.study, self.remote, self.t = study, remote, transport
        self.instance, self.gpus = instance, gpus
        self.log = log
        self.client_factory = client_factory
        self.chunk, self.max_chunk_s = chunk, max_chunk_s
        self.canary_args = canary_args or {}
        self.panels = panels or {}
        self.out = Path(study.runs) / "stages"
        self.out.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ guard and files
    def guard(self, phase: str) -> RG.ProductionGuard:
        g = self.study.guard(phase, self.instance)
        self.t.set_fence(g.epoch, self.instance)
        g.open_instance(self.gpus)
        return g

    def find_file(self, name: str, sha: str) -> Path:
        """The result file ``name`` with content ``sha`` (from an eligible payload) among the fetched results."""
        for p in sorted((Path(self.remote.local) / "results").glob(f"*/{name}")):
            if sha_file(p) == sha:
                return p
        raise FileNotFoundError(f"{name} with sha256 {sha[:12]} not among fetched results")

    def save(self, stage: str, obj):
        p = self.out / f"{stage}.json"
        p.write_text(json.dumps(obj, sort_keys=True, default=str))
        return p

    def run(self, g, units, fn=None):
        return self.study.run(
            g,
            self.remote,
            self.t,
            units,
            chunk=self.chunk,
            max_chunk_s=self.max_chunk_s,
            canary_args=self.canary_args,
            fn=fn,
        )

    # ------------------------------------------------------------------ bank
    def stage_bank(
        self,
        manifest: dict,
        roots_rec: dict,
        blocks=None,
        field_groups=range(8),
        inert=True,
        vocab=True,
    ) -> dict:
        g = self.guard("bank")
        pre = SD.bank_units(manifest, roots_rec, field_groups, vocab)
        pre.append(({"kind": "bank_base", "args": {}, "label": "bank:base"}, "bank:base"))
        res = self.run(g, pre)
        base = res["bank:base"]
        if base is None or base.get("fallback"):
            raise RuntimeError("the bank base unit has no eligible result")
        missing = [
            k for k, v in res.items() if k.startswith("field:") and (v is None or v.get("fallback"))
        ]
        if missing:
            raise RuntimeError(f"field units without eligible results: {missing}")
        g = self.guard("bank")
        states = self.run(g, SD.bank_state_units(manifest, roots_rec, base, blocks, inert))
        out = {"pre": res, "states": states}
        self.save("bank", out)
        return out

    # ------------------------------------------------------------------ confirmatory prep and E2
    def stage_prep(self, bank_bundle_path: Path, calibration, strata, inner_seed: int) -> dict:
        g = self.guard("confirmatory")
        res = self.run(
            g,
            [
                ({"kind": "library", "args": {}, "label": "e2:library"}, "e2:library"),
                ({"kind": "dev_base", "args": {}, "label": "e2:dev_base"}, "e2:dev_base"),
            ],
        )
        lib, devb = res["e2:library"], res["e2:dev_base"]
        if not lib or lib.get("fallback") or not devb or devb.get("fallback"):
            raise RuntimeError("library or dev base has no eligible result")
        fit_args = {
            "bank_sha256": sha_file(bank_bundle_path),
            "calibration": list(calibration),
            "strata": strata,
            "inner_seed": int(inner_seed),
        }
        g = self.guard("confirmatory")
        fit = self.run(
            g,
            [
                (
                    {
                        "kind": "fit",
                        "args": fit_args,
                        "uploads": {"bank_file": str(bank_bundle_path)},
                        "label": "e2:fit",
                    },
                    "e2:fit",
                )
            ],
        )["e2:fit"]
        if not fit or fit.get("fallback"):
            raise RuntimeError("the fit has no eligible result")
        out = {"library": lib, "dev_base": devb, "fit": fit}
        self.save("prep", out)
        return out

    def e2_inputs(self, prep: dict, with_fit=True):
        lib = self.find_file("library.pt", prep["library"]["sha256"])
        dev = self.find_file("dev_base.json", prep["dev_base"]["sha256"])
        fit = self.find_file("fit.pkl", prep["fit"]["sha256"]) if with_fit else None
        return SD.e2_inputs(lib, dev, fit)

    def stage_e2(self, prep: dict, episodes=range(SD.N_EP), arms=None, b: int = SD.B) -> dict:
        inputs, uploads = self.e2_inputs(prep)
        g = self.guard("confirmatory")
        units = SD.e2_units(inputs, uploads, episodes, arms, b)

        def fn(recipe):
            if recipe["args"]["arm"] in LLM_ARMS:
                return SD.reader_unit_fn(
                    self.study, g, self.remote, self.t, recipe, self.client_factory
                )
            args = dict(recipe["args"])
            for name, local in (recipe.get("uploads") or {}).items():
                args[name] = self.t.push(local)
            payload, _d = self.remote.call(
                recipe["kind"], g.epoch, g.instance, self.study.key(recipe), recipe["label"], args
            )
            return payload

        res = self.run(g, units, fn=fn)
        self.save("e2", res)
        return res

    # ------------------------------------------------------------------ S-BoN at B_par from ledger seconds
    def b_par(self) -> dict:
        st = RG.read_study(self.study.state)
        bank_s = field_s = 0.0
        for r in st_records(self.study.state):
            if r.get("type") == "unit_time" and r.get("phase") == "bank":
                if str(r.get("note", "")).startswith("field:"):
                    field_s += float(r["gpu_s"])
                else:
                    bank_s += float(r["gpu_s"])
        _ = st
        b = SD.parity_budget(bank_s, field_s)
        return {
            "B_par": b,
            "inputs": {
                "bank_gpu_s": bank_s,
                "field_gpu_s": field_s,
                "n_ep": SD.N_EP,
                "t_bundle_s": SD.T_BUNDLE_S,
            },
        }

    def stage_sbon(self, prep: dict, bpar: dict, episodes=range(SD.N_EP)) -> dict:
        inputs, uploads = self.e2_inputs(prep, with_fit=False)
        self.study.set_sbon_price(bpar["B_par"])
        g = self.guard("sbon")
        res = self.run(g, SD.sbon_units(inputs, uploads, bpar["B_par"], episodes))
        self.save("sbon", {"bpar": bpar, "results": res})
        return res

    # ------------------------------------------------------------------ designation and the winners manifest
    def designate(self, e2res: dict, sbon: dict, winners_dir: Path) -> dict:
        e1 = SD.designate(e2res)
        rows = []
        for lab, p in list(e2res.items()) + list(sbon.items()):
            parts = lab.split(":")
            arm, ep = (parts[1], int(parts[2])) if parts[0] == "e2" else ("sbon", int(parts[1]))
            fb = p is None or bool(p.get("fallback"))
            w = None if fb else (p or {}).get("winner")
            rows.append(
                {
                    "arm": arm,
                    "episode": ep,
                    "fallback": fb,
                    "edit": SD.winner_edit(arm, None if fb else p),
                    "theta_hash": w.get("theta_hash") if w else None,
                    "dev_lcb": w.get("lcb") if w else None,
                }
            )
        winners_dir.mkdir(parents=True, exist_ok=True)
        path = winners_dir / "winners_manifest.json"
        sha = E2.winners_manifest(rows, e1, path)
        out = {"path": str(path), "sha256": sha, "e1": e1, "n_rows": len(rows)}
        self.save("winners", out)
        return out

    # ------------------------------------------------------------------ the single sealed read
    def stage_sealed(self, prep: dict, winners: dict, test_file: Path, per_unit: int = 16) -> dict:
        man = json.loads(Path(winners["path"]).read_bytes())
        if sha_file(winners["path"]) != winners["sha256"]:
            raise RuntimeError("winners manifest changed after freezing")
        inputs, uploads = self.e2_inputs(prep, with_fit=False)
        states, seen = [{"key": "base", "edit": {"kind": "base"}, "theta_hash": None}], {"base"}
        for r in man["winners"]:
            if r["fallback"] or r["edit"]["kind"] == "base":
                continue
            key = r["theta_hash"] or json.dumps(r["edit"], sort_keys=True)
            if key in seen:
                continue
            seen.add(key)
            edit = dict(r["edit"])
            if edit["kind"] == "lora_sft":
                edit["file"] = str(self.find_file("lora_sft_winner.pt", edit["sha256"]))
            states.append({"key": key, "edit": edit, "theta_hash": r["theta_hash"]})
        g = self.guard("confirmatory")
        units = []
        for i in range(0, len(states), per_unit):
            chunk = states[i : i + per_unit]
            ups = dict(uploads, test_file=str(test_file))
            lora_up = {
                f"lora{j}": s["edit"]["file"]
                for j, s in enumerate(chunk)
                if s["edit"]["kind"] == "lora_sft"
            }
            args = dict(
                inputs,
                test_sha256=sha_file(test_file),
                states=[{k: v for k, v in s.items()} for s in chunk],
            )
            units.append(
                (
                    {
                        "kind": "sealed_eval",
                        "args": args,
                        "uploads": ups,
                        "lora": lora_up,
                        "label": f"sealed:{i // per_unit:03d}",
                    },
                    f"sealed:{i // per_unit:03d}",
                )
            )

        def fn(recipe):
            args = dict(recipe["args"])
            for name, local in (recipe.get("uploads") or {}).items():
                args[name] = self.t.push(local)
            st = []
            for s in args["states"]:
                s = dict(s)
                if s["edit"]["kind"] == "lora_sft":
                    s["edit"] = dict(s["edit"], file=self.t.push(s["edit"]["file"]))
                st.append(s)
            args["states"] = st
            payload, _d = self.remote.call(
                "sealed_eval", g.epoch, g.instance, self.study.key(recipe), recipe["label"], args
            )
            return payload

        res = self.run(g, units, fn=fn)
        self.save("sealed", res)
        return res

    # ------------------------------------------------------------------ E1 generation (single sealed_gen read)
    def stage_gen(self, prep: dict, winners: dict, gen_file: Path, sampled_seeds) -> dict:
        e1 = winners["e1"]
        if e1.get("status") != "designated":
            self.save("gen", {"skipped": e1.get("status")})
            return {"skipped": e1.get("status")}
        inputs, uploads = self.e2_inputs(prep, with_fit=False)
        g = self.guard("confirmatory")
        decs = [["greedy"]] + [[f"sampled:{s}"] for s in sampled_seeds]
        units = []
        for d in decs:
            args = dict(
                inputs,
                items_sha256=sha_file(gen_file),
                decodings=d,
                edit={"kind": "pids", "pids": list(e1["candidate"])},
                theta_hash=e1["theta_hash"],
                save_tensors=d == ["greedy"],
            )
            units.append(
                (
                    {
                        "kind": "e1_gen",
                        "args": args,
                        "uploads": dict(uploads, items_file=str(gen_file)),
                        "label": f"gen:{d[0]}",
                    },
                    f"gen:{d[0]}",
                )
            )

        def fn(recipe):
            args = dict(recipe["args"])
            for name, local in (recipe.get("uploads") or {}).items():
                args[name] = self.t.push(local)
            ser, _ = self.remote.call(
                "edit_bundles",
                g.epoch,
                g.instance,
                hashlib.sha256((self.study.key(recipe) + ":edit").encode()).hexdigest(),
                recipe["label"] + ":edit",
                args,
            )
            args["winner"] = {
                "bundles": ser["bundles"],
                "theta_hash": args["theta_hash"],
                "candidate": args["edit"]["pids"],
            }
            payload, _d = self.remote.call(
                "e1_gen", g.epoch, g.instance, self.study.key(recipe), recipe["label"], args
            )
            return payload

        res = self.run(g, units, fn=fn)
        self.save("gen", res)
        return res

    # ------------------------------------------------------------------ S-TE
    def stage_ste(self, panel_files: dict, gen: dict | None = None) -> dict:
        g = self.guard("ste")
        cal = self.run(
            g, [({"kind": "ste_calibrate", "args": {}, "label": "ste:calibrate"}, "ste:calibrate")]
        )["ste:calibrate"]
        if not cal or cal.get("fallback"):
            raise RuntimeError("S-TE calibration has no eligible result")
        g = self.guard("ste")
        tr = self.run(g, [({"kind": "ste_train", "args": {}, "label": "ste:train"}, "ste:train")])[
            "ste:train"
        ]
        if not tr or tr.get("fallback"):
            raise RuntimeError("S-TE training has no eligible result")
        g = self.guard("ste")
        args = {
            "margins": cal["margins"],
            "lora_factors_sha256": tr["factors_sha256"],
            "panel_files": {k: {"sha256": sha_file(v)} for k, v in panel_files.items()},
        }

        def fn(recipe):
            a = dict(recipe["args"])
            a["panel_files"] = {
                k: {"path": self.t.push(panel_files[k]), "sha256": v["sha256"]}
                for k, v in a["panel_files"].items()
            }
            a["lora_factors"] = tr.get("factors_path")
            wt = ((gen or {}).get("gen:greedy") or {}).get("winner_tensors")
            if wt:
                a["e1_winner_tensors"] = (
                    wt  # descriptive E1-winner generated safety (§II.12.3), same instance
                )
            payload, _d = self.remote.call(
                "ste_generate", g.epoch, g.instance, self.study.key(recipe), recipe["label"], a
            )
            return payload

        gen = self.run(
            g,
            [({"kind": "ste_generate", "args": args, "label": "ste:generate"}, "ste:generate")],
            fn=fn,
        )["ste:generate"]
        out = {"calibrate": cal, "train": tr, "generate": gen}
        self.save("ste", out)
        return out

    # ------------------------------------------------------------------ close
    def close(self, phase: str = "confirmatory"):
        g = self.study.guard(phase, self.instance)
        self.t.set_fence(g.epoch, self.instance)
        rec = C.run_canary(g, self.remote, self.canary_args)
        g.close_instance()
        return rec


def st_records(state_dir) -> list:
    out = []
    p = Path(state_dir) / RG.LOG_NAME
    for line in p.read_text().splitlines():
        out.append(json.loads(line))
    return out


def write_jsonl_gz(path, rows):
    with gzip.open(path, "wt") as f:
        for r in rows:
            f.write(json.dumps(r, sort_keys=True) + "\n")


def now():
    return time.time()


__all__ = ["Controller", "ARMS_PRIMARY", "ARMS_SECONDARY", "H"]
