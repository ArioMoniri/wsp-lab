"""Registered E2 arms: declared information access and training exposure, the per-(arm, episode) context and store,
and the §6.5.5 winner rule (PREREG_WME §6.5, §6.8.3, §7; Board A condition 10).

* **Registry** (:data:`ARM_SPECS`): the two primary arms and the Holm family of nine secondary arms. Each
  :class:`ArmSpec` declares the arm's group (R pool-sharing rankers, D pool-sharing direct, C different action
  space), action space, the resources it may read (:data:`ACCESS`; never :data:`FORBIDDEN`), its training exposure
  (:data:`EXPOSURE`), whether it is charged the shared bank cost, whether it calls an LLM (and so runs under the
  reader token cap), and its charged cost categories (``wm.controls.COST_CATEGORIES``). Every arm uses the one
  ``wm.plan.BUNDLE_DEFINITION`` and every arm that reads field features uses the one shared field
  (:data:`SHARED_FIELD_SPEC`, which includes the vocabulary-row approximation of ``wm.vocabfield``).
* **Context** (:class:`ArmContext`): what an arm's code actually receives. ``get`` refuses any resource not in the
  arm's declared access (and anything forbidden) with :class:`AccessDenied`, and passes every resource through the
  R0 guard (``wm.access``), so a non-θ₀ artefact can never reach a ranker. ``evaluate`` is the only route to a real
  dev evaluation: it charges the shared ``BudgetLedger`` under the arm's name, keyed by θ-hash, so a repeat is
  served from the arm's own store at 0 units without a second evaluation.
* **Store** (:class:`StoreRegistry`): one store per (arm, episode); only the owner may read it (information parity,
  §6.5.4, enforced rather than asserted).
* **Winner** (:func:`select_winner`, §6.5.5): among evaluated candidates that pass every guard bound
  (mean + z·se <= limit, one-sided 0.025) and have dev gain LCB > 0, the largest LCB; ties by smaller total relative
  Frobenius norm, then earlier evaluation; none qualifies -> ``None`` (the base, gain 0).
"""

from __future__ import annotations

import dataclasses
import hashlib
import math

from wsp.wm import access as AC
from wsp.wm import vocabfield as VF
from wsp.wm.plan import BUNDLE_DEFINITION, BudgetExceeded

Z_GUARD = 1.959963984540054  # one-sided 0.025
SHARED_FIELD_SPEC = VF.SHARED_FIELD_SPEC

ACCESS = {
    "dev_evaluation": "real evaluation of the arm's own candidates on the full dev panel through the shared "
    "BudgetLedger (one bundle each; screens fractional; repeats 0)",
    "own_dev_results": "this arm's own dev results in this episode, from its own store (no other arm's)",
    "episode_library": "the episode's declared library subsample (primitive identities and recipes, θ0)",
    "episode_pool": "the seeded per-round candidate pool drawn from the episode subsample (§6.5.3 pool mode)",
    "r0_candidate_description": "per-candidate R0 description: recipe, layers, doses, relative norms and the "
    "shared field's a/q features at θ0 (unsupported tensors marked, never 0)",
    "field": "the shared base field at θ0 (support K-FAC + vocabulary rows), R0",
    "wm_predictor": "the WM-E ensemble trained on the bank (R0 inference only)",
    "statnn_predictor": "the matched static comparator trained on the bank (R0 inference only)",
    "support_penalty": "the shared support penalty pen(c) (function of the candidate alone)",
    "bank": "trajectory-bank states and labels (training exposure only, never read during search)",
    "construction_e2": "construction/e2 shard items (training data or contrast estimation, θ0)",
    "steer_contrasts_all_layers": "construction/e2 contrast directions at every layer (θ0)",
}
FORBIDDEN = {
    "other_arm_results": "another arm's evaluations or store (§6.5.4)",
    "sealed_test": "sealed test or generation items before every winner is frozen",
    "generation_during_search": "generation is never part of search for any arm (§7.1)",
    "r2_tap_observations": "no R2 observation enters prediction during search (§6.5.4)",
    "validation_labels": "bank/validation labels are training data, not search feedback",
}
EXPOSURE = {
    "bank_trained": "trained on the trajectory bank (validation items, bank states)",
    "zero_shot": "zero-shot: no bank, no predictor, no construction data, no fine-tuning",
    "construction_e2_sft": "trained directly on construction/e2 items (the library's shard)",
    "none": "no training; direct evaluation only",
    "field_only": "no training; reads the θ0 field only",
}


@dataclasses.dataclass(frozen=True)
class ArmSpec:
    name: str
    role: str  # primary | secondary
    group: str  # P (primary) | R | D | C
    action_space: str
    mode: str  # planner | pool | greedy | direct | checkpoints
    access: tuple
    exposure: str
    uses_bank: bool = False
    llm: bool = False
    token_cap: str | None = None
    charged: tuple = ("eval_gpu_s",)
    roles: dict | None = None
    notes: str = ""
    bundle_definition: str = BUNDLE_DEFINITION
    field_spec: str = SHARED_FIELD_SPEC

    def __post_init__(self):
        bad = set(self.access) - set(ACCESS)
        if bad:
            raise ValueError(f"{self.name}: undeclared access {sorted(bad)}")
        if set(self.access) & set(FORBIDDEN):
            raise ValueError(f"{self.name}: forbidden access")
        if self.exposure not in EXPOSURE:
            raise ValueError(f"{self.name}: unknown exposure {self.exposure}")


_POOL = ("dev_evaluation", "own_dev_results", "episode_library", "episode_pool")
_READER = (*_POOL, "r0_candidate_description")
_PLANNER = ("dev_evaluation", "own_dev_results", "episode_library", "field", "support_penalty")
READER_TOKEN_CAP = "[SIM: reader_token_cap]"
FUGU_ROLES = {
    "thinker": ["episode_pool", "own_dev_results"],
    "worker": ["episode_pool", "thinker_output", "verifier_critique"],
    "verifier": ["episode_pool", "own_dev_results", "worker_ranking"],
}

ARM_SPECS = {
    s.name: s
    for s in (
        ArmSpec(
            "wm_e",
            "primary",
            "P",
            "episode_subsample",
            "planner",
            (*_PLANNER, "wm_predictor", "bank"),
            "bank_trained",
            uses_bank=True,
            charged=("eval_gpu_s", "planner_cpu_s", "predictor_cpu_s"),
        ),
        ArmSpec(
            "statnn",
            "primary",
            "P",
            "episode_subsample",
            "planner",
            (*_PLANNER, "statnn_predictor", "bank"),
            "bank_trained",
            uses_bank=True,
            charged=("eval_gpu_s", "planner_cpu_s", "predictor_cpu_s"),
        ),
        ArmSpec("random", "secondary", "R", "episode_subsample", "pool", _POOL, "none"),
        ArmSpec(
            "field_ranked",
            "secondary",
            "R",
            "episode_subsample",
            "planner",
            _PLANNER,
            "field_only",
            charged=("eval_gpu_s", "planner_cpu_s"),
            notes="FieldOnly scorer, LCB-free search mode",
        ),
        ArmSpec(
            "scripted_reader",
            "secondary",
            "R",
            "episode_subsample",
            "pool",
            _READER,
            "zero_shot",
            notes="deterministic rule over the same R0 description as the LLM readers",
        ),
        ArmSpec(
            "wsr_reader",
            "secondary",
            "R",
            "episode_subsample",
            "pool",
            _READER,
            "zero_shot",
            llm=True,
            token_cap=READER_TOKEN_CAP,
            notes="single zero-shot LLM reader; truncation counts as failure",
        ),
        ArmSpec(
            "fugu_team",
            "secondary",
            "R",
            "episode_subsample",
            "pool",
            _READER,
            "zero_shot",
            llm=True,
            token_cap=READER_TOKEN_CAP,
            roles=FUGU_ROLES,
            notes="Thinker/Worker/Verifier with access lists (TRINITY/Conductor-style fixed workflow). Not Fugu "
            "and not a replication of Fugu: no learned orchestration, no evolution, zero-shot roles.",
        ),
        ArmSpec(
            "wm_greedy",
            "secondary",
            "R",
            "episode_subsample",
            "greedy",
            (*_PLANNER, "wm_predictor", "bank"),
            "bank_trained",
            uses_bank=True,
            charged=("eval_gpu_s", "planner_cpu_s", "predictor_cpu_s"),
            notes="WM-E scorer, one-step greedy extension of own incumbents: isolates planning",
        ),
        ArmSpec(
            "direct_bon_two_stage",
            "secondary",
            "D",
            "episode_subsample",
            "direct",
            _POOL,
            "none",
            notes="no predictor; stage-1 screen on the first 4 dev groups of every objective (fractional)",
        ),
        ArmSpec(
            "bon_steering",
            "secondary",
            "C",
            "steer_all_layers",
            "direct",
            ("dev_evaluation", "own_dev_results", "steer_contrasts_all_layers"),
            "none",
            notes="single rank-one steering actions at all layers from the construction/e2 contrasts; B direct "
            "evaluations",
        ),
        ArmSpec(
            "lora_sft",
            "secondary",
            "C",
            "lora_sft_construction_e2",
            "checkpoints",
            ("dev_evaluation", "own_dev_results", "construction_e2"),
            "construction_e2_sft",
            charged=("eval_gpu_s", "train_gpu_s"),
            notes="rank-8 LoRA on construction/e2; <= min(B, 32) checkpoint evaluations (1 bundle each); "
            "training compute charged",
        ),
    )
}


def access_table() -> dict:
    """Condition-10 disclosure table: per arm, its access, exposure and accounting."""
    out = {}
    for n, s in ARM_SPECS.items():
        out[n] = {
            "role": s.role,
            "group": s.group,
            "action_space": s.action_space,
            "mode": s.mode,
            "access": list(s.access),
            "exposure": s.exposure,
            "exposure_text": EXPOSURE[s.exposure],
            "uses_bank": s.uses_bank,
            "llm": s.llm,
            "token_cap": s.token_cap,
            "charged": list(s.charged),
            "field_spec": s.field_spec,
            "notes": s.notes,
        }
        if s.roles:
            out[n]["roles"] = {r: list(a) for r, a in s.roles.items()}
    return out


class AccessDenied(PermissionError):
    """An arm asked for a resource outside its declared access (or a forbidden one)."""


class StoreRegistry:
    """One store per (arm, episode); only its owner may read it."""

    def __init__(self):
        self._s: dict[tuple, dict] = {}

    def _store(self, arm, episode):
        return self._s.setdefault((arm, episode), {"records": [], "by_theta": {}})

    def read(self, arm, episode, requester):
        if requester != arm:
            raise AccessDenied(f"{requester} may not read {arm}'s store (information parity)")
        return list(self._store(arm, episode)["records"])


class ArmContext:
    def __init__(
        self, spec: ArmSpec, ledger, episode, evaluate, theta_of, resources, stores, guard
    ):
        if spec.name not in ledger.arms:
            raise KeyError(f"{spec.name} is not an arm of this ledger")
        if not isinstance(guard, AC.R0Guard):
            raise TypeError("an R0 guard is required")
        self.spec, self.ledger, self.episode = spec, ledger, episode
        self._evaluate, self._theta_of = evaluate, theta_of
        self._resources, self._stores, self.guard = dict(resources), stores, guard
        self.reads: list[str] = []

    @property
    def arm(self):
        return self.spec.name

    def get(self, name):
        if name in FORBIDDEN or name not in self.spec.access:
            raise AccessDenied(f"{self.arm}: {name!r} is not in its declared access")
        value = self._resources[name]
        self.guard.check(value)
        self.reads.append(name)
        return value

    def history(self):
        if "own_dev_results" not in self.spec.access:
            raise AccessDenied(f"{self.arm}: own_dev_results not declared")
        return self._stores.read(self.arm, self.episode, requester=self.arm)

    def evaluate(self, candidate, kind: str = "evaluated", seconds_key="seconds"):
        """One real dev evaluation (one bundle of ``kind``) charged to this arm, or its cached repeat at 0 units."""
        if "dev_evaluation" not in self.spec.access:
            raise AccessDenied(f"{self.arm}: dev_evaluation not declared")
        th = self._theta_of(candidate)
        store = self._stores._store(self.arm, self.episode)
        if th in store["by_theta"]:
            self.ledger.charge(self.arm, repr(candidate), 0.0, kind=kind, theta_hash=th)
            return store["by_theta"][th]
        if self.ledger.remaining(self.arm) < 1 - 1e-12:
            raise BudgetExceeded(f"{self.arm}: budget {self.ledger.budget} exhausted")
        obs = dict(self._evaluate(candidate))
        self.ledger.charge(
            self.arm, repr(candidate), float(obs.get(seconds_key, 0.0)), kind=kind, theta_hash=th
        )
        rec = dict(obs, candidate=candidate, theta_hash=th, order=len(store["records"]), kind=kind)
        rec["lcb"] = rec["gain"] - Z_GUARD * rec["se"]
        rec["guard_ok"] = guard_ok(rec)
        store["records"].append(rec)
        store["by_theta"][th] = rec
        return rec

    def close(self, reason: str):
        self.ledger.close(self.arm, reason)


def guard_ok(rec, z: float = Z_GUARD) -> bool:
    return all(m + z * s <= lim for m, s, lim in rec["guards"])


def select_winner(records, z: float = Z_GUARD):
    """§6.5.5: feasible (every guard bound, LCB > 0) maximum LCB; ties smaller rel. norm, then earlier; else None."""
    feas = []
    for r in records:
        lcb = r["gain"] - z * r["se"]
        if guard_ok(r, z) and lcb > 0 and math.isfinite(lcb):
            feas.append((-lcb, r.get("rel_norm", math.inf), r["order"], r))
    if not feas:
        return None
    feas.sort(key=lambda t: t[:3])
    return feas[0][3]


ARM_SEED_RULE = "int(sha256('wme-e2-arm:{arm}:{episode_seed}')[:8], 16)"


def arm_seed(arm: str, episode_seed: int) -> int:
    """Per-(arm, episode) seed for an arm's own randomness (steering draw, LoRA init/order): fixed before search."""
    if arm not in ARM_SPECS:
        raise KeyError(arm)
    return int(hashlib.sha256(f"wme-e2-arm:{arm}:{int(episode_seed)}".encode()).hexdigest()[:8], 16)
