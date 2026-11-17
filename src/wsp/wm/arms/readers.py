"""Reader arms for pool mode (PREREG_WME §6.8.3, group R): scripted reader, WSR-style single LLM reader and the
Fugu-style reader team. All three read only the R0 candidate description of the pool and their own arm's dev
results (zero-shot: no bank, no predictor, no construction data), rank the pool, and return ids; ``wm.arms.pool``
evaluates the top b through the shared ledger.

* **Output contract** (:func:`parse_ranking`, :func:`parse_object`): the reply must be exactly one JSON object
  (one surrounding Markdown code fence is the only normalisation); a ranking is a list of at least b distinct ids
  from the pool. Nothing is repaired, defaulted or guessed: anything else is a failed round (``unparseable:<role>``).
* **ScriptedReader**: a frozen deterministic rule over the same description and own results, in key order:
  (1) candidates sharing a primitive with an own guard-breaching evaluation go last; (2) more primitives shared with
  the best own feasible incumbent first; (3) more steps whose contrast matches the primary objective first; (4)
  larger orientation-signed field a on the primary first (unsupported last); (5) smaller summed relative norm;
  (6) id.
* **WSRReader**: one zero-shot call per round (role ``reader``).
* **FuguTeam**: Thinker -> Worker -> Verifier, following TRINITY/Conductor's fixed role workflow with access lists
  (Fugu-Ultra: agents see others' outputs only through the access list). Access lists (``spec.FUGU_ROLES``) are
  enforced when each prompt is composed (:func:`compose`): the Thinker sees the pool and the team's own dev results
  and writes notes; the Worker sees the pool and the Thinker's notes (and, on revision, the Verifier's critique)
  and ranks; the Verifier sees the pool, the own dev results and the Worker's ranking, and answers ACCEPT or REVISE
  with a critique. At most one revision; the revised ranking is final. Each role has its own pluggable client
  (provider and model). **Not Fugu**: no learned orchestrator, no evolution, no training; zero-shot roles.
* **Tokens**: every call goes through ``llm.guarded_call`` on the arm's per-episode ``TokenLedger`` (the same cap,
  ``[SIM: reader_token_cap]``, for both LLM arms); truncation, provider error and cap exhaustion fail the round.
"""

from __future__ import annotations

import dataclasses
import json
import math

from wsp.wm import p2design as G2
from wsp.wm.arms import llm as L
from wsp.wm.arms import pool as P
from wsp.wm.arms import spec as S

READER_MAX_OUTPUT = 1024
ROLE_MAX_OUTPUT = {"thinker": 512, "worker": 1024, "verifier": 512}
MAX_REVISIONS = 1
PRIMARY_CONTRAST = {
    "sycophancy_real": "syc",
    "gsm8k_gold_lp": "gsm",
    "mathcomp_gold_lp": "mathcomp",
}


class ParseFailure(ValueError):
    pass


@dataclasses.dataclass(frozen=True)
class ReaderTask:
    objective: str = "sycophancy_real"
    guards: str = (
        "Every guardrail (WikiText NLL, GSM8K, MMLU-Pro and competition-math gold log-probability, both refusal "
        "margins, neutral sycophancy) must stay within its registered margin on the dev panel; an edit that "
        "breaches any guardrail earns nothing."
    )

    @property
    def orientation(self) -> float:
        return G2.ORIENT.get(self.objective, 1.0)

    def text(self) -> str:
        d = "lower" if self.orientation < 0 else "higher"
        return (
            f"Primary objective: {self.objective} ({d} native value is better; the oriented gain is positive when "
            f"it improves). {self.guards} The winner is the evaluated candidate with the largest dev lower "
            "confidence bound of oriented gain among those passing every guardrail; if none qualifies, the "
            "unedited model wins."
        )


def parse_object(text: str) -> dict:
    t = (text or "").strip()
    if t.startswith("```"):
        lines = t.splitlines()
        if len(lines) < 3 or not lines[-1].strip() == "```":
            raise ParseFailure("unterminated code fence")
        t = "\n".join(lines[1:-1]).strip()
    try:
        obj = json.loads(t)
    except json.JSONDecodeError as e:
        raise ParseFailure(f"not one JSON object: {e.msg}") from None
    if not isinstance(obj, dict):
        raise ParseFailure("not a JSON object")
    return obj


def parse_ranking(text: str, ids, b: int) -> list:
    r = parse_object(text).get("ranking")
    if not isinstance(r, list) or not all(isinstance(x, str) for x in r):
        raise ParseFailure("ranking must be a list of ids")
    if len(set(r)) != len(r):
        raise ParseFailure("duplicate ids")
    if not set(r) <= set(ids):
        raise ParseFailure("unknown ids")
    if len(r) < b:
        raise ParseFailure(f"fewer than {b} ids")
    return r


def render_history(history) -> str:
    if not history:
        return "(no evaluations yet)"
    out = []
    for h in sorted(history, key=lambda h: h["order"]):
        v = h.get("view")
        desc = (
            P.render_view(dict(v, id=f"e{h['order']:02d}"))
            if v
            else f"e{h['order']:02d} | {h['candidate']}"
        )
        out.append(
            f"{desc} -> dev gain {h['gain']:+.4f} ± {h['se']:.4f} (LCB {h['lcb']:+.4f}); "
            f"guardrails {'pass' if h['guard_ok'] else 'BREACH'}"
        )
    return "\n".join(out)


def render_pool(views) -> str:
    return "\n".join(P.render_view(v) for v in views)


# ------------------------------------------------------------------ scripted reader


class ScriptedReader:
    """Frozen rule, in key order: (1) shares a primitive with an own guard-breaching evaluation -> last; (2) more
    primitives shared with the best own feasible incumbent first; (3) more steps whose contrast matches the primary
    objective first; (4) larger orientation-signed field a on the primary first (unsupported last); (5) smaller
    summed relative norm; (6) id."""

    name = "scripted_reader"

    def __init__(self, task: ReaderTask):
        self.task = task

    def rank(self, views, history, b, round_index):
        pids = lambda v: {s["pid"] for s in v["steps"]}
        breach = set()
        for h in history:
            if not h["guard_ok"]:
                breach |= set(h["candidate"])
        feas = sorted(
            (h for h in history if h["guard_ok"] and h["lcb"] > 0),
            key=lambda h: (-h["lcb"], h["order"]),
        )
        best = set(feas[0]["candidate"]) if feas else set()
        want = PRIMARY_CONTRAST.get(self.task.objective)

        def key(v):
            a = v["field"].get(self.task.objective, {}).get("a")
            oa = -math.inf if a is None else self.task.orientation * a
            return (
                bool(pids(v) & breach),
                -len(pids(v) & best),
                -sum(s["contrast"] == want for s in v["steps"]),
                -oa,
                v["rel_norm_total"],
                v["id"],
            )

        return P.RankResult([v["id"] for v in sorted(views, key=key)], None, [])


# ------------------------------------------------------------------ LLM readers

_FORMAT_RANKING = (
    'Reply with exactly one JSON object and nothing else: {"ranking": ["<id>", ...]} listing at least {b} distinct '
    "candidate ids from the pool, most promising first."
)
WSR_SYSTEM = (
    "You are a zero-shot reader of weight edits to a language model. You are shown, for each candidate edit, its "
    "recipe (family, layer, contrast, dose parameter, relative Frobenius norm) and first-order (a) and curvature (q) "
    "features of a response field computed at the unedited weights, plus the dev results of edits you already chose "
    "to evaluate. Rank the candidate pool by how likely each edit is to improve the primary objective while keeping "
    "every guardrail. " + _FORMAT_RANKING
)
THINKER_SYSTEM = (
    "You are the Thinker in a three-role team (Thinker, Worker, Verifier) that chooses weight edits of a language "
    "model to evaluate. From the candidate pool descriptions and the team's own dev results, write short hypotheses "
    "about which kinds of edits are promising and which risk guardrails. Do not rank. Reply with exactly one JSON "
    'object and nothing else: {"notes": "<at most 120 words>"}.'
)
WORKER_SYSTEM = (
    "You are the Worker in a three-role team (Thinker, Worker, Verifier) that chooses weight edits of a language "
    "model to evaluate. Using the candidate pool descriptions and the notes you were given, rank the pool. "
    + _FORMAT_RANKING
)
VERIFIER_SYSTEM = (
    "You are the Verifier in a three-role team (Thinker, Worker, Verifier). Check the Worker's ranking against the "
    "candidate pool descriptions and the team's own dev results (guardrail risks, repeats of failed edits, ids "
    'that do not exist). Reply with exactly one JSON object and nothing else: {"verdict": "ACCEPT"} or '
    '{"verdict": "REVISE", "critique": "<at most 80 words>"}.'
)
_SECTIONS = {
    "episode_pool": "CANDIDATE POOL",
    "own_dev_results": "OWN DEV RESULTS (this team only)",
    "thinker_output": "THINKER NOTES",
    "worker_ranking": "WORKER RANKING",
    "verifier_critique": "VERIFIER CRITIQUE",
}


def compose(role: str, materials: dict, task_text: str = "", tail: str = "") -> str:
    """The user prompt of a team role from its access-listed materials only (``spec.FUGU_ROLES``)."""
    allowed = S.FUGU_ROLES[role]
    bad = [k for k in materials if k not in allowed]
    if bad:
        raise S.AccessDenied(f"{role} may not read {bad} (access list {allowed})")
    parts = [f"TASK\n{task_text}"] if task_text else []
    for k in allowed:
        if k in materials:
            title = "POOL" if k == "episode_pool" else _SECTIONS[k]
            parts.append(f"{title}\n{materials[k]}")
    if tail:
        parts.append(tail)
    return "\n\n".join(parts)


def _fail(failure, role, usage):
    return P.RankResult(None, f"{failure}:{role}", usage)


class WSRReader:
    name = "wsr_reader"

    def __init__(
        self, client, ledger: L.TokenLedger, task: ReaderTask, max_output_tokens=READER_MAX_OUTPUT
    ):
        self.client, self.ledger, self.task, self.max_out = client, ledger, task, max_output_tokens

    def rank(self, views, history, b, round_index):
        n0 = len(self.ledger.records)
        user = (
            f"TASK\n{self.task.text()}\n\nCANDIDATE POOL\n{render_pool(views)}\n\nOWN DEV RESULTS\n"
            f"{render_history(history)}\n\nReturn at least {b} ids."
        )
        c, failure = L.guarded_call(
            self.client,
            self.ledger,
            "reader",
            WSR_SYSTEM.replace("{b}", str(b)),
            user,
            self.max_out,
        )
        usage = self.ledger.records[n0:]
        if failure:
            return _fail(failure, "reader", usage)
        try:
            order = parse_ranking(c.text, [v["id"] for v in views], b)
        except ParseFailure:
            return _fail("unparseable", "reader", usage)
        return P.RankResult(order, None, usage)


class FuguTeam:
    name = "fugu_team"
    ROLES = ("thinker", "worker", "verifier")

    def __init__(
        self,
        clients: dict,
        ledger: L.TokenLedger,
        task: ReaderTask,
        max_output=None,
        max_revisions=MAX_REVISIONS,
    ):
        if set(clients) != set(self.ROLES):
            raise ValueError(f"clients for exactly {self.ROLES}")
        self.clients, self.ledger, self.task = clients, ledger, task
        self.max_out = dict(ROLE_MAX_OUTPUT, **(max_output or {}))
        self.max_revisions = int(max_revisions)

    def _call(self, role, system, user):
        return L.guarded_call(
            self.clients[role], self.ledger, role, system, user, self.max_out[role]
        )

    def rank(self, views, history, b, round_index):
        n0 = len(self.ledger.records)
        usage = lambda: self.ledger.records[n0:]
        ids = [v["id"] for v in views]
        pool_txt, hist_txt, task = render_pool(views), render_history(history), self.task.text()
        c, f = self._call(
            "thinker",
            THINKER_SYSTEM,
            compose("thinker", {"episode_pool": pool_txt, "own_dev_results": hist_txt}, task),
        )
        if f:
            return _fail(f, "thinker", usage())
        try:
            notes = parse_object(c.text).get("notes")
            if not isinstance(notes, str):
                raise ParseFailure("notes must be a string")
        except ParseFailure:
            return _fail("unparseable", "thinker", usage())
        worker_system = WORKER_SYSTEM.replace("{b}", str(b))
        wmat = {"episode_pool": pool_txt, "thinker_output": notes}
        c, f = self._call(
            "worker", worker_system, compose("worker", wmat, task, f"Return at least {b} ids.")
        )
        if f:
            return _fail(f, "worker", usage())
        try:
            ranking = parse_ranking(c.text, ids, b)
        except ParseFailure:
            return _fail("unparseable", "worker", usage())
        for _ in range(self.max_revisions):
            vmat = {
                "episode_pool": pool_txt,
                "own_dev_results": hist_txt,
                "worker_ranking": json.dumps(ranking),
            }
            c, f = self._call("verifier", VERIFIER_SYSTEM, compose("verifier", vmat, task))
            if f:
                return _fail(f, "verifier", usage())
            try:
                v = parse_object(c.text)
                verdict = v.get("verdict")
                if verdict not in ("ACCEPT", "REVISE") or (
                    verdict == "REVISE" and not isinstance(v.get("critique"), str)
                ):
                    raise ParseFailure("verdict must be ACCEPT or REVISE with a critique")
            except ParseFailure:
                return _fail("unparseable", "verifier", usage())
            if verdict == "ACCEPT":
                break
            wmat = {
                "episode_pool": pool_txt,
                "thinker_output": notes,
                "verifier_critique": v["critique"],
            }
            c, f = self._call(
                "worker", worker_system, compose("worker", wmat, task, f"Return at least {b} ids.")
            )
            if f:
                return _fail(f, "worker", usage())
            try:
                ranking = parse_ranking(c.text, ids, b)
            except ParseFailure:
                return _fail("unparseable", "worker", usage())
        return P.RankResult(ranking, None, usage())
