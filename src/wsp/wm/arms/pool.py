"""Pool mode for the reader arms and random (PREREG_WME §6.5.3-§6.5.5), with R0 candidate descriptions.

* **Rounds.** B bundles split into :data:`ROUNDS` = 4 rounds of b = B/4 real dev evaluations.
* **Pool** (:func:`draw_pool`). Each round draws ``pool_size`` (``[SIM: pool_size]``) distinct candidates: uniform
  sequences of 1-4 distinct primitives from the episode subsample (a stream seeded by (episode seed, round) only,
  so it is shared by every pool arm), plus about :data:`MUTATION_FRACTION` of mutations (add, drop or swap one
  primitive) of the arm's own top feasible incumbents (by dev LCB; at most :data:`N_INCUMBENTS`). Candidates the arm
  already evaluated are excluded. The pool is presented in a seeded shuffled order with ids ``c00``, ``c01``, ...
* **Description** (:func:`describe`, :func:`render_view`): the per-candidate R0 description every reader reads:
  per step the primitive's family, layer, contrast, dose parameter and relative norm; the summed relative norm; the
  shared field's a (signed first-order) and q (curvature) per objective at θ0 - ``None`` is shown as "unsupported",
  never 0. No outcome appears in a description.
* **Ranking.** The arm's ranker returns ids; the top b distinct valid ids are evaluated for real through the arm's
  context (shared ledger, own store). **Failure rule** (:data:`FAILURE_RULE`): a failed ranking (truncation, token
  cap, unparseable or invalid output) evaluates nothing that round; its b bundles are forfeited (the arm closes at
  used + forfeited = B); there is no retry, repair or fallback ranking.
* **Winner** by ``spec.select_winner`` (§6.5.5) over the arm's own evaluations; ``None`` is the base.
"""

from __future__ import annotations

import dataclasses
import hashlib

import numpy as np

from wsp.wm.arms import spec as S

ROUNDS = 4
MAX_LEN = 4
MUTATION_FRACTION = 0.5
N_INCUMBENTS = 4
FAILURE_RULE = (
    "A truncated, over-token-cap, unparseable or invalid ranking is a failed round: no candidate from it is "
    "evaluated, the round's b bundles are forfeited (closed at used + forfeited = B), and there is no retry, repair "
    "or fallback ranking."
)


@dataclasses.dataclass
class RankResult:
    order: list | None
    failure: str | None
    usage: list


def round_seed(episode_seed: int, round_index: int) -> int:
    h = hashlib.sha256(f"wme-e2-pool:{int(episode_seed)}:{int(round_index)}".encode()).hexdigest()
    return int(h[:8], 16)


def _mutations(c, sub):
    c = tuple(c)
    out = []
    others = [p for p in sub if p not in c]
    if len(c) < MAX_LEN:
        out += [("add", c + (p,)) for p in others]
    if len(c) > 1:
        out += [("drop", c[:i] + c[i + 1 :]) for i in range(len(c))]
    out += [("swap", c[:i] + (p,) + c[i + 1 :]) for i in range(len(c)) for p in others]
    return out


def is_mutation(c, inc) -> bool:
    return any(m == tuple(c) for _, m in _mutations(inc, sorted(set(c) | set(inc))))


def draw_pool(sub, pool_size: int, seed: int, incumbents, exclude=()) -> list:
    sub = list(sub)
    seen = {tuple(x) for x in exclude}
    uni, mut = [], []
    rng = np.random.default_rng(seed)
    mrng = np.random.default_rng([seed, 1])
    n_mut = round(pool_size * MUTATION_FRACTION) if incumbents else 0
    tries = 0
    while len(mut) < n_mut and tries < 50 * max(1, n_mut):
        tries += 1
        inc = tuple(incumbents[int(mrng.integers(len(incumbents)))])
        ms = _mutations(inc, sub)
        _, c = ms[int(mrng.integers(len(ms)))]
        if c not in seen:
            seen.add(c)
            mut.append(c)
    tries = 0
    while len(uni) + len(mut) < pool_size and tries < 200 * pool_size:
        tries += 1
        k = int(rng.integers(1, MAX_LEN + 1))
        c = tuple(int(x) for x in rng.choice(sub, size=min(k, len(sub)), replace=False))
        if c not in seen:
            seen.add(c)
            uni.append(c)
    pool = uni + mut
    order = np.random.default_rng([seed, 2]).permutation(len(pool))
    return [pool[i] for i in order]


def describe(c, cid: str, meta: dict, features) -> dict:
    steps = [dict(meta[p], pid=p) for p in c]
    return {
        "id": cid,
        "steps": steps,
        "rel_norm_total": float(sum(s["rel_norm"] for s in steps)),
        "field": features(c) if features else {},
    }


def _num(x, fmt):
    return "unsupported" if x is None else format(x, fmt)


def render_view(v: dict) -> str:
    steps = "; ".join(
        f"{s['family']} L{s['layer']} {s['contrast']} {s['param']:+g} (relF {s['rel_norm']:.1e})"
        for s in v["steps"]
    )
    field = "; ".join(
        f"{o} a={_num(f['a'], '+.4g')} q={_num(f['q'], '.3g')}"
        for o, f in sorted(v["field"].items())
    )
    return f"{v['id']} | {len(v['steps'])} step(s): [{steps}] | relF total {v['rel_norm_total']:.2e} | field: {field}"


class RandomRanker:
    """The random arm: the seeded shuffled pool order (uniform over the pool)."""

    name = "random"

    def rank(self, views, history, b, round_index):
        return RankResult([v["id"] for v in views], None, [])


def _valid(order, ids, b) -> bool:
    if not isinstance(order, list) or not all(isinstance(x, str) for x in order):
        return False
    return len(set(order)) == len(order) and set(order) <= set(ids) and len(order) >= b


def run_pool_episode(ranker, ctx, B: int, pool_size: int, seed: int, features=None, rounds=ROUNDS):
    if B % rounds:
        raise ValueError(f"B = {B} does not split into {rounds} rounds")
    b = B // rounds
    sub = ctx.get("episode_library")
    ctx.get("episode_pool")
    meta = None
    if "r0_candidate_description" in ctx.spec.access:
        meta = ctx.get("r0_candidate_description").payload
    log, failures = [], 0
    for r in range(rounds):
        hist = ctx.history()
        feas = sorted((h for h in hist if h["guard_ok"] and h["lcb"] > 0), key=lambda h: -h["lcb"])
        inc = [h["candidate"] for h in feas[:N_INCUMBENTS]]
        pool = draw_pool(
            sub, pool_size, round_seed(seed, r), inc, exclude=[h["candidate"] for h in hist]
        )
        ids = [f"c{i:02d}" for i in range(len(pool))]
        views = [
            describe(c, i, meta, features) if meta is not None else {"id": i}
            for c, i in zip(pool, ids, strict=True)
        ]
        if meta is not None:
            hist = [dict(h, view=describe(h["candidate"], "", meta, features)) for h in hist]
        res = ranker.rank(views, hist, b, r)
        failure = res.failure
        if failure is None and not _valid(res.order, ids, b):
            failure = "invalid_ranking"
        rec = {
            "round": r,
            "pool_size": len(pool),
            "pool_hash": hashlib.sha256(repr(pool).encode()).hexdigest(),
            "failure": failure,
            "usage": res.usage,
            "n_evaluated": 0,
        }
        if failure is None:
            index = dict(zip(ids, pool, strict=True))
            for cid in res.order[:b]:
                ctx.evaluate(index[cid])
                rec["n_evaluated"] += 1
        else:
            failures += 1
        log.append(rec)
    ctx.close(f"{rounds} rounds; {failures} failed round(s) forfeited")
    hist = ctx.history()
    return {
        "winner": S.select_winner(hist),
        "rounds": log,
        "failures": failures,
        "evaluated": [h["candidate"] for h in hist],
    }
