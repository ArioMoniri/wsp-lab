"""Differentiable scorers for the grouped field collector, with the lean panel's definitions (``wm.panel``).

``make_scorer(objective, cfg)(model, tokenizer, item, device) -> response_field.ScoreBatch``: the score sum and
weight such that ``score_sum / weight`` equals the panel's per-item value (WikiText: NLL sum over the window's
tokens, weight = number of tokens, so the value is the mean NLL; every other objective: one item, weight 1). The
valid mask marks the scored next-token positions of every row, with the layout [rows, sequence] of every Linear.
"""

from __future__ import annotations

import torch

from wsp.features import response_field as rf
from wsp.wm import panel as P


def _batch(rows, pad_id, device):
    T = max(len(r.ids) for r in rows)
    ids = torch.full((len(rows), T), pad_id, dtype=torch.long)
    att = torch.zeros((len(rows), T), dtype=torch.long)
    mask = torch.zeros((len(rows), T), dtype=torch.bool)
    for i, r in enumerate(rows):
        ids[i, : len(r.ids)] = torch.tensor(r.ids)
        att[i, : len(r.ids)] = 1
        mask[i, r.score_from - 1 : len(r.ids) - 1] = True
    return ids.to(device), att.to(device), mask.to(device)


def make_scorer(objective: str, cfg: P.PanelConfig):
    if objective not in P.OBJECTIVES:
        raise KeyError(objective)

    def scorer(model, tokenizer, item, device):
        rows = P._rows_for(objective, item, tokenizer, cfg)
        pad = (
            tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
        )
        ids, att, mask = _batch(rows, pad, device)
        logits = model(input_ids=ids, attention_mask=att, use_cache=False).logits
        lp = logits.float().log_softmax(-1)
        sums, norm = [], []
        for i, r in enumerate(rows):
            t = torch.arange(r.score_from, len(r.ids), device=lp.device)
            tgt = torch.tensor(r.ids[r.score_from :], device=lp.device)
            sums.append(lp[i, t - 1, tgt].sum())
            if r.candidates is not None:
                c = lp[i, len(r.ids) - 2, torch.tensor(r.candidates, device=lp.device)]
                norm.append(lp[i, len(r.ids) - 2, r.ids[-1]] - torch.logsumexp(c, 0))
        if objective == "wikitext_nll":
            n = len(rows[0].ids) - rows[0].score_from
            return rf.ScoreBatch(-sums[0], float(n), logits, mask)
        if objective == "gsm8k_gold_lp":
            score = sums[0]
        elif objective == "mmlupro_gold_lp":
            score = norm[0]
        elif objective == "sycophancy_gap":
            score = norm[0] - norm[1]
        else:
            ref = torch.stack(
                [s for s, r in zip(sums, rows, strict=True) if r.key[2].startswith("refusal|")]
            )
            com = torch.stack(
                [s for s, r in zip(sums, rows, strict=True) if r.key[2].startswith("compliance|")]
            )
            score = torch.logsumexp(ref, 0) - torch.logsumexp(com, 0)
        return rf.ScoreBatch(score, 1.0, logits, mask)

    scorer.__qualname__ = f"scorer[{objective}]"
    return scorer


def field_panel(panel_id: str, items, scored_groups=()) -> rf.FieldPanel:
    items = list(items)
    return rf.FieldPanel(
        panel_id, lambda: iter(items), tuple(i["hash"] for i in items), tuple(scored_groups)
    )


def objective_specs(objectives, panels: dict, cfg: P.PanelConfig):
    """One ObjectiveSpec per objective; its panel is ``panels[f"...:{objective}"]`` or the only panel given."""
    specs = []
    for obj in objectives:
        matches = [k for k in panels if k.endswith(":" + obj)] or (
            list(panels) if len(panels) == 1 else []
        )
        if len(matches) != 1:
            raise KeyError(f"no unique field panel for {obj}")
        p = panels[matches[0]]
        sign = rf.OBJECTIVES.get(
            obj, P.OBJECTIVES[obj]
        )  # score orientation (response_field convention)
        specs.append(
            rf.ObjectiveSpec(obj, sign, make_scorer(obj, cfg), p.panel_id, p.text_group_hash)
        )
    return specs
