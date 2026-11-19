"""The wme-c1 results bundle: what the controller writes from ELIGIBLE unit payloads only (engineering contract
between the drivers and ``wsp.wm.c1.analysis``). Every file is JSON (``.json``) or gzip JSON lines (``.jsonl.gz``)
under one bundle directory; ``bundle.json`` lists every file with its sha256.

OBJ6 = wikitext_nll, gsm8k_gold_lp, mmlupro_gold_lp, refusal_margin_harmful, refusal_margin_harmless, sycophancy_real
(the trajectory objectives, R order). Dev/test also carry mathcomp_gold_lp. ``values[obj]`` are per-item NATIVE
values in the base unit's item order (``panel2.evaluate``): wikitext = mean NLL/token of the window; gsm/math = sum
log P(gold); mmlu = normalised log P(gold letter); refusal = logsumexp(refusal) - logsumexp(compliance);
sycophancy_real = log P(correct|neutral) - log P(correct|suggested), with ``syc_neutral`` and ``syc_suggested`` the
two parts. pp-equivalent values are derived in analysis by ``wm.panel2.pp_value`` (syc_neutral: exp(neutral)).

bank.json
  {"items": {"ids": {obj: [id]}, "groups": {obj: [group]}},
   "base": {"values": {...}, "sketch": {obj: [128]}, "theta_hash"},
   "graph": {"components": [{"id", "stratum", "roots"}], "calibration": [component id], "folds": {...}},
   "states": [{"state_id", "kind": "ladder"|"composition"|"control"|"inert", "component", "family",
               "roots": [root id], "side": "pos"|"neg"|null, "h", "dose", "prefix": [root id]|null,
               "theta_hash", "values": {...}, "sketch": {...}, "R": [16], "R_flags": [obj], "dnll", "endpoint",
               "history": [state_id] (the visited states of its side/order, h = 1..h)}],
   "inert": [{"spec", "states": [...], "endpoint_theta_is_base", "endpoint_outputs_bitwise_base"}],
   "fallbacks": [unit label]}

fits.json
  {"mode": "lcb"|"ranks", "coverage": {"wm": {"coverage", "n", "pass"}, "static": {...}},
   "q_hat": {"wm", "static"}, "shrink": {"wm", "static"}, "c2i": {"wm": [{"spec", "pred": [6], "sigma": [6]}],
   "static": [...]}, "f_k": [6], "components": [component id], "digests": {...}}

e2.json
  {"B": 48, "episodes": [{"index", "family", "rep", "seeds", "subsample": [pid]}],
   "arms": {arm: {str(episode index): {"winner": {"candidate": [pid], "theta_hash", "dev": <dev record>} | null,
                                       "fallback": bool, "usage": {"used", "forfeited"}, "tokens": int|null,
                                       "evaluated": [<dev record>], "failures": int|null}}},
   "sbon": {"B_par", "inputs": {"bank_gpu_s", "field_gpu_s", "n_ep", "t_bundle_s"}, "K", "P"}}
  <dev record> = {"candidate": [pid], "theta_hash", "gain", "lcb", "guards_pass", "guard_bounds":
                  {guard: {"pp"|"native": {"lcb", "ucb"}}}, "rel_norm", "seconds", "order"}
  arms: wm_e, statnn (primaries); random, field_ranked, scripted_reader, wsr_reader, fugu_team, wm_greedy,
        direct_bon_two_stage, bon_steering, lora_sft (Holm secondaries); sbon (S-BoN).

winners.json   the frozen winners manifest (``wm.e2.winners_manifest``) and its sha256; "e1": the designation.

sealed.json   (written only after the winners manifest is frozen)
  {"ids": {obj: [id]}, "groups": {obj: [group]}, "base": {"values": {...}},
   "states": {theta_hash: {"values": {...}, "candidate": [pid]}}}

gen_e1.json
  {"items": "gen_e1_items.jsonl.gz" ({"id", "group", "aliases": [str], "incorrect": str} per TriviaQA group),
   "outputs": "gen_e1_outputs.jsonl.gz" ({"model": "base"|"winner", "prompt": "neutral"|"suggested",
              "decoding": "greedy"|"sampled:<seed>", "id", "ids": [int], "text", "hit_max"}),
   "winner": {"candidate": [pid], "theta_hash"}, "config": {...}}

ste.json
  {"margins": {"guard_harmless", "guard_harmful"}, "null_pass_prob": {...}, "calibration": {...},
   "outputs": "ste_outputs.jsonl.gz" ({"benchmark": "mmlupro_gen"|"supergpqa_gen", "model": "base"|"edit",
              "budget": 2048|4096, "id", "correct", "trunc", "len", "flex_correct"}),
   "guards": "ste_guards.jsonl.gz" ({"panel": "guard_harmless"|"guard_harmful", "model": "base"|"edit"|"e1_winner",
              "id", "refused"}), "n": {...}, "canary": {...}}

ledger.json
  {"gpu_h": {phase: x}, "instance_gpu_h", "charged_gpu_h", "usd", "instances": [{"id", "type", "gpus",
   "launched_at", "terminated_at", "terminated_confirmed"}], "relaunches", "repairs", "amendments": [...]}
"""

OBJ6 = (
    "wikitext_nll",
    "gsm8k_gold_lp",
    "mmlupro_gold_lp",
    "refusal_margin_harmful",
    "refusal_margin_harmless",
    "sycophancy_real",
)
ARMS_PRIMARY = ("wm_e", "statnn")
ARMS_SECONDARY = (
    "random",
    "field_ranked",
    "scripted_reader",
    "wsr_reader",
    "fugu_team",
    "wm_greedy",
    "direct_bon_two_stage",
    "bon_steering",
    "lora_sft",
)
SBON = "sbon"
FILES = (
    "bank.json",
    "fits.json",
    "e2.json",
    "winners.json",
    "sealed.json",
    "gen_e1.json",
    "ste.json",
    "ledger.json",
)
