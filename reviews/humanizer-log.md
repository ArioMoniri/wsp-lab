# Humanizer log: abstract, introduction, related work

Base: a74ed08 (branch paper/intro-related). Guide: docs/WRITING_GUIDE.md.

Checks run after editing:
- `source scripts/env.sh && python scripts/check_cites.py`: passed (33 verified keys, 9 section files).
- Every `\cite` key, every number, every inline math expression and every section heading was compared with
  the a74ed08 versions. All are unchanged. The only token difference is trailing punctuation after `H1` and `H3`.
- Stock-phrase grep (delve, leverage, notably, importantly, interestingly, in this work, testament, landscape,
  robust, realm, worth noting): no matches.
- `paper/main.pdf` rebuilt with latexmk: 4 pages, no errors.

All 12 text paragraphs were revised. Each line gives the first words before → after, then the reason.

## abstract.tex

1. **Abstract, paragraph 1.** Opening "Can a model's weight changes" unchanged. Edited sentences: "Recent work by Takehi et al. shows that…" → "reports that…"; "We will test whether weight-only features…" → "Concretely, it asks whether weight-only features…", with the four questions kept as one clause; "Its scope is deliberately limited to…" → "The scope stays with prediction and auditing: we are not studying…". Reason: removes the "We will test whether … whether … whether … and how" run-on and the "deliberately" filler; the four questions are unchanged.

## introduction.tex

2. **Introduction, paragraph 1.** "Can a model's weight changes tell us" (opening unchanged). "This question matters when evaluating…" → "The question matters once evaluating…"; "Weight-space methods offer a possible alternative: instead of…" → "Weight-space methods offer one alternative. Rather than executing…"; "They could help identify which edits…" → "They could point to which edits…". Reason: breaks a colon-joined sentence and varies lengths; "possible" removed as hedge filler. Claim unchanged.

3. **Introduction, paragraph 2.** "The immediate motivation is the report by Takehi et al." → "Our starting point is the report by Takehi et al."; "This is a striking instance of…" → "The result is a weight edit with a large measured behavioral effect."; "We ask a narrower, testable question:" → "We ask a narrower question that can be tested." Reason: drops the evaluative "striking" and the colon; the Takehi numbers and the "not evidence that" limitation are unchanged.

4. **Introduction, paragraph 3 (H1–H4).** "Our study is organized around four hypotheses. H1 asks…" → "…The first, H1, asks…"; "H2 asks whether modeling edit order…" → "H2 concerns edit order: does modeling it…"; "H3 tests transfer across edit families:" → "Transfer across edit families is the subject of H3."; "H4 compares an LLM asked to…" → "H4 pits an LLM that reads…"; "is more robust to novel families" → "holds up better on novel families". Reason: breaks the "H_n asks/tests/compares" parallel and removes "robust" as filler. Hypothesis content, the 0.6, 15%, 0.5 and four-edit thresholds, and the "planned decision criteria, not results" sentence are unchanged.

5. **Introduction, paragraph 4 (alignment sub-study).** "An alignment-edit sub-study will make the prediction problem concrete…" → "The alignment-edit sub-study puts the prediction problem on concrete ground…"; "We will then test whether…" → "We will also test whether…", with "name the edit family" in place of "identify which edit family". Reason: varies openings; the model list is reordered but unchanged in content. "The measures are complementary" kept as written.

6. **Introduction, paragraph 5 (contributions and limitation).** "The planned work has three contributions. First, we will construct…" → "The work has three planned parts. The first builds…"; "Second, we will compare…" → "The second compares…"; "This directly tests whether…" → "That comparison is the direct test of whether…"; "The central limitation is that this is a study of a planned edit zoo" → "The main limitation is specific. This is a study of a planned edit zoo…". Reason: removes the First/Second/Third enumeration written as prose and states the limitation in specific terms, as the guide asks.

## related.tex

7. **Related, "Learning from model weights."** "Weight-space learning spans methods for understanding weight geometry…" → "Weight-space learning covers work on weight geometry…"; "Our question concerns a narrower target:" → "Our question is a narrower one:"; "Those results concern LoRA performance…" → "Those numbers concern…"; "they do not establish performance for merged edits…" → "We cannot read off their performance on merged edits…"; "We therefore use canonicalized…" → "We therefore treat…". Reason: removes the stock "spans" construction and the verb "establish" repeated across paragraphs. The W2T numbers (MAE 0.32, Spearman 95.35%), the cited claims, and the scope limits are unchanged.

8. **Related, "Several adjacent lines…"** (paragraph 2). "Several adjacent lines predict model performance with information obtained by executing the model." → "Several neighboring lines predict performance by running the model first."; "Delta Activations represents fine-tuned models by measuring changes in…" → "…by measuring how their internal activations change…"; "Fast probing uses…" → "Fast probing reads…"; "reducing evaluation latency…" → "cuts evaluation latency…"; "TuneAhead combines…" → "TuneAhead pairs…"; "Our primary prediction setting uses…" → "Our main setting uses…"; "This distinction makes any accuracy comparison conditional on…" → "Any accuracy comparison therefore has to account for…". Reason: varies sentence openings and shortens the Fast probing sentence. The AUROC 0.75, the hour-to-three-minutes figure, and the 1.47 pp RMSE over 370 runs are unchanged.

9. **Related, "Edit trajectories and predictive representations."** (paragraph 3). "Weight changes may carry information about an edit's effect, but the effect of a later edit need not be…" → split into two sentences; "Separately, a study across CIFAR…" → "A separate study across CIFAR…"; "This motivates recording…, while cautioning against…" → "…and it cautions against…"; "It is a proposal for prediction with verification, not evidence that…" → "That is a proposal about prediction with verification. It is not evidence that…". Reason: removes the "not X, but Y" contrast and varies openings. Claims unchanged, and "training-format serialization" is kept as the source's own term.

10. **Related, paragraph 4 (sequence models and JEPA).** "We test sequence models because each observed edit arrives in context: an edit can alter…" → "…in context. An edit can change…"; "Mamba-3 combines…" → "Mamba-3 pairs…"; "That architectural description does not establish an advantage…" → "That description is architectural, and it does not show an advantage…"; "In the JEPAWG work on physical…, the main-text configuration omits…" → "The JEPAWG work on physical parameterized systems omits…"; the ResidualPred sentence is split in two; "These findings inform design choices, but neither result establishes…" → "Those findings inform our design choices, but neither one shows…"; "…also have levels: exact sequence-level equivalence requires additional stated conditions, rather than following from…" → "…hold at different levels: exact sequence-level equivalence needs additional stated conditions, and standard JEPA training alone does not supply them". Reason: removes the colon-chained and "also have levels" constructions and varies sentence length. The ResidualPred and JEPAWG findings, and the "does not establish transfer" limit, are unchanged.

11. **Related, "Alignment edits and evaluation limits."** (paragraph 5). "Refusal-related weight edits provide a useful stress test because…" → "…make a useful stress test, because refusal can change while task capability changes too."; "Heretic combines… and Arbitrary-Rank Ablation is its default modifier" → "…and its default modifier is Arbitrary-Rank Ablation"; "We plan to compare this family with classic directional ablation, while measuring capability…" → split, with "Capability goes into the measurements; we do not assume it survives."; "that score is not itself a measure of harmfulness" → "the score alone does not measure harmfulness". Reason: splits a long sentence and varies openings. "33 case-insensitive strings", "first 100 response tokens", and the motivation to report refusal measures alongside capability are unchanged.

12. **Related, final paragraph.** "Prior work also illustrates why…" (opening kept). "Surgical Refusal Ablation orthogonalizes…, and reports low refusal rates together with…" → split, "Across five models it reports low refusal rates alongside…"; "The latter is evidence about those tested edits, not a general claim…" → "That is evidence about the edits that were tested. It does not show that…"; "Our alignment sub-study will treat these outcomes as reasons…, not as expected results for Qwen." → "…treats outcomes like these as reasons to measure several endpoints, and we do not take them as predicted results for Qwen."; "our contribution is therefore an empirical test…" → "our contribution is an empirical test…"; the closing sentence is reworded to "We keep the boundaries of the source claims in view…". Reason: varies openings and sentence length and replaces the "not X, but Y" contrasts. The Surgical Refusal and GLM-5.3-Flash measurements and the ranks 1–12 are unchanged.

## Not changed

- Section headers (`\section`, `\paragraph`) and the `\citep` keys.
- The opening sentences of the abstract ("Can a model's weight changes predict…") and the introduction ("Can a model's weight changes tell us…") are kept word for word. Later sentences in those paragraphs were edited as described above.

## Board B revision (2026-11-04)

- Corrected the task-vector claim to match the verified excerpt's direction: interference collapses under the training-format wrapper, not under the untrained chat template. Updated `claim_we_make` to reflect the excerpt.
- Made the default TMLR build anonymous and documented the switch for an identified arXiv preprint. Revised the title to identify the work as a proposed study.
- Added the H1-H4 predictions, thresholds, and falsification criteria specified in `docs/HYPOTHESIS.md`, plus H4's two directional comparisons.
- Rewrote the three mechanically phrased sentences, used proposed or future language for planned work, split bundled claims, and shortened the abstract's research questions.
- Re-audited each citation in the abstract, introduction, and related work against its `claim_we_make` entry. `python scripts/check_cites.py` passed (33 verified keys; nine section files).
- Rebuilt the four-page PDF. Page one identifies the submission as anonymous, and the PDF Author metadata is empty.
