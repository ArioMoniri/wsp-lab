# Challenger Seat Report: Round 13 Council (20 Nov 2026)
## Weight-Space Reading, WSP, and the Anatomy of the (a) Negative Result

### BLUF
1. Neither P1 nor P2 demonstrates that calibrated, edit-specific trajectory prediction (a) is achievable.
2. P1 measures coarse ternary direction detection on N=18 changed labels, failing on inert architectures (Muse false-alarm rate 53–88%).
3. P2 is an adaptive forward-search agent whose dev gains evaporate on held-out 9B benchmarks (MATH-500: 97.70% to 97.35%).
4. The Stage W2 tie (WM E=0.881 vs Pooled E=0.881, d=0.000) is an identifiability and data limit, not model capacity.
5. In-context sequence learners (LC-PFN E=2.638) already lost decisively to the pooled curve; an SSM cannot conjure absent signal.
6. Re-scoping (a) to direction or ranking chases noise within measurement error (vLLM discordance 0.058–0.082) at high GPU cost.
7. WSR readers hallucinate phantom edits and cannot rescue (b) against direct multi-fidelity search at compute parity.
8. (a) must remain dead: conduct a zero-GPU variance decomposition on exposed W2 data, freeze 9B, and protect Study M.

---

### 1. External Evidence Audit and Citation Verification
External URLs were fetched directly via HTTP (all returned HTTP 200; zero fetch failures):
- **User Citation Error:** `https://arxiv.org/abs/2609.25541` is Jeon, Cho, and Suh, *"A JEPA Recipe for Tabular Foundation Models"* (147 tabular datasets, value vs JEPA arms), completely unrelated to weight editing or world models.
- **P1:** Kocabay, *"Weight-Space Reading"* (`https://www.alphaxiv.org/abs/2611.weight-space-reading-language-models`; code `https://github.com/suayptalha/wsr`). Sole author; Poindexter Labs.
- **P2:** Hu et al., *"Exploring Weight-Space Programming: Capabilities, Generalization, and Limitations"* (`https://arxiv.org/abs/2611.09382`, HTML `https://arxiv.org/html/2611.09382v1`). Kocabay is not an author.

---

### 2. Premise 1: P1 and P2 Do Not Show (a) Is Possible
The user's premise conflates crude direction heuristics on tiny samples with calibrated trajectory regression against statistical baselines.

#### Failure Modes of P1 (Weight-Space Reading)
1. **Direction vs. Trajectory:** P1 does not predict trajectories or rates. Sandboxed LLMs classify whether five behaviors shift $\text{up}$, $\text{down}$, or $\text{same}$.
2. **Microscopic N ($N=18$):** The headline result—27B reading own weights at 0.730 balanced accuracy vs 0.578 text-only (+0.152, 95% CI [+0.056, +0.256])—rests on **10 non-destructive changed interventions yielding 18 changed behavior labels** across single runs.
3. **Response Bias:** Weight access made readers call "change" indiscriminately: 27B change recall rose from 0.278 to 0.667, but specificity dropped from 0.878 to 0.793.
4. **Phantom Predictions:** On Muse-Glimmer-30B, **0 of 40 interventions** changed behavior, yet readers predicted changes for **53% to 88% of them** (Muse-30B on 85%), blind to post-sublayer RMSNorm cancellation.
5. **Null Self-Reading:** The crossed 2×2 interaction includes zero in every round (+0.114, 95% CI [−0.093, +0.315] subtle; +0.085, 95% CI [−0.078, +0.243] full-dose). On destructive edits, cross-family models detect damage equally well (Muse-30B 0.594 vs 27B 0.593 on 27B).
6. **Missing Pooled Baseline:** P1 compares only to text prompts and a 0.50 floor, never against a path-type statistical prior.

#### Selection Effects and Headroom Collapse in P2 (WSP)
1. **Forward Search, Not World Modeling:** P2 evaluates GPT-6 Astra executing 50 forward requests and 300 Python analyses per search. It performs adaptive hill-climbing without gradients.
2. **Validation Failure:** 6 of 16 trajectories retained original weights because candidate gains collapsed on validation (e.g., 9B GPQA Diamond gained +6.25 pp dev but ended negative).
3. **Headroom Collapse at 9B:** P2's gains occurred only on crippled 0.8B models (MATH-500: 8.35% to 37.80%). On Qwen3.5-9B, **MATH-500 held-out accuracy fell from 97.70% to 97.35% (−0.35 pp)**, despite a +29.69 pp dev gain.

---

### 3. Premise 2: Diagnosing the W2 Tie — Identifiability and Noise, Not Capacity
In Stage W2 (`docs/reviews/wme/wme-stagew2-2026-11-18.md`), the nested World Model (WM) **tied** the pooled same-path-type curve:
- **Zero-shot ($k=0$):** WM $E = 0.768$ vs Pooled $E = 0.771$ (mean $d$ on S / L / M: 0.048 / −0.034 / −0.004).
- **Few-shot ($k=2$):** WM $E = 0.881$ vs Pooled $E = 0.881$ ($d = 0.000 / 0.000 / 0.000$; run-mean $d = 0.000$ across all 15 runs).
- **Shrinkage:** Inner LOORO selected shrinkage $s = 0.0$ for **all 15 held-out runs** at $k=2$, collapsing the WM completely to the pooled curve.

The tie is an **identifiability and noise ceiling problem**:
1. **Low Within-Path Variance:** Within path types (S, L, M), learning rates and step schedules dominate. Residual edit-specific variance is negligible.
2. **Feature Bankruptcy:** R1 features (first-order Jacobians $J_1$ at $\theta_0$ on 64 items) carry no edit-specific trajectory signal beyond path type.
3. **Noise Floor:** Base re-run correctness discordance from vLLM batch non-invariance is **0.058 on MMLU-Pro and 0.082 on SuperGPQA**. Monte Carlo evaluation noise swamps true run variation.

#### Zero-GPU Diagnostic on Exposed W/W2 Data
Before any GPU spending, execute this CPU diagnostic on `data/wme/`:
1. **Variance Decomposition:** Fit a linear mixed-effects model on the 15 W2 runs: $\sigma^2_{\text{total}} = \sigma^2_{\text{path\_type}} + \sigma^2_{\text{run(path\_type)}} + \sigma^2_{\text{eval\_noise}}$.
2. **Noise Floor:** Estimate $\sigma^2_{\text{eval\_noise}}$ from observed discordances (0.058/0.082) and binomial sampling variance.
3. **Decision Rule:** If $\sigma^2_{\text{run(path\_type)}} \le \sigma^2_{\text{eval\_noise}}$, no model family can beat the pooled curve.

---

### 4. Proposal 1: The Mamba/SSM Memory Idea Is Conceptually Flawed
Adding an SSM memory layer fails on first principles:
1. **Ill-Defined Sequence:** At $k=0$, there is no evaluation sequence. At $k=2$, the "sequence" is two scalar points. An SSM cannot build meaningful state over length 2.
2. **In-Context Learners Already Lost:** Generic sequence models were **tested in Stage W2 and crushed by the pooled curve**:
   - **LC-PFN (`lcpfn`):** $E = 2.638$ at $k=2$ (S: 1.517, L: 1.264, M: 2.491).
   - **LC Ensemble (`lc_ensemble`):** $E = 3.18$ at $k=2$; **Persistence:** $E = 3.17$ at $k=2$.
   - **Nested WM and Pooled Curve:** $E = 0.881$ at $k=2$.
   Generic sequence models trail the pooled curve by **1.75 to 2.30 error units** and overfit sequence noise.
3. **Sample Starvation:** SSMs require extensive sequence pre-training. Fitting an SSM on 15 runs with 64-step paths guarantees extreme overfitting.

---

### 5. Proposal 2: Why Re-Scoping (a) Is an Unpowerable Distraction
1. **Majority Baseline Dominance:** On structured paths, direction is largely deterministic by path type ($>90\%$ accuracy). Where direction fluctuates, effect sizes fall below the 5.8–8.2% noise floor.
2. **Severe Power Deficit:** Stage W simulations (`docs/reviews/wme/wme-stagew-2026-11-18.md` §4) proved zero-shot power at $P=40$ was only **0.235 (95% CI [0.200, 0.273])**. Achieving 0.8 power for subtle within-type ranking requires $>120$ independent runs ($>80$ GPU-h, $>\$400$), yielding zero practical value.
3. **Study M Pre-Ruling:** In `SYNTHESIS-r12.md`, the council ruled (a) descriptive only for Study M. Gemma-3 post-norms cancel uniform rescales, and Gate 0 (`RULING-wmm-gate0.md`) revealed non-monotone dose responses on LoRA families.

---

### 6. Proposal 3: P1/P2 for (b) — Search and Reader Arms
1. **Hallucination Risk:** P1's readers exhibit 53–88% false-alarm rates on inert edits (Muse). Injecting hallucinated predictions into search corrupts candidate ranking.
2. **Compute Parity Defeat:** Stage W E2 simulations (`docs/reviews/wme/wme-stagew-2026-11-18.md` §5) proved that at compute parity (180.2 evaluation equivalents), direct multi-fidelity best-of-N search beats the WM arm (mean true-gain difference **−0.0851**, power **0.006**). Expensive LLM reader calls worsen this trade-off.
3. **The 9B Ceiling:** P2's −0.35 pp drop on held-out 9B MATH mirrors our findings: headroom at 9B is narrow, and naive parameter editing degrades established capabilities.

---

### 7. What Evidence Would Change My Mind
1. **For (a) Trajectory Modeling:** A CPU variance decomposition on exposed W2 data proving $\sigma^2_{\text{run}} / \sigma^2_{\text{eval\_noise}} > 2.0$, AND a cross-validated predictor beating the pooled curve by $\Delta E \ge 0.10$ ($p < 0.01$, Holm-adjusted) without GPU spending.
2. **For SSM Memory Architectures:** An empirical demonstration on synthetic non-Markovian trajectories where an SSM beats an MLP baseline by $\ge 30\%$ MSE when $N \le 40$.
3. **For (b) Reader Features:** A double-blind audit proving an LLM reader achieves specificity $\ge 0.95$ on inert controls while delivering rank correlation $\rho > 0.60$ against ground truth on non-destructive edits.

---

### Decision Table

| decision | rationale | test or gate | cost (GPU-h, $) | kill or fallback |
|---|---|---|---|---|
| **Kill (a) permanently** | Stage W2 confirmed nested WM ties pooled curve ($d=0.000$); P1/P2 offer zero support; LC-PFN already lost ($E=2.638$) | CPU-only variance decomposition on exposed `data/wme/` | 0 GPU-h, $0 | If $\sigma^2_{\text{run}} \le \sigma^2_{\text{noise}}$, permanently close (a) in report |
| **Reject Mamba/SSM memory** | Severe data starvation ($N=15$); sequence length at $k=2$ is 2; generic sequence models overfit noise | None (architectural rejection) | 0 GPU-h, $0 | Keep frozen WM-CP canonical path representation |
| **No WSR reader in H-E2** | Readers hallucinate on inert edits (Muse 53–88%); best-of-N dominates at parity ($d=-0.0851$) | Existing E2 planning simulation | 0 GPU-h, $0 | Retain direct multi-fidelity best-of-N search |
| **Protect Study M & C1** | Gemma-3 post-norms cancel rescales; Gate 0b is final attempt for MedQA harness | Gate 0b strict accuracy $\ge 64.1\%$, truncation $\le 5\%$ | $\le 1.0$ local GPU-h, $0 | If Gate 0b fails, test MedMCQA once; else kill Study M |
| **Maintain 9B freeze** | Exploiting exploratory data risks confirmation bias; uncorrected search replicates P2's 9B failure | Board A review | 0 GPU-h, $0 | Confirmatory runs require fresh seeds and sealed splits |
