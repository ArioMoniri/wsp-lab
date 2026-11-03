# Target Venue and Submission Requirements

## Primary Venue: TMLR (Transactions on Machine Learning Research)

**Status:** Confirmed as primary target. Rolling submission via OpenReview ([https://openreview.net/group?id=TMLR](https://openreview.net/group?id=TMLR)).
Verified against official guidelines: [TMLR Submissions](https://jmlr.org/tmlr/submissions.html) and [Author Guide](https://jmlr.csail.mit.edu/tmlr/author-guide.html).

### 1. Length Policy
- **No strict page limit:** Submissions may be any length. However, length must be justified by content; unusually long manuscripts (excluding appendices) are explicitly warned to face reviewing delays.
- **Appendices:** Allowed after the reference section (for proofs, derivations, complete benchmark tables, and supplementary ablation runs). Note that reading the appendix is at the discretion of the reviewers.

### 2. Double-Blind Review Policy
- **Strictly Double-Blind:** All submissions must be completely anonymized (no author names, affiliations, or identifying acknowledgments in text, figures, code, or metadata).
- **OpenReview Profiles:** All co-authors must hold complete, active OpenReview profiles with declared institutional domains and personal conflicts of interest.
- **Preprints Permitted:** Authors may maintain preprints on arXiv or other servers at any point (anonymously or identified), but the TMLR submission must not link to or cite an identified version. Reviewers are forbidden from intentionally searching for author identities.

### 3. Review Timeline & Workflow
- **Phase 1 (Submission):** Authors submit an anonymized PDF and suggest appropriate Action Editors (AEs).
- **Phase 2 (Desk Evaluation & Assignment):** An Action Editor is assigned within 1 week. The AE evaluates scope, adherence to formatting, and basic quality, then assigns at least 3 qualified reviewers. Upon assignment, the paper is made public on OpenReview (without author identities).
- **Phase 3 (Reviews & Rebuttal):** Reviewers typically submit reviews within 4 weeks. Reviews are visible to authors immediately upon submission (kept private from the general public and other reviewers until 3 reviews are submitted). Authors may engage in direct open-ended discussion, post rebuttals, and update the PDF manuscript. Reviewers submit final recommendations after at least 2 weeks have elapsed since all 3 reviews became public.
- **Phase 4 (AE Decision):** Decisions are:
  - *Accept as is*
  - *Accept with minor revision* (verified by AE before publication)
  - *Reject* (with indication of whether resubmission of a significantly revised version is encouraged)

### 4. Certifications to Target
Accepted papers are automatically considered for public endorsements:
- **Reproducibility Certification (Primary Target):** Awarded to papers whose primary focus includes reproducing published results (e.g. replicating the Takehi et al. 2026 blog claims) and contributing significant added value (such as our multi-family edit zoo, Mamba/JEPA trajectory architectures, and statistical baselines).
- **Featured Certification:** For papers of oral/spotlight caliber at top-tier conferences, demonstrating significant empirical and theoretical contributions.
- **Outstanding Certification:** Best-paper equivalent awarded by the Editors-in-Chief.
- **J2C Certification:** Provides eligibility for presentation at the joint NeurIPS/ICLR/ICML Journal-to-Conference track.

### 5. Code-Release & Supplementary Material Expectations
- **Supplementary Packages:** Authors may upload up to 100 MB of supplementary material (PDF or ZIP).
- **Reproducibility Standard:** TMLR strongly encourages releasing clean, anonymized source code, configuration manifests, and datasets to guarantee full reproducibility.
- **Broader Impact Statement:** Mandatory for studies with potential safety implications. For our work, this covers the abliteration sub-study: our checklist explicitly specifies that **no uncensored model weights are released**, only weight-delta statistics, audit predictors, and mathematical evaluations.
- **Licensing:** Published papers are open-access under CC BY 4.0; authors retain copyright.
- **Authorship Quotas:** Submissions consume budget under the Generalized Harmonic Quota Rule ($N_1 = 2$, $N_9 = 9$).

### 6. Official Template Status
The official template from [JmlrOrg/tmlr-style-file](https://github.com/JmlrOrg/tmlr-style-file) has been downloaded into `paper/template/`:
- `paper/template/tmlr.sty`: Official style package (supports `[preprint]` and `[accepted]` options).
- `paper/template/tmlr.bst`: Official BibTeX style file.
- `paper/template/main.tex`: Official LaTeX submission skeleton.
- `paper/template/main.bib`: Official example bibliography.
- `paper/template/math_commands.tex`: Standard math macro definitions.

---

## Fallback Venues (Deadlines as of November 2026)

If TMLR review outcome requires major redirection or a conference venue is preferred:

### Fallback 1: ACL 2027 (via ACL Rolling Review - ARR)
- **Submission Deadline:** **January 4, 2027 (AoE)** (January 2027 ARR cycle, eligible for ACL 2027 commitment).
- **Conference Location & Dates:** Kyoto, Japan | August 17–22, 2027.
- **Why it fits:** The leading venue for computational linguistics, language model behavioral interventions, and alignment ablation. Our study's focus on LLM safety alignment (Heretic, ARA, directional abliteration, and refusal steering) aligns directly with ACL's Interpretability and Safety tracks.
- **Format:** Standard ACL format via ARR; fast bimonthly review turnaround with structured author responses.

### Fallback 2: ICML 2027 (International Conference on Machine Learning)
- **Submission Deadline:** **Late January 2027 (AoE)** (projected abstract deadline mid-January 2027; full paper late January 2027).
- **Conference Location & Dates:** Riocentro, Rio de Janeiro, Brazil | July 31 – August 6, 2027.
- **Why it fits:** Flagship venue for machine learning architectures and representation learning. Ideal if our state-space trajectory modeling (H2 / Mamba-3 sequence prediction) and joint-embedding representations (H3 / JEPA over weight space) demonstrate strong theoretical and empirical performance gains.
- **Format:** Double-blind, 8-page main text limit (plus references/appendices).

### Secondary Contingency: COLM 2027 (Conference on Language Modeling)
- **Submission Deadline:** **Late March 2027 (~March 31, 2027 AoE)**.
- **Conference Dates:** Expected October 2027.
- **Why it fits:** Dedicated venue for deep research into language models, scaling, and mechanistic/weight-space introspection.
