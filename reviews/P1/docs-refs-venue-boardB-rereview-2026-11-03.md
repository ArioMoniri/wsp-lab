# Board B re-review: docs/refs-venue, 3bcea5f

**Verdict: REVISE.** The quotation repairs substantially improve the ledger, but the prior claims cannot all be used in the paper, and this review does not certify all source metadata or every correction. The automated verifier exits 0 for 33 entries; it checks normalized substring presence, not semantic support or PDF fidelity.

The 12 prior entries and six additional entries selected using Python `random.seed(42)` are below. Verbatim means contiguous in the local text after the verifier's Unicode and whitespace normalization; it is not an independent check of PDF extraction. Support is for the revised ledger claim as worded, not for its older version.

| Key | Local excerpt contiguous | Revised claim supported | Note |
|---|---|---|---|
| han2026survey | yes | yes | Taxonomy. |
| w2t2026 | yes | yes | Architecture, not reported predictive performance. |
| jepawg2026physics | yes | yes | Conditional lower-dimensional concentration; no optimizer or predictor claim. |
| harness2026recursive | yes | yes | Task and harnessing recursion; not multi-agent evaluation. |
| falk2025modelhubs | yes | yes | Public heterogeneous repositories. |
| falk2025zoosize | yes | yes | Dataset diversity and generalization, not zoo-size effects. |
| goel2025diffinterpretation | yes | yes | Synthetic diffs and compatible adapter. |
| deltaactivations2025 | yes | yes | Base-relative activation embeddings. |
| leapverify2026 | yes | yes | Proposed prediction and acceptance criterion; no acceleration result. |
| mamba3 | yes | yes | Three architecture components. |
| abliteration2025comparative | yes | yes | Sixteen models and GSM8K sensitivity. |
| weightarithmetic2025steering | yes | yes | Two fine-tunes explicitly required. |
| takehi2026wsp | yes | yes | Two separate passages support the stated result. |
| wilson2026spectra | yes | yes | Source describes prior work's predictive association. |
| predictability2026dynamics | yes | yes | Restricted to the two studied regimes. |
| jepahmm2026 | yes | yes | Equivalence conditional on sufficient conditions. |
| surgicalrefusal2026 | yes | yes | Method, not a demonstrated absence of collateral damage. |
| refusalaliases2026 | yes | yes | Rank-k writer updates and random aliases. |

**Non-verbatim local excerpts found: 0** among these 18, using normalized matching. This is not a PDF transcription audit.

The following previous claims are unusable on the revised evidence: W2T's two-level architecture and numerical MAE/Spearman results; JEPAWG physics's latent dimension, optimizer and prediction setup; Leap+Verify's demonstrated training acceleration; unconditional JEPA/HMM equivalence; a single direction sufficing to bypass 320B MoE safety; gradient-free contrastive weight steering; reliable multi-agent evaluation from Harness2; and SRA avoiding collateral degradation. The old zoo-size assertion is also unsupported by the revised diversity passage. These limitations matter to H1–H3 baselines and motivation; they do not themselves establish that any hypothesis is false. The W2T numerical benchmark and JEPAWG implementation details need separate citations if retained. A full assessment of which corrected claims are indispensable to H1–H4 remains open: no hypothesis-to-paper dependency audit was completed.

The verifier diff against ef652e9 retains the previous substring and ordered-author checks and adds status, length and title checks; no loosening was found. Its successful API comparison reports matching ordered arXiv author lists in YAML and BibTeX for all arXiv entries, but I did not independently compare every author against individual arXiv pages. VENUE.md links each venue and policy source; it now avoids the previous unsupported detailed deadlines. The style repository is linked. These links alone do not verify every assertion on those external pages.

Scores: claims supported **3/5**, novelty **3/5** (provisional; no results assessed), clarity **4/5**, reproducibility **3/5**. Before approval, complete the H1–H4 dependency audit, replace or remove the unusable claims where relied upon, and independently verify bibliography and PDF fidelity. This review is deliberately limited and does not certify all 23 corrections.
