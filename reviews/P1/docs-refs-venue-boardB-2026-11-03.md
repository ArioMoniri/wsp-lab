# Board B review: docs/refs-venue, 57a51f8

**Verdict: REVISE.** The reference ledger contains fabricated or reconstructed quotations and incorrect author lists. Recheck every entry against the actual source before using this as a verified bibliography. This audit is a sample, not an exhaustive verification of all 33 entries. Whitespace was normalized for matching; “no” means the complete stored excerpt was not found, not necessarily that every constituent phrase is absent. Local full text was checked for the four requested papers; the other eight were checked against their arXiv abstract and HTML, with local text as a cross-check. Claim support below means the stored passage supports the entire stated claim; “no” includes insufficient evidence, not necessarily a false claim.

| key | verbatim found | claim supported | note |
|---|---|---|---|
| han2026survey | yes | yes | Local 2603.10090.txt; taxonomy. |
| w2t2026 | no | no | Local 2603.15990.txt; ellipsis joins prose and a fabricated table transcription; passage does not establish full architecture. |
| jepawg2026physics | yes | no | Local 2607.07127.txt; excerpt only establishes covariance omission, not dimension, optimizer or prediction. |
| harness2026recursive | no | no | Local 2610.14224.txt; framework claim exceeds passage and reliability/multi-agent evaluation are not established. |
| falk2025modelhubs | yes | yes | [HTML](https://arxiv.org/html/2510.02096); conclusion supports feasibility. |
| falk2025zoosize | yes | no | [HTML](https://arxiv.org/html/2504.10141); excerpt states evaluation, not the asserted magnitude of its results. |
| goel2025diffinterpretation | no | no | [HTML](https://arxiv.org/html/2510.05092); stored wording not found; verify actual method and exact passage. |
| deltaactivations2025 | no | no | [HTML](https://arxiv.org/html/2509.04442); stored wording not found; baseline use is our interpretation. |
| leapverify2026 | no | no | [HTML](https://arxiv.org/html/2602.19580); stored wording not found; locate evidence for acceleration. |
| mamba3 | no | no | [HTML](https://arxiv.org/html/2603.15569); local text matches but HTML text extraction does not, so source-format discrepancy needs manual resolution; passage alone does not establish improved prediction. |
| abliteration2025comparative | no | no | [HTML](https://arxiv.org/html/2512.13655); ellipsis joins noncontiguous text; verify quantitative claim. |
| weightarithmetic2025steering | no | no | [HTML](https://arxiv.org/html/2511.05408); excerpt not found and does not establish absence of gradient fine-tuning. |

Eight of these 12 complete excerpts were not found in the specified source representation; Mamba-3 is a source-format ambiguity because the local extracted text matches. Resolve against the PDF before counting it as an invented quotation. Exact matches in local extracted text are not independent proof that the PDF transcription is correct.

## Bibliography

[W2T](https://arxiv.org/abs/2603.15990) authors are Xiaolong Han, Ferrante Neri, Zijian Jiang, Fang Wu, Yanfang Ye, Lu Yin, Zehong Wang, not the listed authors. [Delta Activations](https://arxiv.org/abs/2509.04442) authors are Zhiqiu Xu, Amish Sethi, Mayur Naik, Ser-Nam Lim. [Mamba-3](https://arxiv.org/abs/2603.15569) authors are Aakash Lahoti, Kevin Y. Li, Berlin Chen, Caitlin Wang, Aviv Bick, J. Zico Kolter, Tri Dao, Albert Gu. [Comparative Analysis](https://arxiv.org/abs/2512.13655) author is Richard J. Young. [Weight Arithmetic](https://arxiv.org/abs/2511.05408) authors are Constanza Fierro and Fabien Roger. Correct both YAML and BibTeX; audit remaining entries independently. The sampled IDs and years correspond to the displayed abstract pages; other bibliography fields were not exhaustively checked.

## Venue and template

[ACL 2027](https://2027.aclweb.org/) confirms January 4 and Kyoto dates, but VENUE.md does not cite the actual deadline source. [ICML](https://icml.cc/) confirms location and dates but says its call is forthcoming; late January is explicitly a projection and needs a source and clear unconfirmed status. COLM's March 31 deadline is unsourced and presented too definitely. ARR is not bimonthly: [official schedule](https://github.com/acl-org/aclrollingreview/blob/main/dates.md) describes 10-week cycles. Check every detailed TMLR timeline, certification, anonymity and quota assertion against the linked [editorial policies](https://jmlr.org/tmlr/editorial-policies.html) and [author guide](https://jmlr.org/tmlr/author-guide.html); this review does not certify each one. The style file SHA-256 is `816214ff5919aa457b6b443bee52b15d9561421417b7f8a50cc84651519f0002`, identical to the current [official repository](https://github.com/JmlrOrg/tmlr-style-file) master file. Other template files were not compared.

Scores: claims supported **1/5**, novelty **N/A**, clarity **3/5**, reproducibility **2/5**. Blocking: repair all quotations, support each complete claim with located evidence, correct all bibliography metadata, and source or mark speculative venue information. Re-review required.
