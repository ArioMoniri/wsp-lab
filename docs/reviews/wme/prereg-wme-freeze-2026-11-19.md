# PREREG_WME v2: freeze record (19 Nov 2026)

**Status: FROZEN 19 Nov 2026.**
- Registration: `docs/PREREG_WME.md`, Part I as superseded by Part II through §II.28.
- Branch `prereg/wme-freeze`, from main `967d61d`.
- Annotated tag `prereg-wme-v2-frozen` on the freeze commit; it is not pushed.
- No confirmatory outcome exists at the freeze.
- Nothing sealed was opened for the freeze: the checks are hash-only.

## Freeze conditions

The conditions were checked by `scripts/wme_freeze_check.py` (report `docs/reviews/wme/wme-freeze-check-2026-11-19.json`,
sha256 `b9d9804f37800d5de5d67331c5cab44cb7cfa633c8408dfab5bcf86d1cb79fae`, `all_met: true`).

| # | Condition | Evidence | Met |
|---|---|---|---|
| (i) | Board A acceptance (Part I); Board A accepts v2 with conditions 1–14 resolved (Part II) | The chain below ends in freeze re-check 3, **ACCEPT WITH CONDITIONS**. Its conditions (two wording boundaries and the serialized launch restriction) are recorded in §II.28.1–§II.28.3. | yes |
| (ii) | No SIM placeholder remains open | 18 of 18 Part I occurrences resolve to Appendix C.2; 0 open in Part II outside Appendix C (grep in the check report, `placeholders`) | yes |
| (iii) | No MANIFEST placeholder remains open | 44 of 44 Part I occurrences resolve to Appendix C.1, and every C.1 field is in the freeze manifest; 0 open in Part II outside Appendix C. Two fields bind at registered events: the panel split hashes (bound at the v2e build) and the winners-manifest hash (after all episodes, before the sealed read). | yes |
| (ii) Part II | Freeze-build prerequisites: panels built and re-audited after every exposure, including stage W2 | `audit.panels_v2e.clean = true`; panel audit and root-lineage reports below | yes |
| (iii) Part II | Stage-W2 outcome known | W2 failed, so H-T3 is not registered | yes |
| (iv) | A dated freeze commit records the manifest, simulation-report and panel hashes | This record and §II.28.5; every simulation-report hash resolves to exactly one committed file | yes |

## Hashes

**Freeze manifest.** `docs/prereg/wme_manifest.json`, sha256 **`02514cb89f7229458e631232cf42930db36530cf26becf544485517d37ede41e`** (generated at `ec829cf`).

**Simulation reports:**

| Report | File | sha256 |
|---|---|---|
| `cal_v2b` | `docs/reviews/wme/wme-v2b-calibration-2026-11-17.json` | `4a0742df232ae1340efa540ff9e6cc27a374dc2bdc34cdaa1a89fd449aea9720` |
| `cheap_v2b` | `docs/reviews/wme/wme-powersim-v2b-2026-11-17.json` | `bfa64d8a17b89e8c6afedc58c3383e55c6366985e0e5e31c974dbaffad5b1c7b` |
| `cpsel_v2b` | `docs/reviews/wme/wme-v2b-cpsel-2026-11-17.json` | `f20b2cde4bcc0957ac623ded4901d9ddfc3e5c5f0b48a5ac1eed70389ad41216` |
| `e2sens_v2d` | `docs/reviews/wme/wme-v2d-e2sens-2026-11-19.json` | `56c3ce4f5594a53655bb731f2e82877efb3f9d8d6024a48241394b6ee2469935` |
| `ph100b` | `docs/reviews/wme/wme-ph100b-2026-11-17.json` | `2e825cd3c05c9cdf3aa9ea9f79d256e176596bac30983d50001a90d2409e2857` |
| `r6_v2c` | `docs/reviews/wme/wme-v2c-r6-2026-11-18.json` | `0fd6e6640089021113c8109a79cced747f37e378bd0f6791a62c2e4e40553fd7` |
| `refit_v2b` | `docs/reviews/wme/wme-powersim-refit-v2b-2026-11-17.json` | `ed9e5a79bba49658db0fc1fb1055d32299bb6166e28eb0ed7f0fd707e48d8fef` |
| `sbon_v2c` | `docs/reviews/wme/wme-v2c-sbon-2026-11-18.json` | `716a49e11bd60a02ea82b823e5468238778aba2731f9a66725a0701d9cb5eefb` |
| `full_refit` | `docs/reviews/wme/wme-powersim-refit-2026-11-17.json` | `424076b96ccf9f1c729900419b2b76fe74f5bf8a8739336f12c87ed95288b5ac` |

**Stage reports:**
- stage W `docs/reviews/wme/wme-stagew-2026-11-18.json`: `4b19a3945b96be1fe1cd9a9163b4a5506d996126c1c0a5b37e62a061a46d6c09`;
- stage W2 `docs/reviews/wme/wme-stagew2-2026-11-18.json`: `8443bd8a1df6c332b7b71b0b120d91d025e8fc7f296f36aabb167a6abcbaf302`.

**Panels** (hash only):

| Panel | Manifest sha256 | Content hash |
|---|---|---|
| `wme-c1` | `9dd5c9b6bbc9fade22a9adabdeb3d2626ffeea0a35f21dd62ffb986585d9c208` | `89739c7818e29d618ac3bcb52d7e51cd29a9361e7774a71b412646775c630c1d` |
| `wme-c2-gen` | `4cdb7728c237acb9d9b44bd68f94ec47940383be3348adb231205fd65bab790e` | `a24d5f0ef7de185136cb31464336d6c0537c631fa21b25b184619fb04497c43b` |

**Audits:**
- panel audit `docs/reviews/wme/wme-v2e-panel-audit-2026-11-19.json`: `a251d79880d964a94c390d5e6604f53631d44938ef2b8f20426f30aab50f5311`;
- root lineage `docs/reviews/wme/wme-v2f-root-lineage-2026-11-19.json`: `0df383ae561f81afc42cfd24b0fd337edfb44797be656ef1f10dd3810281b2cf`.

**Run-guard evidence** (§II.27):
- property report `docs/reviews/wme/wme-v2i-runguard-property-2026-11-19.json`: `713f0ecdf5e679837f8cc0df92bd71e7266f47ff12d9e48e270c2302fea4cb9b`;
- mutation report `docs/reviews/wme/wme-v2i-property-mutations-2026-11-19.json`: `08ae3734df491eecc3f2c63f1664aec073764c3656c2e9545dcccc4cde9b62e1`.

## The Board A chain

| Review | Stage | Verdict | sha256 |
|---|---|---|---|
| `docs/reviews/council/board-a-prereg-wme-2026-11-17.md` | v1 | ACCEPT WITH CONDITIONS (14 changes) | `2bbfcb0fe9417832d51427e75b43c753cc2b5f85b6e86042a1900bfbde9d759a` |
| `docs/reviews/council/board-a-prereg-wme-v2-2026-11-18.md` | v2 | ACCEPT WITH CONDITIONS (changes 1-7) | `4104d87d17dec564f603be30d63421f7b915b6b997efbb812b67be6e1e2d2f19` |
| `docs/reviews/council/board-a-prereg-wme-v2-confirm-2026-11-19.md` | confirmation | ACCEPT WITH CONDITIONS (blockers A, B) | `040e6be1e303534519501939a7606b71fd2a69779ab5f6f6b9e8119feeace234` |
| `docs/reviews/council/board-a-prereg-wme-v2-final-2026-11-19.md` | final re-check | ACCEPT WITH CONDITIONS; freeze blocked by B (A closed) | `9d42e363e818486959d0dea7c9ec178dfade6ff42500e6c0fc6cb85c21d501ff` |
| `docs/reviews/council/board-a-prereg-wme-v2-freeze-2026-11-19.md` | freeze re-check | REJECT (B open) | `153049ca42dd8baf2a3a7f16528b59ed5c299945063b82a839e6e3bfdc32f5cf` |
| `docs/reviews/council/board-a-prereg-wme-v2-freeze2-2026-11-19.md` | freeze re-check 2 | REJECT (B open) | `7cfcc549b089d619a1274eacb5c3cf4963f0e77be6bdc77852dcfdec5bf3520f` |
| `docs/reviews/council/board-a-prereg-wme-v2-freeze3-2026-11-19.md` | freeze re-check 3 | ACCEPT WITH CONDITIONS (B closed for the serialized production path) | `c035754641342f152a01f5c0760f21a3fcbd897449128fe4be01c593adbade16` |

## What is registered

| Class | Claims |
|---|---|
| **Registered primaries** | **H-E1**: own family, one-sided 0.025, conjunctive gates. **H-E2**: own family, family-stratified sign-flip, one-sided 0.025; a conditional non-symmetry claim under the E2-scope ruling. |
| **Registered secondaries** | **S-BoN**: in the H-E2 secondary Holm family; reported whatever the result. **S-TE**: own family, Holm over 2 benchmarks at 0.025. |
| **Descriptive** | **D-HT**: old H-T, WM-CP vs the static family, on the bank. **D-HT2**: stages W and W2. **D-S**: the stage-S slope. **D-C2**: C2 controls for every arm; validity preconditions halt the run. R6 is descriptive only. C2 is not a gate. |
| **Negative result** | **(a)**: on Qwen3.5-9B, edit-specific trajectory prediction did not beat path-type averages (stages W and W2). Reported by the binding §II.23 rule. |
| **Not registered** | **H-T3**: W2 failed its pre-declared rule. |
| **Unresolved, not run** | **S-E1-sec**: not powered (0 passing cells). |

**Budget.**
- Registered total: **66.93 GPU-h-eq, about $360**. That is 56.66 design + 8.43 S-BoN + 1.84 S-TE.
- The absolute combined limits are 90 GPU-h-eq and $1,000 (`CapExceeded`, an engineering abort that no decision can
  override). A phase above 1.25× its price pauses.

## Launch restriction (chair decisions, binding, verbatim)

> (1) The 9B confirmatory run uses ONLY the supported serialized execution path: one controller host owning the single authoritative local study log, one serialized guard call stream (no shared-object threads), one active confirmatory GPU instance at a time; the "up to 4 concurrent instances" plan is withdrawn for this freeze. Any parallel adapter would require a dated amendment with atomic aggregate reservations and Board A check before use.
>
> (2) Dead-man allowance for the single instance equals the remaining combined budget; no other confirmatory instance may run.

**Runtime scope (§II.28.1).**
- Calls sharing a `StudyLog` object must be serialized, including across threads.
- A cached refresh verifies the last cached record and the appended tail, not the whole cached prefix.
- Immutable-prefix storage is an operating assumption.
- A verified fresh replay is required on every restart and recovery.
- The hash chain and head are not protection against a coordinated rollback of the log and its head.

**Mandatory before the first confirmatory unit (§II.28.3):**
1. pinned-engine inert identity;
2. real 1-vs-2-GPU parity;
3. committed per-engine canary references;
4. evidence and fallback binding;
5. unique physical-instance identities;
6. deployed persistence and repair;
7. authoritative time and dead-man reconciliation;
8. `ProductionGuard` only.

None of these is performed by the freeze.
