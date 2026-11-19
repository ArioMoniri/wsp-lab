# Chair ruling, 19 Nov 2026: wme-c1 Phase L failure (HF generate not deterministic)

**Input.** `docs/reviews/wme/wme-c1-launch-2026-11-20.md` and its `.json`.

Precondition 1 (pinned-engine inert identity) and precondition 3 (committed per-engine canary references) failed on
HF `generate` at the registered batch 128:
- three runs gave three token-id hashes on one GPU;
- GPU 0 and GPU 1 also differ.

Teacher-forced and vLLM are bitwise reproducible on that stack. vLLM reproduces the stage-W canary `b34cc04e…`.

No confirmatory unit has run. No `wme-c1` or `wme-c2-gen` item has been evaluated, and no sealed file has been read.
Every decision below is therefore made before any data. Under Part I §11.4 none of them is an adaptation.

## Decisions

**C1. Repair first, with the fallback declared now.** One amendment (§II.31, v2l) records both options. Board A
checks it before Phase L is re-run.

- **(a) Primary repair (engine config only; the engine is unchanged).** HF generation runs with:
  - `torch.use_deterministic_algorithms(True)`;
  - `CUBLAS_WORKSPACE_CONFIG=:4096:8`;
  - a deterministic attention backend (eager or math SDPA; no flash or memory-efficient kernels);
  - a fixed batch composition and order, at the registered batch 128.

  It passes if 3 of 3 runs on one GPU and GPU 0 vs GPU 1 give bitwise-identical token-id hashes on the 64 exposed
  canary items.
  - First test it locally on the RTX with Qwen3.5-0.8B. This is engineering only and costs $0.
  - Then test it in the Phase L re-run.
  - Also record its throughput. If that throughput breaches the cap projection under C3, the fallback applies.
- **(b) Pre-declared fallback, used only if (a) fails Phase L.** E1 generation and the `hf_gen` canary move to the
  pinned vLLM engine, with all other settings as registered (greedy, 128 new tokens, non-thinking, the same prompts
  and stop rule).
  - This replaces the engine named in §II.4 delta 4 and §II.13 item 6.
  - The `hf_gen` canary becomes a second vLLM canary on the E1 generation prompts.
  - Because the choice between (a) and (b) is made by an outcome-blind engineering check, (b) is not an adaptation.
- **(c) Rejected.** Option 3 (HF without bitwise canaries) conflicts with §II.22.6.

**C2. The §II.29 (v2j) decisions are accepted**, including these:
- **G1:** the hash-bound 72-root, 12-block graph. The frozen manifest hash `c8d50fe2…` is binding over the §II.4
  prose.
- **G2:** `claude-opus-5-5` for the reader arms.
- **G3a:** the chart column is set to 0.
- **G18:** π is the mean over the 8 samples, matching pilot 3's code, which the registration cites.
- The 8 realised inert controls; the shortfall is reported.

The §II.30 (v2k) re-pricing rule is accepted.

**C3. Budget margin.**
- The projected total is about 80–86 GPU-h-eq.
- Phase L has already spent 3.81 GPU-h-eq ($18.26).
- The cap is 90 GPU-h-eq.

That leaves little margin. Raising the cap is a resource decision for the user. In dollars it stays inside the
authorised $1,000. If the user approves, §II.31 records it before any data. Until then the cap stays at 90. The
existing rule applies: the run stops at the cap, with loss dispositions as registered.

**C4. Order.**
1. Implement §II.31 and test (a) locally.
2. Board A check of §II.31 (a separate delegate).
3. Re-run Phase L on Lambda.
4. On pass, start the confirmatory phases.

C1 must not launch Lambda before the chair confirms that steps 1–2 are done and the cap decision is recorded.
