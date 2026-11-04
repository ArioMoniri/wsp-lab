# Board A review: `feat/plots-design` (head 536ba07)

- Reviewer: reviewer-ml (Board A), read-only; no code edited
- Scope: `git diff main...feat/plots-design` (24 files, +1914/-1), worktree `.worktrees/feat/plots-design`
- Verdict: **REVISE**. Two blocking items, both small to fix. The plotting code, tests and Lottie asset are sound.

## Blocking

| # | Where | Finding | Failure scenario |
|---|---|---|---|
| B1 | `scripts/demo_figures.py:68-69, 76-78, 87-89, 97-102` (default output dir `:54` = `paper/figures`) | The demo figures plot invented numbers as if they were results: an SSM "Δ MATH-500" sweep, a "Spearman ρ on MATH-500" method bar chart ranking Mamba-2 SSM 0.36 above JEPA 0.34, a predicted-vs-measured curve, and four small-multiple benchmarks. None is labelled synthetic. Only `demo_latent_scatter` says "toy zoo" (`:63`). The script docstring says "toy data", but the PNGs in `paper/figures/` do not. Affected: `paper/figures/demo_method_bars.png`, `demo_sweep_with_ci.png`, `demo_pred_vs_measured.png`, `demo_small_multiples.png`. | A reader, or the paper build, takes `demo_method_bars.png` as a result. It shows a method ranking that was never run, with no marker anywhere in the figure or path to say so. **Fix:** add a visible "TOY DATA, NOT RESULTS" label to every demo panel, and write demo output somewhere outside `paper/figures/` (for example `docs/style_examples/`). |
| B2 | `sessions/2026-11-03-003.md:4` | Adds `Host: <aws-host>`. That hostname encodes the rented instance's private AWS IP (<private-ip>). The board ASK (`board/P1.md`, 23:22:42Z) says the repo is public and that no rented-instance IPs may appear in commits or posts. **Pre-existing on `main`, not introduced here, but it needs a separate fix:** `docs/INFRA.md:4` also has the host and instance id `<instance-id>`, and `sessions/2026-11-03-001.md:4,8` and `-002.md:4` repeat the host. | Publishing the branch exposes the rented instance's address and instance id in a public repo, which the ASK prohibits. **Fix here:** delete or redact the `Host:` line. **Escalate:** scrub `main`. |

## Non-blocking (fix before paper use)

| # | Where | Finding |
|---|---|---|
| N1 | `src/wsp/plots/panels.py:382` (`small_multiples`) | Panel y tick labels collide with the neighbouring panel's frame (`demo_small_multiples.png`: "0.05", "-0.25", "0.025", "-0.125" overlap the left panel). Fix with `subplots_adjust(wspace=…)` or constrained layout. |
| N2 | `src/wsp/plots/panels.py:51-68`, `docs/FIGURE_STYLE.md` ("Shaded CI bands = bootstrap 95 % over edits") | The code resamples **runs**, which is what the brief asked for and is the correct unit for seed uncertainty. The docs say "over edits", and `panels.py:8` and `style.py:10-11` say "run/edit axis". Reconcile the wording. Also: the bands show no n, and a percentile bootstrap with 2–3 runs is unstable (`panels.py:59` accepts n = 2). Print n in the caption or legend. |
| N3 | `src/wsp/plots/style.py:3` | "Copied from the JEPAWG papers" overstates; `docs/FIGURE_STYLE.md` says "modelled on". The JEPAWG conventions quoted in `panels.py:6-12` (for example "alpha ~0.35") cannot be checked from the repo. Board B should check them against `paper/refs/verified.yaml`, and the wording should be softened. |
| N4 | `scripts/make_lottie.py`, `scripts/render_pipeline_gif.py:32-52` | The Lottie JSON and the GIF are two independent drawings with no shared source. The GIF is not rendered from the JSON. The Lottie has **not been rendered visually**: no Lottie renderer is installed, so I checked its structure only. A browser check of `site/index.html` is still needed. |
| N5 | `scripts/render_pipeline_gif.py:52`, `assets/pipeline.gif` frame 0 | "BASE MODEL" (16 pt bold) is flush against the box border. Cosmetic. |
| N6 | `assets/pipeline.gif` | 826 kB binary committed to git. No size budget is set for it, and the test caps it at 1 MB (`tests/test_assets.py:120`). Consider fewer frames or a palette to bring it under 300 kB. |
| N7 | `site/index.html:2-3` | Canvas is 900 × 320 px CSS against a 1200 × 300 (4:1) animation, so it letterboxes. The CDN script has no SRI hash; the version is pinned, which is acceptable. |
| N8 | `panels.py:328` (`method_bars`), `panels.py:381` (`small_multiples`) | Figure width is chosen from panel count, not from the house rule (3.25 single / 6.75 double only). A 5-method bar chart therefore becomes double width. Pass the width explicitly for paper figures. |
| N9 | `tests/test_plots.py:176-184` | The bootstrap test only checks that the band straddles the mean. It does not compare against an independent percentile computation, and it does not prove that resampling is over runs rather than points. Add a test that recomputes the 2.5/97.5 percentiles from the same seeded index draws. |
| N10 | `panels.py` (all `plt.subplots` calls) | Panels leave figures registered in pyplot. Callers must close them; the demo does (`demo_figures.py:109`). Document this. |

## Checks

1. **Style (`style.py`)**
   - rcParams match the house rules: dpi 200, font 8, axes.linewidth 0.6, legend.frameon False, savefig.bbox tight (`style.py:82-117`).
   - All eight Okabe–Ito hex codes match the published values (`style.py:32-41`).
   - Ordered colour is viridis (`:70`).
   - Widths are 3.25 / 6.75 in (`:25-26`).
   - `savefig` writes PDF and PNG (`:149-152`).
2. **Panels (`panels.py`)**
   - `_bootstrap_ci` (`:51-68`) is a real bootstrap: whole runs are resampled with replacement, 1000 replicates, seeded, 2.5 / 97.5 percentiles.
   - `pred_vs_measured` overlays measured points as open circles (`:277-288`) with a dashed zero line.
   - `method_bars` hatches baselines with `///` (`:336-346`) and checks that each CI contains its score (`:324-325`).
   - No panel function calls `plt.close`. Each returns a `Figure`.
3. **Tests**
   - `pytest -q`: 59 passed, 2 deselected. The deselected tests are the pre-existing GPU tests `tests/test_gpu.py` and `tests/test_eval_gpu.py` on `main`.
   - `ruff check src tests scripts` and `ruff format --check` both pass.
   - Assertions are substantive, not smoke-only. They cover the palette, the band geometry, the hatch rule, the Lottie regenerating byte-identically, and the GIF timing.
   - Gap: N9.
4. **Lottie and README**
   - `assets/lottie/pipeline.json`: v 5.7.4, fr 30, ip 0, op 100 (3.33 s), 1200 × 300, 22 layers. Every layer has `ks` (o/p/a/s/r). A keyframe and shape walk found 0 problems. 26,972 bytes (≤ 150 kB).
   - Stage labels in the animation: BASE MODEL, abliterate / LoRA / merge, DELTA TOKENS, SSM, FORECAST.
   - GIF: 100 frames, 3.33 s, 1200 × 300, loops. I viewed frames 0 and 50; they show the pipeline.
   - `site/index.html:7` references `../assets/lottie/pipeline.json`.
   - `README.md` gained exactly one line, the GIF, directly under the hero SVG.
5. **Figures**
   - `python scripts/demo_figures.py --out <tmp> --seed 0` run into a scratch directory, not the worktree. All five PNGs regenerate byte-identical to the committed ones.
   - Sizes: 55.7, 58.1, 43.9, 40.3 and 160.8 kB, all under 300 kB.
   - Visual style matches `docs/FIGURE_STYLE.md` (white background, full-box thin axes, viridis colourbar, CI bands, hatched baselines), except N1 and B1.
6. **Secrets and IPs**
   - No keys, tokens or private-key material in added lines.
   - Only host identifier: B2.

## Notes

- The main checkout has uncommitted `board/P1.md` edits from other agents and a newer `main` head (`30f67a0`). Those edits are not part of this review and were left untouched.
- The `feat/plots-design` worktree has untracked `.venv` and `data/data`. They are not in the branch diff and should not be committed.
- The demo run and all viewing were done on copies in `/tmp`; the worktree was not modified.
