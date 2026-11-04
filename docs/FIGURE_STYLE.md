# Figure style — modelled on JEPAWG (arXiv 2607.07127 / 2610.30847)

Observed in the JEPAWG papers: white background; one panel per idea; thin axes; matplotlib default
serif-free font at 8–9 pt; latent-space plots as 2-D scatter coloured by the physical/control parameter
with a viridis colour bar; parameter sweeps as lines with shaded ±1 s.d. over seeds; bifurcation/response
curves as the norm of a derivative vs. the parameter with the known critical point as a dashed vertical
line; method comparison bars with hatched baselines; small multiples for the four systems.

House rules (`wsp.plots` implements them):
- `plt.rcParams`: figure.dpi 200, font.size 8, axes.linewidth 0.6, legend.frameon False, savefig.bbox tight.
- Colour: viridis for ordered variables (edit strength, KL target), Okabe–Ito palette for categories
  (base / abliterated / heretic / LoRA / random-direction). Colour-blind safe always.
- Widths: single column 3.25 in, double 6.75 in (TMLR). Export PDF + PNG.
- Shaded CI bands = bootstrap 95 % over edits; markers for individual seeds when n ≤ 10.
- Our analogue of JEPAWG's "derivative of decoded weights vs parameter" plot is
  "predicted Δscore vs. edit strength, with the measured Δscore overlaid" per edit family.
- Latent plot: JEPA latent (2-D PCA) coloured by measured MATH-500, shape = edit family.

## Diagram style — modelled on the Meta^n README figures (added 4 Nov 2026)

Reference copies (MIT, © 2026 Zae Myung Kim; licence alongside) live in `assets/reference/meta-n/`:
`paper_at_glance.png`, `meta_improv_concept_diffs.png`, `build_step.png`, `run_step.png`
(from github.com/minnesotanlp/meta-n, docs/figures). Look at them before drawing any schematic for the README or
the paper. JEPAWG governs *data plots*; Meta^n governs *diagrams and the README hero composition*.

What those four figures do well, and the house rule derived from each:

1. **"At a glance" = concept on the left, headline curve on the right, one image.** Left: a schematic of the method;
   right: the single result that justifies it, with the held-out point marked by a distinct marker (★) and the one
   thing to notice circled with a dashed red ellipse. Italic white-on-dark label chips inside the figure carry the
   takeaway ("Faster convergence & stronger generalisation"), so the image reads without its caption.
   *Ours:* left = edit → world model → predicted Δscore → real evaluation only when uncertain; right = predicted vs
   measured Δscore on held-out edit families, with the held-out family starred. Chips: "predicts before evaluating",
   "held-out family". The README hero (Lottie) animates exactly this composition; the paper's Figure 1 is its static twin.
2. **Depth is nesting, coloured by a light-to-dark ramp of one hue.** Nested rounded boxes (cream → amber → dark amber)
   show hierarchy without arrows; roles are annotated outside with thin arrows. *Ours:* an edit chain is drawn as nested
   boxes base ⊂ edit₁ ⊂ edit₂ … (step semantics), ramp = viridis-light to viridis-dark, never more than four levels.
3. **"Prior (1), prior (2), ours (3)" side-by-side concept diff** with identical primitives in each column, a shared
   legend, and the only difference being the thing we claim. *Ours:* (1) evaluate after every edit, (2) cheap proxy
   metrics after every edit, (3) weight-space world model before the edit. Same circle/rectangle vocabulary in all
   three; "(ours)" in italics in the title of column 3 only.
4. **Build step: a vertical chain of states with the score under each, green when it improved, red when it regressed,
   and the artefact that caused the change in a monospace box to the right.** *Ours:* one real trajectory from the
   zoo: state boxes = checkpoint id + measured score (green/red by sign of Δy), right-hand boxes = the edit record
   (family, rank, α, target layers) *and* the predictor's forecast for that step with its interval, so a reader sees
   where the world model was right and where it was not. Real run ids only, never illustrative numbers.
5. **Run step: a single vertical flow, nested stack in the middle, final score in green at the bottom.** *Ours:* the
   inference-time flow (candidate edit → canonical features → predictor → gate → evaluate/skip).

Drawing rules shared by all diagrams:
- Rounded rectangles, 1 pt navy outline (#1F3A5F), white or light-grey fill for inputs/outputs, Okabe–Ito fills for
  edit families; green #009E73 / vermillion #D55E00 for improvement / regression; dashed red only for the one
  callout per figure.
- Text: serif (TeX Gyre Termes / Times) inside diagrams to match the TMLR body font; monospace for ids, code and
  edit parameters; italic for the takeaway chips; no bold except the state label.
- One legend per figure, inside the frame, white box with a thin outline. No titles inside the image for the paper;
  the README version may add the italic chip.
- Build with `src/wsp/plots/diagrams.py` (to be written in P5 next to `panels.py`: matplotlib patches + `annotate`), export SVG for the README and PDF for the
  paper from the same function; the Lottie hero is generated from the same layout constants
  (`assets/lottie/`), so the animation and the static figure never drift apart.
- Every number in a diagram comes from a run id named in the caption (`docs/RESULTS.md`), per AGENTS.md: no demo values.
