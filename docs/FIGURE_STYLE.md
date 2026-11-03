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
