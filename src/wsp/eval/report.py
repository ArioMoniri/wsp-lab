"""Report completed runs from their per-item records, including blog gaps."""

import argparse
import json
import math
from dataclasses import fields
from pathlib import Path

from wsp.schema import EvalRecord, aggregate

BLOG = {"math500": 7.9, "gsm8k": 13.8, "ifeval": 21.0, "loop_rate": 93.0}


def wilson(successes, n):
    z = 1.959963984540054
    p = successes / n
    center = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return 100 * (center - half), 100 * (center + half)


def report(runs, out, log):
    lines = [
        "## P2 base-model reproduction",
        "",
        (
            "Model: Qwen/Qwen3.5-0.8B, thinking enabled, seed 0. "
            "Sampling: temperature 0.6, top-p 0.95, top-k 20, presence penalty 1.5."
        ),
        "",
        "| Run id | Token cap | Benchmark | n | Score | 95% CI | Blog (%) | Gap (pp) |",
        "|---|---:|---|---:|---:|---|---:|---:|",
    ]
    for run in runs:
        run = Path(run)
        manifest = json.loads((run / "manifest.json").read_text())
        if manifest.get("status") != "complete":
            raise ValueError(f"{run}: incomplete run")
        records = [
            EvalRecord(
                **{
                    k: v
                    for k, v in json.loads(line).items()
                    if k in {f.name for f in fields(EvalRecord)}
                }
            )
            for line in (run / "records.jsonl").read_text().splitlines()
        ]
        summary = json.loads((run / "summary.json").read_text())
        for name, score in aggregate(records).items():
            if name == "loop_rate":
                values = [
                    float(r.hit_max_tokens)
                    for r in records
                    if r.benchmark in ("gsm8k", "math500", "ifeval")
                ]
            else:
                values = [r.correct for r in records if r.benchmark == name]
            binary = name not in ("wikitext_ppl", "kl_to_base")
            shown = f"{100 * score:.2f}%" if binary else f"{score:.6g}"
            ci = ("[{:.2f}, {:.2f}]%".format(*wilson(sum(values), len(values)))) if binary else "—"
            blog = BLOG.get(name)
            gap = f"{100 * score - blog:+.2f}" if blog is not None else "—"
            lines.append(
                f"| {run.name} | {summary['generation']['max_new_tokens']} | "
                f"{name} | {len(values)} | {shown} | {ci} | "
                f"{blog if blog is not None else '—'} | {gap} |"
            )
    lines.extend(
        [
            "",
            (
                "Intervals: 95% Wilson intervals over items for binary rates; "
                "no interval claimed for perplexity or KL. Loop rate is the fraction "
                "of generations hitting the cap across the generation benchmarks present "
                "in that run. The 32768-token run includes GSM8K and MATH-500 only, "
                "so its loop denominator differs from the full cheap battery."
            ),
            "",
            (
                "Blog reference values supplied in the task: MATH-500 7.9%, GSM8K 13.8%, "
                "IFEval 21.0%, loop 93% at 32k. These are reference numbers, not measurements "
                "from this repository. Gaps can reflect fixed subsets, prompting, answer "
                "extraction, token budget and backend. A budget comparison alone does not "
                "identify their individual effects. The reproduction gate requires a "
                "second complete same-seed run within 1.5 percentage points; the GPU "
                "smoke test establishes exact repeatability only for eight short items."
            ),
            "",
            (
                f"Tmux log: `{log}`. Raw records, summaries, and manifests are under "
                "`data/runs/p2/`; manifests include model revision and dataset fingerprints."
            ),
            "",
        ]
    )
    path = Path(out)
    previous = path.read_text() if path.exists() else "# Results\n"
    prefix = previous.split("## P2 base-model reproduction")[0].rstrip()
    path.write_text(prefix + "\n\n" + "\n".join(lines))


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--runs", nargs="+", required=True)
    parser.add_argument("--out", default="docs/RESULTS.md")
    parser.add_argument("--log", required=True)
    args = parser.parse_args()
    report(args.runs, args.out, args.log)


if __name__ == "__main__":
    main()
