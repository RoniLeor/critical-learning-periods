"""Render partial or complete reconstruction results without selecting test checkpoints."""

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

parser: argparse.ArgumentParser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--results", type=Path, required=True)
args: argparse.Namespace = parser.parse_args()
runs: list[dict[str, Any]] = [
    json.loads(path.read_text()) for path in sorted(args.results.glob("s*-blur*.json"))
]
complete: list[dict[str, Any]] = [run for run in runs if run["complete"]]
fig: Any
axes: Any
fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(15, 4.5), sharey=True)
deficit: int
axis: Any
colors: tuple[str, ...] = ("#155e75", "#a16207", "#9f1239")
for axis, deficit in zip(axes, [0, 40, 100], strict=True):
    run: dict[str, Any]
    for run in runs:
        if run["deficit"] != deficit:
            continue
        history: list[dict[str, Any]] = run["history"]
        axis.plot(
            [row["epoch"] for row in history],
            [row["accuracy"] * 100 for row in history],
            label=f"Seed {run['seed']}",
            color=colors[run["seed"]],
        )
    if deficit:
        axis.axvline(x=deficit, color="#555555", linestyle="--", label="Blur removed")
    axis.set_title(f"{deficit} blurred + 160 clear epochs")
    axis.set_xlabel("Completed training epochs")
    axis.set_ylim(0, 100)
    axis.grid(alpha=0.2)
    axis.legend()
axes[0].set_ylabel("Clear CIFAR-10 test accuracy (%)")
fig.suptitle("Paper reconstruction · All-CNN · full CIFAR-10 · final-epoch evaluation")
fig.tight_layout()
fig.savefig(args.results / "accuracy.png", dpi=160)
plt.close(fig)
lines: list[str] = [
    "# Paper reconstruction results",
    "",
    f"Completed {len(complete)} of 9 planned runs. This is a documented reconstruction, not author code.",
    "",
    "| Seed | Blurred epochs | Completed epochs | Final/current accuracy | Complete |",
    "|---|---:|---:|---:|---|",
]
for run in runs:
    lines.append(
        f"| {run['seed']} | {run['deficit']} | {len(run['history'])} | {run['history'][-1]['accuracy'] * 100:.2f}% | {run['complete']} |"
    )
if len(complete) == 9:
    lines.extend(["", "Across-seed final accuracy: mean ± sample standard deviation.", ""])
    for deficit in [0, 40, 100]:
        scores: np.ndarray = np.asarray(
            [run["history"][-1]["accuracy"] * 100 for run in complete if run["deficit"] == deficit]
        )
        lines.append(
            f"- {deficit} blurred epochs: {scores.mean():.2f}% ± {scores.std(ddof=1):.2f} percentage points."
        )
lines.extend(
    [
        "",
        "![Accuracy](accuracy.png)",
        "",
        "See docs/protocol.md in the repository and config.json for source settings and reconstruction assumptions. Each condition receives 160 clear epochs. Test checkpoints were not selected or tuned. Three seeds describe training variability but do not establish permanent impairment.",
        "",
        "Original paper: https://arxiv.org/html/1711.08856v3",
    ]
)
(args.results / "report.md").write_text("\n".join(lines) + "\n")
print(f"Report saved: {len(complete)}/9 complete")
