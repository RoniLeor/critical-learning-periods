"""Compare completed All-CNN models with repeated direct Monte Carlo Fisher probes."""

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import Tensor, nn

from .core import AllCNN, Transform
from .fisher import FisherTrace

parser: argparse.ArgumentParser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--data", type=Path, required=True)
parser.add_argument("--results", type=Path, required=True)
parser.add_argument("--out", type=Path, required=True)
args: argparse.Namespace = parser.parse_args()
OUT: Path = args.out
OUT.mkdir(parents=True, exist_ok=True)
torch.set_num_threads(2)
data: dict[str, Tensor] = torch.load(f=args.data / "tensors.pt", weights_only=True)
generator: torch.Generator = torch.Generator().manual_seed(20260930)
indices: list[Tensor] = []
label: int
for label in range(10):
    candidates: Tensor = torch.where(data["labels"] == label)[0]
    indices.append(candidates[torch.randperm(len(candidates), generator=generator)[:30]])
selected: Tensor = torch.cat(indices)
images: Tensor = Transform()(images=data["train"][selected])
(OUT / "probes.json").write_text(
    json.dumps(
        {
            "training_indices": selected.tolist(),
            "labels": data["labels"][selected].tolist(),
            "sampling_seed": 20260930,
        },
        indent=2,
    )
)
results: list[dict[str, Any]] = []
deficit: int
for deficit in (0, 40, 100):
    path: Path = args.results / f"s0-blur{deficit}.pt"
    state: dict[str, Any] = torch.load(f=path, map_location="cpu", weights_only=False)
    if len(state["history"]) != deficit + 160:
        raise ValueError("Only final models are eligible")
    model: nn.Sequential = AllCNN()()
    model.load_state_dict(state["model"])
    before: dict[str, Tensor] = {name: value.clone() for name, value in model.state_dict().items()}
    samples: list[list[float]] = []
    repeat: int
    for repeat in range(5):
        raw: dict[str, float] = FisherTrace(model=model, device=torch.device("cpu"))(
            images=images, seed=900 + repeat
        )
        groups: list[float] = [
            sum(value for name, value in raw.items() if int(name) // 3 == group)
            for group in range(9)
        ]
        assert np.isclose(sum(groups), sum(raw.values()))
        samples.append(groups)
        print(
            json.dumps({"deficit": deficit, "repeat": repeat, "total_trace": sum(groups)}),
            flush=True,
        )
    assert all(torch.equal(value, model.state_dict()[name]) for name, value in before.items())
    assert all(parameter.grad is None for parameter in model.parameters())
    counts: list[int] = [
        sum(
            parameter.numel()
            for name, parameter in model.named_parameters()
            if int(name.split(".")[0]) // 3 == group
        )
        for group in range(9)
    ]
    results.append(
        {
            "deficit": deficit,
            "seed": 0,
            "epoch": len(state["history"]),
            "accuracy": state["history"][-1]["accuracy"],
            "checkpoint_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "parameter_counts": counts,
            "traces": samples,
        }
    )
    (OUT / "measurements.json").write_text(
        json.dumps(
            {
                "method": "Direct model Fisher, eval mode, per-example posterior-sampled gradients; not paper variational estimator",
                "probe_count": len(selected),
                "repeats": 5,
                "results": results,
            },
            indent=2,
        )
    )
fig: Any
axes: Any
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(13, 4.8))
colors: tuple[str, ...] = ("#155e75", "#b7791f", "#a1264b")
position: int
result: dict[str, Any]
summary: list[dict[str, Any]] = []
for position, result in enumerate(results):
    values: np.ndarray = np.asarray(result["traces"])
    means: np.ndarray = values.mean(axis=0)
    totals: np.ndarray = values.sum(axis=1)
    shares: np.ndarray = values / totals[:, None] * 100
    axes[0].bar(position, totals.mean(), yerr=totals.std(ddof=1), color=colors[position], capsize=5)
    axes[1].plot(
        range(1, 10),
        shares.mean(axis=0),
        marker="o",
        color=colors[position],
        label=f"Blur {result['deficit']} epochs",
    )
    axes[1].fill_between(
        range(1, 10),
        shares.mean(axis=0) - shares.std(axis=0, ddof=1),
        shares.mean(axis=0) + shares.std(axis=0, ddof=1),
        color=colors[position],
        alpha=0.12,
    )
    summary.append(
        {
            "deficit": result["deficit"],
            "accuracy": result["accuracy"],
            "total_trace_mean": float(totals.mean()),
            "label_sampling_sd": float(totals.std(ddof=1)),
            "layer_trace_mean": means.tolist(),
            "layer_share_percent": shares.mean(axis=0).tolist(),
        }
    )
axes[0].set_xticks(range(3), ["No blur", "40 epochs", "100 epochs"])
axes[0].set_ylabel("Total model Fisher trace")
axes[0].set_title("Final model sensitivity")
axes[1].set_xlabel("Convolution block (Conv + BatchNorm parameters)")
axes[1].set_ylabel("Share of total trace (%)")
axes[1].set_xticks(range(1, 10))
axes[1].set_title("Distribution across nine blocks")
axes[1].legend()
fig.suptitle("Final All-CNN models · seed 0 · 300 identical clear training probes")
fig.text(
    0.5,
    0.01,
    "Error bars/bands: sampling SD over 5 label draws; not confidence intervals or training-seed variability.",
    ha="center",
    fontsize=9,
)
fig.tight_layout(rect=(0, 0.035, 1, 1))
fig.savefig(OUT / "comparison.png", dpi=170)
fig.savefig(OUT / "comparison.pdf")
(OUT / "summary.json").write_text(json.dumps(summary, indent=2))
print(json.dumps(summary, indent=2))
