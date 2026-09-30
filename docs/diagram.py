"""Render a paper-style architecture and results figure from committed measurements."""

import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

OUT: Path = Path(__file__).resolve().parent
RESULTS: Path = OUT.parent / "results" / "seed0"
COLORS: tuple[str, ...] = ("#343434", "#0072B2", "#D55E00")
DEFICITS: tuple[int, ...] = (0, 40, 100)
CLEAR_EPOCHS: int = 160
CHANNELS: tuple[int, ...] = (96, 96, 192, 192, 192, 192, 192, 192, 10)
SIZES: tuple[int, ...] = (32, 32, 16, 16, 16, 8, 8, 8, 8)
plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 9,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.7,
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
    }
)
fig: Any = plt.figure(figsize=(14, 7.4), facecolor="white")
architecture: Any = fig.add_axes(rect=(0.04, 0.60, 0.93, 0.35))
architecture.set_xlim(left=-0.7, right=11.6)
architecture.set_ylim(bottom=-1.0, top=2.5)
architecture.axis("off")
architecture.text(x=-0.7, y=2.25, s="(a)  All-CNN architecture", weight="bold", fontsize=11)
architecture.text(x=-0.35, y=0.65, s="RGB\ninput", ha="center", va="center")
architecture.text(x=-0.35, y=-0.15, s="3 × 32²", ha="center", fontsize=8)
index: int
channels: int
for index, channels in enumerate(CHANNELS):
    center: float = index + 0.65
    height: float = 0.65 + SIZES[index] / 32 * 0.65
    architecture.add_patch(
        Rectangle(
            xy=(center - 0.35, 0.65 - height / 2),
            width=0.70,
            height=height,
            facecolor="#e7eef3" if index in (2, 5) else "#f4f4f4",
            edgecolor="#333333",
            linewidth=0.8,
        )
    )
    architecture.annotate(
        text="",
        xy=(center - 0.36, 0.65),
        xytext=(center - 0.63, 0.65),
        arrowprops={"arrowstyle": "->", "lw": 0.7},
    )
    architecture.text(x=center, y=1.62, s=f"Conv {index + 1}", ha="center", fontsize=8)
    architecture.text(
        x=center,
        y=0.65,
        s=f"{'3 × 3' if index < 7 else '1 × 1'}\n{channels}",
        ha="center",
        va="center",
        linespacing=1.7,
    )
    architecture.text(x=center, y=-0.15, s=f"{channels} × {SIZES[index]}²", ha="center", fontsize=8)
    if index in (2, 5):
        architecture.text(x=center, y=1.95, s="stride 2", ha="center", fontsize=8, color=COLORS[1])
architecture.annotate(
    text="", xy=(9.35, 0.65), xytext=(9.04, 0.65), arrowprops={"arrowstyle": "->", "lw": 0.7}
)
architecture.text(x=9.82, y=0.65, s="Global\naverage pool", ha="center", va="center")
architecture.annotate(
    text="", xy=(10.85, 0.65), xytext=(10.35, 0.65), arrowprops={"arrowstyle": "->", "lw": 0.7}
)
architecture.text(x=11.2, y=0.65, s="10\nscores", ha="center", va="center")
architecture.text(
    x=-0.35,
    y=-0.7,
    s="Each block: convolution → batch normalization → ReLU. "
    "Tensor labels: channels × spatial size. No fully connected layers.",
    fontsize=9,
)

protocol: Any = fig.add_axes(rect=(0.08, 0.19, 0.235, 0.32))
accuracy: Any = fig.add_axes(rect=(0.405, 0.19, 0.235, 0.32))
fisher: Any = fig.add_axes(rect=(0.73, 0.19, 0.235, 0.32))
protocol.set_title(label="(b)  Training protocol", loc="left", fontsize=11, pad=14)
accuracy.set_title(label="(c)  Clear-test accuracy", loc="left", fontsize=11, pad=14)
fisher.set_title(label="(d)  Final-model Fisher", loc="left", fontsize=11, pad=14)
summary: list[dict[str, Any]] = json.loads((RESULTS / "fisher-summary.json").read_text())
deficit: int
for index, deficit in enumerate(DEFICITS):
    protocol.barh(
        y=index,
        width=CLEAR_EPOCHS,
        left=deficit,
        height=0.45,
        facecolor="white",
        edgecolor=COLORS[index],
        linewidth=1.1,
    )
    if deficit:
        protocol.barh(
            y=index,
            width=deficit,
            height=0.45,
            facecolor="#dddddd",
            edgecolor=COLORS[index],
            hatch="////",
            linewidth=0.8,
        )
    protocol.text(
        x=deficit + CLEAR_EPOCHS / 2, y=index, s="160 clear", ha="center", va="center", fontsize=8
    )
    record: dict[str, Any] = json.loads((RESULTS / f"s0-blur{deficit}.json").read_text())
    history: list[dict[str, Any]] = record["history"]
    row: dict[str, Any]
    epochs: list[int] = [row["epoch"] for row in history]
    scores: list[float] = [100 * row["accuracy"] for row in history]
    accuracy.plot(
        epochs, scores, color=COLORS[index], linewidth=1.1, label=f"{deficit} blur epochs"
    )
    accuracy.plot(epochs[-1], scores[-1], marker="o", color=COLORS[index], markersize=3)
    accuracy.annotate(
        text=f"{scores[-1]:.2f}",
        xy=(epochs[-1], scores[-1]),
        xytext=(3, 5),
        textcoords="offset points",
        fontsize=8,
        color=COLORS[index],
    )
    if deficit:
        accuracy.axvline(x=deficit, color=COLORS[index], linestyle=":", linewidth=0.7)
    fisher.plot(
        range(1, 10),
        summary[index]["layer_share_percent"],
        color=COLORS[index],
        marker=("o", "s", "^")[index],
        markersize=3,
        linewidth=1.1,
        label=f"{deficit} blur epochs",
    )
protocol.set_yticks(ticks=range(3), labels=["No blur", "40 blur", "100 blur"])
protocol.invert_yaxis()
protocol.set_xlim(left=0, right=270)
protocol.set_xticks(ticks=[0, 100, 200, 260])
protocol.set_xlabel(xlabel="Training epoch")
protocol.text(
    x=0, y=-0.31, s="Hatched: bilinear 32 → 8 → 32 blur", transform=protocol.transAxes, fontsize=8
)
accuracy.set_xlim(left=0, right=292)
accuracy.set_ylim(bottom=0, top=100)
accuracy.set_xlabel(xlabel="Training epoch")
accuracy.set_ylabel(ylabel="Top-1 accuracy (%)")
accuracy.set_xticks(ticks=[0, 100, 200, 260])
fisher.set_xticks(ticks=range(1, 10))
fisher.set_ylim(bottom=0, top=35)
fisher.set_xlabel(xlabel="Convolution block")
fisher.set_ylabel(ylabel="Share of model-Fisher trace (%)")
fisher.legend(frameon=False, fontsize=8, loc="upper left")
fig.text(
    x=0.04,
    y=0.055,
    s="CIFAR-10 · independent reconstruction · training seed 0 only. "
    "Fisher: 300 clear training probes, mean of 5 posterior-label draws.",
    fontsize=9,
)
fig.text(
    x=0.04,
    y=0.025,
    s="Final-model measurements, not Fisher trajectories during training. "
    "One seed does not establish a permanent deficit or a causal mechanism.",
    fontsize=8,
    color="#555555",
)
extension: str
for extension in ("png", "svg", "pdf"):
    fig.savefig(OUT / f"architecture.{extension}", dpi=220, facecolor="white")
plt.close(fig)
svg: Path = OUT / "architecture.svg"
svg.write_text("\n".join(line.rstrip() for line in svg.read_text().splitlines()) + "\n")
