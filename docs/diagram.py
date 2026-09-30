"""Render the repository's research architecture figure from editable vector primitives."""

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

OUT: Path = Path(__file__).resolve().parent
NAVY: str = "#101e31"
PANEL: str = "#1b2e46"
INK: str = "#edf5ff"
MUTED: str = "#a3b8cf"
TEAL: str = "#64ddcc"
ORANGE: str = "#ffbd75"
fig: Any
axis: Any
fig, axis = plt.subplots(figsize=(15.2, 10))
fig.patch.set_facecolor(NAVY)
axis.set_facecolor(NAVY)
axis.set_xlim(0, 1520)
axis.set_ylim(1000, 0)
axis.axis("off")
axis.text(55, 56, "CRITICAL LEARNING PERIODS", color=TEAL, fontsize=12, weight="bold")
axis.text(55, 112, "Does early blur leave a lasting mark?", color=INK, fontsize=29, weight="bold")
axis.text(
    55,
    155,
    "CIFAR-10  /  All-CNN reconstruction  /  accuracy + final-model Fisher sensitivity",
    color=MUTED,
    fontsize=12,
)
cards: list[tuple[str, str, str]] = [
    ("INPUT", "RGB image", "3 × 32 × 32"),
    ("BLOCKS 1–3", "3×3 conv: 96, 96, 192\nLast convolution: stride 2", "192 × 16 × 16"),
    ("BLOCKS 4–6", "3×3 conv: 192, 192, 192\nLast convolution: stride 2", "192 × 8 × 8"),
    ("BLOCKS 7–9", "3×3: 192 → 1×1: 192\n→ 1×1: 10 class maps", "10 × 8 × 8"),
    ("READOUT", "Global average pooling\nNo fully connected layer", "10 class scores"),
]
index: int
card: tuple[str, str, str]
for index, card in enumerate(cards):
    x: int = 55 + index * 290
    axis.add_patch(
        FancyBboxPatch(
            xy=(x, 205),
            width=250,
            height=205,
            boxstyle="round,pad=0,rounding_size=14",
            facecolor=PANEL,
            edgecolor="#304965",
            linewidth=1.1,
        )
    )
    axis.text(x + 20, 240, card[0], color=TEAL, fontsize=10, weight="bold")
    axis.text(x + 20, 292, card[1], color=INK, fontsize=11, linespacing=1.7)
    axis.text(x + 20, 380, card[2], color=ORANGE, fontsize=13, weight="bold")
    if index < 4:
        axis.add_patch(
            FancyArrowPatch(
                posA=(x + 258, 305),
                posB=(x + 282, 305),
                arrowstyle="-|>",
                mutation_scale=15,
                color=MUTED,
                linewidth=1.5,
            )
        )
axis.text(
    55,
    450,
    "Each convolution block: Conv → BatchNorm → ReLU. No skip connections. This is a CNN, not an MLP.",
    color=MUTED,
    fontsize=11,
)
axis.text(
    55,
    515,
    "SAME CLEAR TRAINING. DIFFERENT EARLY EXPERIENCE.",
    color=INK,
    fontsize=14,
    weight="bold",
)
condition: int
deficit: int
for condition, deficit in enumerate([0, 40, 100]):
    y: int = 550 + 78 * condition
    x = 375
    scale: float = 3.8
    axis.text(
        55,
        y + 27,
        ["Clear baseline", "Blur removed at 40", "Blur removed at 100"][condition],
        color=INK,
        fontsize=12,
    )
    if deficit:
        axis.add_patch(
            FancyBboxPatch(
                xy=(x, y),
                width=deficit * scale,
                height=46,
                boxstyle="round,pad=0,rounding_size=5",
                facecolor=ORANGE,
                edgecolor="none",
            )
        )
        axis.text(
            x + deficit * scale / 2,
            y + 29,
            f"{deficit} blur",
            ha="center",
            color=NAVY,
            fontsize=12,
            weight="bold",
        )
    axis.add_patch(
        FancyBboxPatch(
            xy=(x + deficit * scale, y),
            width=160 * scale,
            height=46,
            boxstyle="round,pad=0,rounding_size=5",
            facecolor=TEAL,
            edgecolor="none",
        )
    )
    axis.text(
        x + (deficit + 80) * scale,
        y + 29,
        "160 clear epochs",
        ha="center",
        color=NAVY,
        fontsize=12,
        weight="bold",
    )
    axis.text(
        x + (deficit + 160) * scale + 14, y + 29, str(deficit + 160), color=MUTED, fontsize=11
    )
axis.text(
    375,
    785,
    "Blur: bilinear 32 → 8 → 32. All 50,000 training images; evaluate on 10,000 clear test images.",
    color=MUTED,
    fontsize=11,
)
axis.add_patch(
    FancyBboxPatch(
        xy=(55, 830),
        width=1410,
        height=110,
        boxstyle="round,pad=0,rounding_size=12",
        facecolor=PANEL,
        edgecolor="none",
    )
)
axis.text(80, 870, "ACCURACY", color=TEAL, fontsize=11, weight="bold")
axis.text(80, 911, "Does a performance deficit remain?", color=INK, fontsize=14)
axis.text(750, 870, "FISHER SENSITIVITY", color=ORANGE, fontsize=11, weight="bold")
axis.text(750, 911, "Which layers' weights affect predictions most?", color=INK, fontsize=14)
axis.text(
    55,
    976,
    "Independent reconstruction of Achille, Rovere & Soatto (ICLR 2019). Assumptions and estimator differences are documented.",
    color=MUTED,
    fontsize=10,
)
fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
fig.savefig(OUT / "architecture.png", dpi=150, facecolor=NAVY)
fig.savefig(OUT / "architecture.svg", facecolor=NAVY)
plt.close(fig)
