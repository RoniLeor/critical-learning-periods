"""Paper architecture, explicit schedule, image transform, and resumable epoch steps."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from torch import Tensor, nn
from torch.nn import functional

CHANNELS: tuple[int, ...] = (96, 96, 192, 192, 192, 192, 192, 192, 10)
STRIDES: tuple[int, ...] = (1, 1, 2, 1, 1, 2, 1, 1, 1)
KERNELS: tuple[int, ...] = (3, 3, 3, 3, 3, 3, 3, 1, 1)
MEAN: tuple[float, ...] = (0.4914, 0.4822, 0.4465)
STD: tuple[float, ...] = (0.2023, 0.1994, 0.2010)
CLEAR_EPOCHS: int = 160
INITIAL_LR: float = 0.05
DECAY: float = 0.97
WEIGHT_DECAY: float = 0.001
BATCH: int = 128
SIZE: int = 32
BLUR_SIZE: int = 8
PADDING: int = 4


@dataclass(frozen=True)
class AllCNN:
    """Build Appendix A's nine Conv-BatchNorm-ReLU blocks with global pooling."""

    def __call__(self) -> nn.Sequential:
        blocks: list[nn.Module] = []
        incoming: int = 3
        outgoing: int
        kernel: int
        stride: int
        for outgoing, kernel, stride in zip(CHANNELS, KERNELS, STRIDES, strict=True):
            blocks.extend(
                [
                    nn.Conv2d(
                        in_channels=incoming,
                        out_channels=outgoing,
                        kernel_size=kernel,
                        stride=stride,
                        padding=kernel // 2,
                        bias=True,
                    ),
                    nn.BatchNorm2d(num_features=outgoing),
                    nn.ReLU(),
                ]
            )
            incoming = outgoing
        blocks.extend([nn.AdaptiveAvgPool2d(output_size=1), nn.Flatten()])
        return nn.Sequential(*blocks)


@dataclass(frozen=True)
class Schedule:
    """Zero-based epochs; decay continues through blur removal without reset."""

    deficit: int

    def __call__(self, *, epoch: int) -> tuple[bool, float]:
        if self.deficit < 0 or not 0 <= epoch < self.deficit + CLEAR_EPOCHS:
            raise ValueError("Epoch outside deficit plus 160 clear epochs")
        return epoch < self.deficit, INITIAL_LR * DECAY**epoch


@dataclass(frozen=True)
class Blur:
    """Apply PIL bilinear 32-to-8-to-32 resizing before data augmentation."""

    def __call__(self, *, images: np.ndarray) -> Tensor:
        output: np.ndarray = np.empty_like(images)
        index: int
        pixels: np.ndarray
        for index, pixels in enumerate(images):
            small: Image.Image = Image.fromarray(pixels).resize(
                size=(BLUR_SIZE, BLUR_SIZE), resample=Image.Resampling.BILINEAR
            )
            output[index] = np.asarray(
                small.resize(size=(SIZE, SIZE), resample=Image.Resampling.BILINEAR)
            )
        return torch.from_numpy(output).permute(0, 3, 1, 2).contiguous()


@dataclass(frozen=True)
class Transform:
    """Seeded zero-padded random crop/flip, then explicit CIFAR normalization."""

    def __call__(self, *, images: Tensor, seed: int | None = None) -> Tensor:
        pixels: Tensor = images
        count: int = len(images)
        if seed is not None:
            generator: torch.Generator = torch.Generator().manual_seed(seed)
            offsets: Tensor = torch.randint(
                high=2 * PADDING + 1, size=(count, 2), generator=generator
            )
            padded: Tensor = functional.pad(input=pixels, pad=(PADDING,) * 4)
            rows: Tensor = offsets[:, 0, None, None] + torch.arange(SIZE)[None, :, None]
            columns: Tensor = offsets[:, 1, None, None] + torch.arange(SIZE)[None, None, :]
            pixels = padded.permute(0, 2, 3, 1)[
                torch.arange(count)[:, None, None], rows, columns
            ].permute(0, 3, 1, 2)
            flips: Tensor = torch.rand(size=(count,), generator=generator) < 0.5
            pixels = torch.where(flips[:, None, None, None], pixels.flip(dims=[3]), pixels)
        mean: Tensor = torch.tensor(MEAN).reshape(1, 3, 1, 1)
        std: Tensor = torch.tensor(STD).reshape(1, 3, 1, 1)
        return (pixels.float() / 255.0 - mean) / std


@dataclass
class Epoch:
    """One full shuffled epoch; no image subsampling or mixed precision."""

    model: nn.Module
    optimizer: torch.optim.Optimizer
    device: torch.device
    batch: int = BATCH

    def __call__(self, *, images: Tensor, labels: Tensor, seed: int) -> float:
        self.model.train()
        order: Tensor = torch.randperm(len(labels), generator=torch.Generator().manual_seed(seed))
        total: Tensor = torch.zeros(size=(), device=self.device)
        start: int
        for start in range(0, len(labels), self.batch):
            indices: Tensor = order[start : start + self.batch]
            inputs: Tensor = Transform()(images=images[indices], seed=seed + start).to(self.device)
            target: Tensor = labels[indices].to(self.device)
            self.optimizer.zero_grad(set_to_none=True)
            loss: Tensor = functional.cross_entropy(input=self.model(inputs), target=target)
            loss.backward()
            self.optimizer.step()
            total += loss.detach() * len(indices)
        return float(total.cpu()) / len(labels)


@dataclass(frozen=True)
class Checkpoint:
    """Atomically replace the resumable state, preserving the last completed epoch."""

    path: Path

    def __call__(self, *, state: dict[str, Any]) -> None:
        temporary: Path = self.path.with_suffix(".tmp")
        torch.save(obj=state, f=temporary)
        temporary.replace(self.path)
