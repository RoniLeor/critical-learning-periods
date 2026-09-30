"""Batched predictions that preserve the model training mode."""

from dataclasses import dataclass

import torch
from torch import Tensor, nn


@dataclass
class Predict:
    """Evaluate without gradients and preserve the caller's model training mode."""

    model: nn.Module
    device: torch.device
    batch: int

    def __call__(self, *, images: Tensor) -> Tensor:
        was_training: bool = self.model.training
        self.model.eval()
        predictions: list[Tensor] = []
        start: int
        try:
            with torch.inference_mode():
                for start in range(0, len(images), self.batch):
                    predictions.append(
                        self.model(images[start : start + self.batch].to(self.device))
                        .argmax(dim=1)
                        .cpu(),
                    )
        finally:
            self.model.train(mode=was_training)
        return torch.cat(predictions)
