"""Monte Carlo model-Fisher traces, using per-example sampled-label gradients."""

from dataclasses import dataclass

import torch
from torch import Tensor, nn


@dataclass
class FisherTrace:
    """Measure stage traces without modifying parameters, buffers, or existing gradients."""

    model: nn.Module
    device: torch.device

    def __call__(self, *, images: Tensor, seed: int) -> dict[str, float]:
        named: list[tuple[str, nn.Parameter]] = list(self.model.named_parameters())
        parameters: tuple[nn.Parameter, ...] = tuple(parameter for _, parameter in named)
        names: list[str] = [name.split(".")[0] for name, _ in named]
        totals: dict[str, Tensor] = {name: torch.zeros((), device=self.device) for name in names}
        generator: torch.Generator = torch.Generator().manual_seed(seed)
        was_training: bool = self.model.training
        self.model.eval()
        sample: Tensor
        try:
            for sample in images:
                log_probabilities: Tensor = self.model(
                    sample.unsqueeze(dim=0).to(self.device)
                ).log_softmax(dim=1)
                sampled_label: Tensor = torch.multinomial(
                    log_probabilities.detach().exp().cpu(), num_samples=1, generator=generator
                ).to(self.device)
                score: Tensor = log_probabilities.gather(dim=1, index=sampled_label).sum()
                gradients: tuple[Tensor, ...] = torch.autograd.grad(
                    outputs=score, inputs=parameters
                )
                name: str
                gradient: Tensor
                for name, gradient in zip(names, gradients, strict=True):
                    totals[name] += gradient.detach().square().sum()
        finally:
            self.model.train(mode=was_training)
        return {name: value.item() / len(images) for name, value in totals.items()}
