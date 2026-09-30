"""Focused tests of the actual paper reconstruction primitives."""

from pathlib import Path

import numpy as np
import pytest
import torch
from torch import Tensor, nn

from clp.core import AllCNN, Blur, Checkpoint, Epoch, Schedule, Transform
from clp.fisher import FisherTrace
from clp.metrics import Predict


def test_architecture() -> None:
    """Verify every published layer and that real forward/backward passes work."""
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(3)
        model: nn.Sequential = AllCNN()()
        convolutions: list[nn.Conv2d] = [layer for layer in model if isinstance(layer, nn.Conv2d)]
        assert [layer.out_channels for layer in convolutions] == [
            96,
            96,
            192,
            192,
            192,
            192,
            192,
            192,
            10,
        ]
        assert [layer.stride for layer in convolutions] == [
            (1, 1),
            (1, 1),
            (2, 2),
            (1, 1),
            (1, 1),
            (2, 2),
            (1, 1),
            (1, 1),
            (1, 1),
        ]
        assert [layer.kernel_size for layer in convolutions] == [(3, 3)] * 7 + [(1, 1)] * 2
        assert sum(isinstance(layer, nn.BatchNorm2d) for layer in model) == 9
        assert sum(isinstance(layer, nn.ReLU) for layer in model) == 9
        output: Tensor = model(torch.randn(size=(2, 3, 32, 32)))
        assert output.shape == (2, 10)
        output.sum().backward()
        assert all(parameter.grad is not None for parameter in model.parameters())


@pytest.mark.parametrize("deficit", [0, 40, 100])
def test_schedule(deficit: int) -> None:
    """All conditions have exactly 160 clear epochs and never reset their rate."""
    schedule: Schedule = Schedule(deficit=deficit)
    states: list[tuple[bool, float]] = [schedule(epoch=index) for index in range(deficit + 160)]
    assert sum(blurred for blurred, _ in states) == deficit
    assert sum(not blurred for blurred, _ in states) == 160
    assert states[0][1] == 0.05
    assert states[deficit][0] is False
    assert states[deficit][1] == pytest.approx(0.05 * 0.97**deficit)
    if deficit:
        assert states[deficit - 1][0] is True
        assert states[deficit][1] == pytest.approx(states[deficit - 1][1] * 0.97)
    with pytest.raises(ValueError, match="outside"):
        schedule(epoch=deficit + 160)
    with pytest.raises(ValueError, match="outside"):
        schedule(epoch=-1)


def test_blur() -> None:
    """Uniform color survives and high-frequency detail is removed without mutation."""
    pixels: np.ndarray = np.full(shape=(2, 32, 32, 3), fill_value=128, dtype=np.uint8)
    pixels[1, :, ::2] = 255
    pixels[1, :, 1::2] = 0
    original: np.ndarray = pixels.copy()
    result: Tensor = Blur()(images=pixels)
    assert result.shape == (2, 3, 32, 32)
    assert result.dtype == torch.uint8
    assert torch.all(result[0] == 128)
    assert result[1].float().std() < 10
    assert np.array_equal(pixels, original)


def test_transform() -> None:
    """Normalization, augmentation, reproducibility, and RNG isolation are checked."""
    images: Tensor = (
        torch.arange(3 * 32 * 32)
        .remainder(256)
        .to(torch.uint8)
        .reshape(1, 3, 32, 32)
        .repeat(3, 1, 1, 1)
    )
    original: Tensor = images.clone()
    rng: Tensor = torch.get_rng_state().clone()
    clean: Tensor = Transform()(images=images)
    augmented: Tensor = Transform()(images=images, seed=123)
    assert torch.equal(rng, torch.get_rng_state())
    assert torch.equal(images, original)
    assert augmented.shape == clean.shape
    assert torch.equal(augmented, Transform()(images=images, seed=123))
    assert not torch.equal(augmented, clean)
    assert not torch.equal(augmented, Transform()(images=images, seed=124))
    assert clean[0, 0, 0, 0].item() == pytest.approx(-0.4914 / 0.2023)
    assert clean[0, 1, 0, 0].item() == pytest.approx(-0.4822 / 0.1994)


def test_epoch_checkpoint(tmp_path: Path) -> None:
    """An actual optimizer update resumes identically from the production checkpoint."""
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(12)
        images: Tensor = torch.randint(high=256, size=(7, 3, 32, 32), dtype=torch.uint8)
        labels: Tensor = torch.tensor([0, 1, 2, 0, 1, 2, 0])
        original: Tensor = images.clone()
        classifier: nn.Linear = nn.Linear(in_features=3072, out_features=3)
        model: nn.Sequential = nn.Sequential(nn.Flatten(), classifier)
        optimizer: torch.optim.SGD = torch.optim.SGD(
            params=model.parameters(), lr=0.001, momentum=0.9
        )
        epoch: Epoch = Epoch(model=model, optimizer=optimizer, device=torch.device("cpu"), batch=3)
        before: Tensor = classifier.weight.detach().clone()
        loss: float = epoch(images=images, labels=labels, seed=10)
        assert np.isfinite(loss) and loss > 0
        assert not torch.equal(before, classifier.weight)
        assert torch.equal(images, original)
        path: Path = tmp_path / "latest.pt"
        checkpoint: Checkpoint = Checkpoint(path=path)
        checkpoint(
            state={"model": model.state_dict(), "optimizer": optimizer.state_dict(), "epoch": 1}
        )
        saved: dict = torch.load(f=path, weights_only=True)
        expected: float = epoch(images=images, labels=labels, seed=11)
        weights: Tensor = classifier.weight.detach().clone()
        model.load_state_dict(saved["model"])
        optimizer.load_state_dict(saved["optimizer"])
        assert epoch(images=images, labels=labels, seed=11) == expected
        assert torch.equal(classifier.weight, weights)
        checkpoint(state={"epoch": 2})
        assert torch.load(f=path, weights_only=True) == {"epoch": 2}
        assert not path.with_suffix(".tmp").exists()


def test_fisher() -> None:
    """For balanced binary zero logits, sampled Fisher equals the exact analytic trace."""
    with torch.random.fork_rng():
        torch.manual_seed(7)
        model: nn.Module = nn.Linear(in_features=2, out_features=2, bias=False)
        nn.init.zeros_(model.weight)
        images: Tensor = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        before: Tensor = model.weight.detach().clone()
        model.weight.grad = torch.ones_like(model.weight)
        state: Tensor = torch.random.get_rng_state().clone()
        trace: dict[str, float] = FisherTrace(model=model, device=torch.device("cpu"))(
            images=images,
            seed=5,
        )
        # Each example has trace 0.5 * ||x||², averaging to 7.5.
        assert trace["weight"] == pytest.approx(7.5)
        assert torch.equal(model.weight, before)
        assert torch.equal(model.weight.grad, torch.ones_like(model.weight))
        assert torch.equal(state, torch.random.get_rng_state())
        assert model.training


def test_prediction() -> None:
    """Batched prediction restores train/eval mode and leaves parameters untouched."""
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(91)
        model: nn.Linear = nn.Linear(in_features=2, out_features=2)
        images: Tensor = torch.ones(size=(7, 2))
        before: Tensor = model.weight.detach().clone()
        predictor: Predict = Predict(model=model, device=torch.device("cpu"), batch=3)
        expected: Tensor = model(images).argmax(dim=1)
        assert torch.equal(predictor(images=images), expected)
        assert model.training
        assert torch.equal(model.weight, before)
        assert model.weight.grad is None
        model.eval()
        assert torch.equal(predictor(images=images), expected)
        assert not model.training
