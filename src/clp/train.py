"""Run or resume the full-data reconstruction; benchmark mode never produces study results."""

import argparse
import hashlib
import json
from pathlib import Path
from time import perf_counter
from typing import Any

import torch
from torch import Tensor, nn

from .core import (
    BATCH,
    CLEAR_EPOCHS,
    DECAY,
    INITIAL_LR,
    MEAN,
    STD,
    WEIGHT_DECAY,
    AllCNN,
    Checkpoint,
    Epoch,
    Schedule,
    Transform,
)
from .metrics import Predict

parser: argparse.ArgumentParser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--data", type=Path, required=True)
parser.add_argument("--out", type=Path, required=True)
parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
parser.add_argument("--deficits", type=int, nargs="+", default=[0, 40, 100])
parser.add_argument("--benchmark", action="store_true")
parser.add_argument("--device", choices=["cpu", "mps", "cuda"])
args: argparse.Namespace = parser.parse_args()
args.out.mkdir(parents=True, exist_ok=True)
torch.set_num_threads(4)
device: torch.device = torch.device(
    args.device
    or (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
)
data: dict[str, Tensor] = torch.load(
    f=args.data / "tensors.pt", map_location="cpu", weights_only=True
)
if len(data["train"]) != 50000 or len(data["test"]) != 10000:
    raise ValueError("Full official CIFAR-10 is required")
source: dict[str, Any] = json.loads((args.data / "source.json").read_text())
config: dict[str, Any] = {
    "dataset": source,
    "architecture": "Appendix A All-CNN reconstruction",
    "batch": BATCH,
    "clear_epochs": CLEAR_EPOCHS,
    "learning_rate": INITIAL_LR,
    "decay": DECAY,
    "weight_decay": WEIGHT_DECAY,
    "momentum": 0.0,
    "learning_rate_reset": False,
    "conv_bias": True,
    "padding": "same",
    "blocks": "Conv-BatchNorm-ReLU including final 1x1 block; global average pool; logits",
    "initialization": "PyTorch default Conv2d and BatchNorm2d",
    "blur": "PIL bilinear 32-to-8-to-32 before augmentation",
    "augmentation": "zero pad 4, uniform crop 32, horizontal flip p=0.5",
    "normalization_mean": MEAN,
    "normalization_std": STD,
    "torch": torch.__version__,
    "device": str(device),
    "code_sha256": {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(Path(__file__).parent.glob("*.py"))
    },
}
config_path: Path = args.out / "config.json"
serialized: str = json.dumps(config, indent=2)
if config_path.exists() and config_path.read_text() != serialized:
    raise ValueError("Configuration differs from existing run; use a separate output directory")
config_path.write_text(serialized)
seed: int
for seed in args.seeds:
    deficit: int
    for deficit in args.deficits:
        name: str = f"s{seed}-blur{deficit}"
        checkpoint_path: Path = args.out / f"{name}.pt"
        torch.manual_seed(seed)
        model: nn.Sequential = AllCNN()().to(device)
        optimizer: torch.optim.SGD = torch.optim.SGD(
            params=model.parameters(), lr=INITIAL_LR, weight_decay=WEIGHT_DECAY, momentum=0.0
        )
        trainer: Epoch = Epoch(model=model, optimizer=optimizer, device=device)
        if args.benchmark:
            warm_loss: float = trainer(
                images=data["train"][:BATCH], labels=data["labels"][:BATCH], seed=seed
            )
            started: float = perf_counter()
            measured_loss: float = trainer(
                images=data["train"][:1024], labels=data["labels"][:1024], seed=seed
            )
            seconds: float = perf_counter() - started
            benchmark: dict[str, object] = {
                "images": 1024,
                "seconds": seconds,
                "loss": measured_loss,
                "estimated_train_epoch_seconds": seconds * 50000 / 1024,
                "estimated_1860_epoch_training_hours": seconds * 50000 / 1024 * 1860 / 3600,
                "note": "Timing estimate only; excludes evaluation/checkpoint overhead; not a study result",
            }
            (args.out / "benchmark.json").write_text(json.dumps(benchmark, indent=2))
            print(json.dumps(benchmark), flush=True)
            raise SystemExit(0)
        history: list[dict[str, Any]] = []
        if checkpoint_path.exists():
            saved: dict[str, Any] = torch.load(
                f=checkpoint_path, map_location="cpu", weights_only=False
            )
            if saved["config"] != config or saved["seed"] != seed or saved["deficit"] != deficit:
                raise ValueError("Checkpoint identity mismatch")
            model.load_state_dict(saved["model"])
            optimizer.load_state_dict(saved["optimizer"])
            torch.set_rng_state(saved["rng"])
            if device.type == "mps":
                torch.mps.set_rng_state(saved["mps_rng"])
            if device.type == "cuda":
                torch.cuda.set_rng_state(saved["cuda_rng"])
            history = saved["history"]
        schedule: Schedule = Schedule(deficit=deficit)
        test_inputs: Tensor = Transform()(images=data["test"])
        predictor: Predict = Predict(model=model, device=device, batch=BATCH)
        epoch: int
        for epoch in range(len(history), deficit + CLEAR_EPOCHS):
            started = perf_counter()
            blurred: bool
            rate: float
            blurred, rate = schedule(epoch=epoch)
            optimizer.param_groups[0]["lr"] = rate
            loss: float = trainer(
                images=data["blurred"] if blurred else data["train"],
                labels=data["labels"],
                seed=seed * 1000000 + epoch * 1000,
            )
            predictions: Tensor = predictor(images=test_inputs)
            accuracy: float = float((predictions == data["test_labels"]).float().mean())
            row: dict[str, Any] = {
                "epoch": epoch + 1,
                "blurred": blurred,
                "lr": rate,
                "loss": loss,
                "accuracy": accuracy,
                "seconds": perf_counter() - started,
            }
            history.append(row)
            state: dict[str, Any] = {
                "config": config,
                "seed": seed,
                "deficit": deficit,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "rng": torch.get_rng_state(),
                "mps_rng": torch.mps.get_rng_state() if device.type == "mps" else None,
                "cuda_rng": torch.cuda.get_rng_state() if device.type == "cuda" else None,
                "history": history,
                "test_predictions": predictions,
                "test_labels": data["test_labels"],
            }
            Checkpoint(path=checkpoint_path)(state=state)
            (args.out / f"{name}.json").write_text(
                json.dumps(
                    {
                        "seed": seed,
                        "deficit": deficit,
                        "complete": len(history) == deficit + CLEAR_EPOCHS,
                        "history": history,
                        "test_predictions": predictions.tolist(),
                        "test_labels": data["test_labels"].tolist(),
                    },
                    indent=2,
                )
            )
            print(json.dumps({"run": name, **row}), flush=True)
        print(
            json.dumps({"completed": name, "final_accuracy": history[-1]["accuracy"]}), flush=True
        )
