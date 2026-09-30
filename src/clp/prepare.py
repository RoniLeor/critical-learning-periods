"""Verify full official CIFAR-10, audit duplicate pixels, and cache exact blur tensors."""

import argparse
import hashlib
import json
from pathlib import Path

import PIL
import torch
from torch import Tensor
from torchvision.datasets import CIFAR10

from .core import Blur

parser: argparse.ArgumentParser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--data", type=Path, required=True)
args: argparse.Namespace = parser.parse_args()
args.data.mkdir(parents=True, exist_ok=True)
training: CIFAR10 = CIFAR10(root=str(args.data), train=True, download=True)
testing: CIFAR10 = CIFAR10(root=str(args.data), train=False, download=True)
assert len(training) == 50000 and len(testing) == 10000
train_hashes: set[str] = {hashlib.sha256(image.tobytes()).hexdigest() for image in training.data}
test_hashes: set[str] = {hashlib.sha256(image.tobytes()).hexdigest() for image in testing.data}
state: dict[str, Tensor] = {
    "train": torch.from_numpy(training.data).permute(0, 3, 1, 2).contiguous(),
    "blurred": Blur()(images=training.data),
    "labels": torch.tensor(training.targets, dtype=torch.long),
    "test": torch.from_numpy(testing.data).permute(0, 3, 1, 2).contiguous(),
    "test_labels": torch.tensor(testing.targets, dtype=torch.long),
}
torch.save(obj=state, f=args.data / "tensors.pt")
source: dict[str, object] = {
    "url": CIFAR10.url,
    "published_md5": CIFAR10.tgz_md5,
    "archive_sha256": hashlib.sha256((args.data / CIFAR10.filename).read_bytes()).hexdigest(),
    "train": len(training),
    "test": len(testing),
    "classes": training.classes,
    "exact_shared_pixel_hashes": len(train_hashes & test_hashes),
    "split_policy": "official splits preserved, no deduplication or subsampling",
    "pillow": PIL.__version__,
}
(args.data / "source.json").write_text(json.dumps(source, indent=2))
print(json.dumps(source, indent=2))
