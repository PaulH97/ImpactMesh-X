"""Simple sample transforms for ImpactMeshDataset: spatial augmentations and
band normalization. Compose them in a list and apply in order, e.g.:

    from norm_stats import NORM_STATS
    transforms = Compose([RandomHorizontalFlip(), RandomVerticalFlip(),
                           Normalize(NORM_STATS["flood"])])
    ds = ImpactMeshDataset(..., transforms=transforms)
"""
import json
import random
from pathlib import Path

import torch


class Compose:
    """Apply a list of sample transforms in order."""

    def __init__(self, transforms: list[callable]):
        self.transforms = transforms

    def __call__(self, sample: dict) -> dict:
        for t in self.transforms:
            sample = t(sample)
        return sample


def _apply_spatial(sample: dict, fn) -> dict:
    """Apply a spatial op to every modality tensor and the mask, so
    augmentations never desync the image from its label."""
    for mod, x in sample["image"].items():
        sample["image"][mod] = fn(x)
    if isinstance(sample.get("mask"), torch.Tensor):
        sample["mask"] = fn(sample["mask"])
    return sample


class RandomHorizontalFlip:
    """Flip the width dimension with probability `p`."""

    def __init__(self, p: float = 0.5):
        self.p = p

    def __call__(self, sample: dict) -> dict:
        if random.random() >= self.p:
            return sample
        return _apply_spatial(sample, lambda x: x.flip(-1))


class RandomVerticalFlip:
    """Flip the height dimension with probability `p`."""

    def __init__(self, p: float = 0.5):
        self.p = p

    def __call__(self, sample: dict) -> dict:
        if random.random() >= self.p:
            return sample
        return _apply_spatial(sample, lambda x: x.flip(-2))


class RandomRotate90:
    """Rotate by k * 90 degrees, k sampled uniformly from {1, 2, 3}."""

    def __init__(self, p: float = 0.5):
        self.p = p

    def __call__(self, sample: dict) -> dict:
        if random.random() >= self.p:
            return sample
        k = random.randint(1, 3)
        return _apply_spatial(sample, lambda x: torch.rot90(x, k, dims=(-2, -1)))


class Normalize:
    """Per-band z-score normalization from stats {"means": {mod: [...]}, "stds": {mod: [...]}}.

    `stats` is either that dict directly (e.g. from `norm_stats.NORM_STATS`) or
    a path to a JSON file holding it. Modality keys must match `sample["image"]`
    keys (e.g. "S1RTC", "S2L2A", "DEM").
    """

    def __init__(self, stats: str | dict, eps: float = 1e-6):
        if isinstance(stats, (str, Path)):
            with open(stats) as f:
                stats = json.load(f)
        self.mean, self.std = {}, {}
        for mod, mean_list in stats["means"].items():
            self.mean[mod] = torch.tensor(mean_list, dtype=torch.float32).view(1, -1, 1, 1)
            self.std[mod] = torch.clamp(
                torch.tensor(stats["stds"][mod], dtype=torch.float32), min=eps
            ).view(1, -1, 1, 1)

    def __call__(self, sample: dict) -> dict:
        for mod, x in sample["image"].items():
            if mod not in self.mean:
                continue
            mean, std = self.mean[mod], self.std[mod]
            # DEM has no time dimension (C, H, W); the rest do (T, C, H, W).
            # Both broadcast fine against a (1, C, 1, 1) stat as long as we
            # match rank, so unsqueeze the static case up front.
            if x.dim() == 3:
                sample["image"][mod] = (x.unsqueeze(0) - mean) / std
                sample["image"][mod] = sample["image"][mod].squeeze(0)
            else:
                sample["image"][mod] = (x - mean) / std
        return sample
