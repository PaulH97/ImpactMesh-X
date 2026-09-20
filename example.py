"""Minimal usage example: load a hazard with the official split, build a
DataModule, and look at one sample. Run after downloading the Flood (or Fire)
release folder.

    python example.py --root /path/to/ImpactMesh-X/Flood --task flood
"""
import argparse
import json
import warnings

# Harmless: PyTorch tries to init NVML for GPU info even on CPU-only/driverless
# nodes and warns when that fails. Does not affect anything below.
warnings.filterwarnings("ignore", message="Can't initialize NVML")

from datamodule import ImpactMeshDataModule
from transforms import Compose, Normalize, RandomHorizontalFlip, RandomVerticalFlip
from utils import plot_sample, print_metadata

SPLIT_FILE = {"flood": "splits/flood_splits.json", "fire": "splits/fire_splits.json"}
NORM_FILE = {"flood": "splits/flood_norm.json", "fire": "splits/fire_norm.json"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="path to the release folder for one hazard")
    ap.add_argument("--task", required=True, choices=["flood", "fire"])
    args = ap.parse_args()

    # Official split, positive (hazard-present) samples only -- see README.
    with open(SPLIT_FILE[args.task]) as f:
        split = json.load(f)

    train_transforms = Compose([
        RandomHorizontalFlip(), RandomVerticalFlip(),
        Normalize(NORM_FILE[args.task]),
    ])

    dm = ImpactMeshDataModule(
        root=args.root, task=args.task, pool="positive",
        train_ids=split["train"], val_ids=split["val"], test_ids=split["test"],
        train_transforms=train_transforms,
        batch_size=4, num_workers=2,
    )
    dm.setup()

    sample = dm.train_dataset[0]
    print_metadata(sample, task=args.task)
    plot_sample(sample, task=args.task, save_path="example_sample.png")
    print("\nSaved example_sample.png")


if __name__ == "__main__":
    main()
