"""Lightning DataModule for ImpactMeshDataset.

No config framework, no built-in split logic -- pass in a list of patch_ids
per split (or none, to use every available patch in one dataset). Splitting
is left to you: a plain random split, k-fold, or your own leakage-aware
grouping all work the same way, since this module just filters the metadata
file down to the ids you give it.

    dm = ImpactMeshDataModule(
        root="/path/to/ImpactMesh-X/Flood",
        task="flood",
        train_ids=train_ids, val_ids=val_ids, test_ids=test_ids,
    )
    trainer.fit(model, datamodule=dm)
"""
from typing import Optional

import torch
from lightning.pytorch import LightningDataModule
from torch.utils.data import DataLoader, default_collate

from dataset import ImpactMeshDataset


def metadata_collate(batch: list[dict]) -> dict:
    """Collate that pops `metadata` and lists it instead of stacking, since
    it holds per-sample things (id, timestamps) of varying length/type."""
    metadata = [b.pop("metadata") for b in batch]
    out = default_collate(batch)
    out["metadata"] = metadata
    return out


class ImpactMeshDataModule(LightningDataModule):
    def __init__(
        self,
        root: str,
        task: str,
        train_ids: Optional[list[str]] = None,
        val_ids: Optional[list[str]] = None,
        test_ids: Optional[list[str]] = None,
        modalities: list[str] = ["S2L2A", "S1RTC", "DEM"],
        s2_version: str = "clear",
        pool: Optional[str] = None,
        batch_size: int = 8,
        num_workers: int = 8,
        pin_memory: bool = True,
        train_transforms: Optional[callable] = None,
        val_transforms: Optional[callable] = None,
        test_transforms: Optional[callable] = None,
    ):
        super().__init__()
        self.root = root
        self.task = task
        self.train_ids = train_ids
        self.val_ids = val_ids
        self.test_ids = test_ids
        self.modalities = modalities
        self.s2_version = s2_version
        self.pool = pool
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory if torch.cuda.is_available() else False
        self.train_transforms = train_transforms
        self.val_transforms = val_transforms
        self.test_transforms = test_transforms

    def _make_dataset(self, patch_ids, transforms):
        return ImpactMeshDataset(
            root=self.root, task=self.task, patch_ids=patch_ids,
            modalities=self.modalities, s2_version=self.s2_version,
            pool=self.pool, transforms=transforms,
        )

    def setup(self, stage: Optional[str] = None):
        self.train_dataset = self._make_dataset(self.train_ids, self.train_transforms)
        self.val_dataset = self._make_dataset(self.val_ids, self.val_transforms)
        self.test_dataset = self._make_dataset(self.test_ids, self.test_transforms)
        print(f"[ImpactMesh-{self.task}] Train={len(self.train_dataset)} "
              f"Val={len(self.val_dataset)} Test={len(self.test_dataset)}")

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset, batch_size=self.batch_size, shuffle=True,
            num_workers=self.num_workers, pin_memory=self.pin_memory,
            drop_last=True, collate_fn=metadata_collate,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset, batch_size=self.batch_size, shuffle=False,
            num_workers=self.num_workers, pin_memory=self.pin_memory,
            collate_fn=metadata_collate,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset, batch_size=self.batch_size, shuffle=False,
            num_workers=self.num_workers, pin_memory=self.pin_memory,
            collate_fn=metadata_collate,
        )
