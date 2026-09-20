import json
from lightning.pytorch import LightningDataModule
from torch.utils.data import DataLoader
from pathlib import Path
import pandas as pd
import torch
import warnings

from fusion.data.datasets import Sen12Landslides, ImpactMesh
from fusion.data.utils import metadata_collate

warnings.filterwarnings("ignore", message="Can't initialize NVML")


class ImpactMeshDataModule(LightningDataModule):
    def __init__(
        self,
        dataset_cfg,
        batch_size=8,
        num_workers=8,
        pin_memory=True,
        train_transforms=None,
        val_transforms=None,
        test_transforms=None,
        test_shuffle=False,
    ):
        super().__init__()
        self.root = Path(dataset_cfg.root)
        self.long_version = dataset_cfg.long_version
        self.split_json_path = Path(dataset_cfg.split_json)
        self.task = dataset_cfg.task
        self.modalities = list(dataset_cfg.get("modality_infos", {"S2L2A": {}, "S1RTC": {}, "DEM": {}}).keys())
        self.temporal_mode = dataset_cfg.get("temporal_mode", "sits")
        self.s2_version = dataset_cfg.get("s2_version")
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory if torch.cuda.is_available() else False
        self.train_transforms = train_transforms
        self.val_transforms = val_transforms
        self.test_transforms = test_transforms
        self.test_shuffle = test_shuffle

        self.paths = {
            "S2L2A": ("S2L2A", "_S2L2A.zarr.zip"),
            "S1RTC": ("S1RTC", "_S1RTC.zarr.zip"),
            "DEM":   ("DEM",   "_DEM.tif"),
            "MASK":  ("MASK",  f"_annotation_{self.task}.tif"),
        }

        if self.long_version:
            self.paths["S1RTC"] = ("S1RTC_long", "_S1RTC.zarr.zip")
            self.paths["S2L2A"] = (f"S2L2A_long/{self.s2_version}", "_S2L2A.zarr.zip")

    ANCHOR_COLS = {"pre_month": "pre_month_date", "pre": "pre_event_date",
                   "post": "event_window_date", "post_month": "post_event_date"}

    def _load_anchors(self) -> dict:
        """patch_id -> per-modality {pre_month, pre, post, post_month} acquisition
        dates that the ImpactMesh team anchored the annotation to.
        `select_timesteps` prefers these over its own cloud/date heuristic, so the
        input frames match what the label is based on -- und quad-temporal ist damit
        exakt die Originalserie und nicht nur eine Serie derselben Laenge.
        """
        hazard = "flood" if self.task == "flood" else "fire"
        meta_path = self.root / "data" / f"{hazard}_patch_meta_pos.parquet"
        if not meta_path.exists():
            warnings.warn(
                f"[ImpactMesh] anchor metadata not found at {meta_path} -- "
                "falling back to the cloud/date heuristic for all samples."
            )
            return {}
        mods = {"S2L2A": "S2", "S1RTC": "S1"}
        cols = ["patch_id"] + [f"{p}_{c}" for p in mods.values() for c in self.ANCHOR_COLS.values()]
        df = pd.read_parquet(meta_path, columns=cols)
        return {
            row["patch_id"]: {
                mod: {slot: row[f"{prefix}_{col}"] for slot, col in self.ANCHOR_COLS.items()}
                for mod, prefix in mods.items()
            }
            for row in df.to_dict("records")
        }

    def _build_file_list(self, event_ids: list[str], anchors: dict) -> list[dict]:
        """Build a list of dicts with paths for each event id."""
        files = []
        data_dir = self.root / "data"

        for eid in event_ids:
            entry = {"id": eid, "anchors": anchors.get(eid)}

            for mod in self.modalities:
                folder, ext = self.paths[mod]
                entry[mod] = str(data_dir / folder / f"{eid}{ext}")

            folder, ext = self.paths["MASK"]
            entry["MASK"] = str(data_dir / folder / f"{eid}{ext}")

            files.append(entry)
        return files

    def setup(self, stage=None):
        if not self.split_json_path.exists():
            raise FileNotFoundError(f"Split file not found: {self.split_json_path}")

        with open(self.split_json_path) as f:
            splits = json.load(f)

        anchors = self._load_anchors()
        train_files = self._build_file_list(splits.get("train", []), anchors)
        val_files = self._build_file_list(splits.get("val", []), anchors)
        test_files = self._build_file_list(splits.get("test", []), anchors)

        task_print = "Fire" if self.task == "wildfire" else self.task.capitalize()
        print(f"[ImpactMesh-{task_print}] Split: Train={len(train_files)}, Val={len(val_files)}, Test={len(test_files)}")

        common = dict(modalities=self.modalities, temporal_mode=self.temporal_mode, task=self.task)
        self.train_dataset = ImpactMesh(train_files, transforms=self.train_transforms, **common)
        self.val_dataset = ImpactMesh(val_files, transforms=self.val_transforms, **common)
        self.test_dataset = ImpactMesh(test_files, transforms=self.test_transforms, **common)

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset, batch_size=self.batch_size, shuffle=True,
            num_workers=self.num_workers, pin_memory=self.pin_memory, drop_last=True,
            collate_fn=metadata_collate, persistent_workers=self.num_workers > 0
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset, batch_size=self.batch_size, shuffle=False,
            num_workers=self.num_workers, pin_memory=self.pin_memory,
            collate_fn=metadata_collate, persistent_workers=self.num_workers > 0
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset, batch_size=self.batch_size, shuffle=self.test_shuffle,
            num_workers=self.num_workers, pin_memory=self.pin_memory,
            collate_fn=metadata_collate
        )


class Sen12LsDataModule(LightningDataModule):
    def __init__(
        self,
        dataset_cfg,
        batch_size=16,
        num_workers=8,
        pin_memory=True,
        train_transforms=None,
        val_transforms=None,
        test_transforms=None,
        test_shuffle=False,
    ):
        super().__init__()
        self.root = Path(dataset_cfg.root)
        self.task = dataset_cfg.task
        self.version = dataset_cfg.version
        self.splits_file = dataset_cfg.splits_file
        self.modalities = [m.upper() for m in dataset_cfg.modality_infos.keys()]
        self.temporal_mode = dataset_cfg.get("temporal_mode", "sits")
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory if torch.cuda.is_available() else False
        self.train_transforms = train_transforms
        self.val_transforms = val_transforms
        self.test_transforms = test_transforms
        self.test_shuffle = test_shuffle

    def _build_file_list(self, entries: list[dict], data_dir: Path) -> list[dict]:
        files = []
        for e in entries:
            entry = {**e}  # shallow copy
            for mod in self.modalities:
                if mod != "DEM":
                    entry[mod] = str(data_dir / e[mod.lower()])
            files.append(entry)
        return files

    def setup(self, stage=None):
        
        with open(self.splits_file) as f:
            splits = json.load(f)
        
        data_dir = self.root / "data" / f"data_{self.version}"
        
        train_files = self._build_file_list(splits["train"], data_dir)
        val_files = self._build_file_list(splits["val"], data_dir)
        test_files = self._build_file_list(splits["test"], data_dir)
        
        print(f"[Sen12Landslides] Split: Train={len(train_files)}, Val={len(val_files)}, Test={len(test_files)}")

        common = dict(modalities=self.modalities, temporal_mode=self.temporal_mode, task=self.task)
        self.train_dataset = Sen12Landslides(train_files, transforms=self.train_transforms, **common)
        self.val_dataset = Sen12Landslides(val_files, transforms=self.val_transforms, **common)
        self.test_dataset = Sen12Landslides(test_files, transforms=self.test_transforms, **common)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, self.batch_size, shuffle=True,
                          num_workers=self.num_workers, pin_memory=self.pin_memory, drop_last=True,
                          collate_fn=metadata_collate)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, self.batch_size, shuffle=False, 
                          num_workers=self.num_workers, pin_memory=self.pin_memory, drop_last=True,
                          collate_fn=metadata_collate)
    
    def test_dataloader(self):
        return DataLoader(self.test_dataset, self.batch_size, shuffle=self.test_shuffle,
                          num_workers=self.num_workers, pin_memory=self.pin_memory, drop_last=True,
                          collate_fn=metadata_collate)