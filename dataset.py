"""PyTorch Dataset for the ImpactMesh-X release.

Loads a patch's full multi-temporal stack for every requested modality --
S1RTC, S2L2A (in one of its three cloud variants: clear/cloudy/realistic),
DEM, and the annotation mask -- with no timestep selection or event-anchor
logic. If you want fewer timesteps (e.g. only the first post-event frame),
select them yourself downstream; keeping that decision out of the dataset
keeps this loader simple and the full series available to everyone.

Expects the release layout produced by the dataset build pipeline:

    <root>/
      S1RTC/{patch_id}_S1RTC.zarr.zip
      S2L2A/{clear,cloudy,realistic}/{patch_id}_S2L2A.zarr.zip
      DEM/{patch_id}_DEM.tif
      MASK/{patch_id}_annotation_{flood,wildfire}.tif   # task="fire" reads the "wildfire" mask
      {hazard}_patch_meta.parquet
"""
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import rasterio
import torch
import zarr
from torch.utils.data import Dataset

NO_DATA_VALUE = -9999.0


def clean_nodata(data: np.ndarray, nodata: Optional[float], replace: float = 0.0) -> np.ndarray:
    """Replace non-finite values (NaN/inf) and an optional sentinel value with `replace`."""
    if data.dtype != np.float32:
        data = data.astype(np.float32, copy=False)
    invalid = ~np.isfinite(data)
    if nodata is not None and np.isfinite(nodata):
        invalid |= data == nodata
    if invalid.any():
        data[invalid] = replace
    return data


class ImpactMeshDataset(Dataset):
    """Multi-modal, multi-temporal dataset for one hazard (flood or fire).

    Args:
        root: Path to the release folder for one hazard, e.g. ".../ImpactMesh-X/Flood".
        task: "flood" or "fire".
        patch_ids: Which patches to load. If None, loads every patch_id in the
            metadata file (optionally filtered by `pool`).
        modalities: Any combination of "S2L2A", "S1RTC", "DEM".
        s2_version: Which S2L2A cloud variant to load ("clear", "cloudy", "realistic").
        pool: Restrict to "positive" (hazard present) or "negative" (hazard-free
            control) samples. None keeps both.
        transforms: Optional callable applied to the finished sample dict.
        no_data_value: Sentinel value to treat as missing, in addition to NaN/inf.
        no_data_replace: Value substituted for missing pixels.
    """

    def __init__(
        self,
        root: str | Path,
        task: str,
        patch_ids: Optional[list[str]] = None,
        modalities: list[str] = ["S2L2A", "S1RTC", "DEM"],
        s2_version: str = "clear",
        pool: Optional[str] = None,
        transforms: Optional[callable] = None,
        no_data_value: float = NO_DATA_VALUE,
        no_data_replace: float = 0.0,
    ):
        if task not in ("flood", "fire"):
            raise ValueError(f"task must be 'flood' or 'fire', got {task!r}")
        if s2_version not in ("clear", "cloudy", "realistic"):
            raise ValueError(f"s2_version must be 'clear', 'cloudy' or 'realistic', got {s2_version!r}")

        self.root = Path(root)
        self.task = task
        self.modalities = [m.upper() for m in modalities]
        self.s2_version = s2_version
        self.transforms = transforms
        self.no_data_value = no_data_value
        self.no_data_replace = no_data_replace

        meta_path = self.root / f"{task}_patch_meta.parquet"
        if not meta_path.exists():
            raise FileNotFoundError(f"metadata file not found: {meta_path}")
        meta = pd.read_parquet(meta_path)
        meta["patch_id"] = meta["patch_id"].astype(str)

        if pool is not None:
            meta = meta[meta["pool"] == pool]
        if patch_ids is not None:
            meta = meta[meta["patch_id"].isin(set(patch_ids))]

        self.meta = meta.set_index("patch_id", drop=False)
        self.patch_ids = self.meta["patch_id"].tolist()

        self.paths = {
            "S1RTC": ("S1RTC", "_S1RTC.zarr.zip"),
            "S2L2A": (f"S2L2A/{s2_version}", "_S2L2A.zarr.zip"),
            "DEM": ("DEM", "_DEM.tif"),
        }
        # The release's mask files use "wildfire" in their name regardless of the
        # "fire" task value used everywhere else in this API.
        mask_suffix = "_annotation_wildfire.tif" if task == "fire" else "_annotation_flood.tif"
        self.mask_path = ("MASK", mask_suffix)

    def __len__(self) -> int:
        return len(self.patch_ids)

    def _load_tif(self, path: Path) -> torch.Tensor:
        with rasterio.open(path, "r") as f:
            nodata = f.nodata if f.nodata is not None else self.no_data_value
            data = clean_nodata(f.read(), nodata, self.no_data_replace)
        return torch.from_numpy(data)  # (C, H, W)

    def _load_zarr(self, path: Path) -> dict:
        """Returns bands (T, C, H, W), timestamps, and the event_date attribute."""
        with zarr.storage.ZipStore(str(path), mode="r") as store:
            g = zarr.open_consolidated(store, mode="r")
            fill = g["bands"].attrs.get("_FillValue", None)
            bands = clean_nodata(g["bands"][:], fill, self.no_data_replace)
            times_ns = g["time"][:]
            event_date = g.attrs.get("event_date", None)
        timestamps = pd.to_datetime(times_ns.astype("datetime64[ns]"))
        return {"bands": torch.from_numpy(bands), "timestamps": timestamps, "event_date": event_date}

    def __getitem__(self, idx: int) -> dict:
        pid = self.patch_ids[idx]
        row = self.meta.loc[pid]

        sample = {
            "image": {},
            "mask": None,
            "metadata": {"id": pid, "timestamps": {}, "event_date": row["event_date"]},
        }

        for mod in self.modalities:
            folder, suffix = self.paths[mod]
            path = self.root / folder / f"{pid}{suffix}"
            if mod == "DEM":
                sample["image"]["DEM"] = self._load_tif(path)
            else:
                data = self._load_zarr(path)
                sample["image"][mod] = data["bands"]
                sample["metadata"]["timestamps"][mod] = data["timestamps"]

        mask_folder, mask_suffix = self.mask_path
        mask_path = self.root / mask_folder / f"{pid}{mask_suffix}"
        with rasterio.open(mask_path, "r") as f:
            mask = f.read(1)
        sample["mask"] = torch.from_numpy(mask.astype(np.int64))

        return self.transforms(sample) if self.transforms else sample
