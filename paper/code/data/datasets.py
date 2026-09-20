import numpy as np
import torch
import xarray as xr
from torch.utils.data import Dataset
import pandas as pd
from typing import Optional
import zarr
import rasterio
from fusion.data.utils import (
    select_timesteps, plot_sample, print_metadata, clean_nodata, add_event_targets,
    relative_days, day_of_year, days_between)


class ImpactMesh(Dataset):
    """Multi-modal dataset for ImpactMesh."""

    def __init__(
        self,
        files: list[dict],
        modalities: list[str] = ["S2L2A", "S1RTC", "DEM"],
        transforms: Optional[callable] = None,
        no_data_value: float = -9999.0,
        no_data_replace: float = 0.0,
        temporal_mode: str = "multi-temporal",
        task: Optional[str] = None
    ):
        self.files = files
        self.modalities = [m.upper() for m in modalities]
        self.transforms = transforms
        self.no_data_value = no_data_value
        self.no_data_replace = no_data_replace
        self.temporal_mode = temporal_mode

        if task == "wildfire":
            self.task = "fire"
        elif task == "flood":
            self.task = "flood"
        else:
            raise ValueError(f"task must be 'wildfire' or 'flood', got {task!r}")

    def __len__(self) -> int:
        return len(self.files)

    def _load_tif(self, path: str) -> torch.Tensor:
        with rasterio.open(path, "r") as f:
            nodata = f.nodata if f.nodata is not None else self.no_data_value
            data = clean_nodata(f.read(), nodata, self.no_data_replace)
        return torch.from_numpy(data[np.newaxis, ...])

    def _load_zarr(self, path: str):
        with zarr.storage.ZipStore(path, mode="r") as store:
            root = zarr.open_consolidated(store, mode="r")
            # S2: xarray schreibt _FillValue=-9999 in die attrs; S1: kein Attr -> None,
            # dort ist NaN das einzige Invalid-Signal.
            fill = root["bands"].attrs.get("_FillValue", None)
            bands = clean_nodata(root["bands"][:], fill, self.no_data_replace)
            t_ns = root["time"][:]
            cloud_mask = root["cloud_mask"][:] if "cloud_mask" in root else None
            event_date = root.attrs.get("event_date", None)

        timestamps = pd.to_datetime(t_ns.astype("datetime64[ns]"))

        if cloud_mask is not None:
            cloud_mask = torch.from_numpy(cloud_mask.astype(np.float32))
            cloud_frac = cloud_mask.mean(dim=(1, 2))
        else:
            cloud_frac = None

        return torch.from_numpy(bands), timestamps, cloud_frac, cloud_mask, event_date

    def __getitem__(self, idx: int) -> dict:
        entry = self.files[idx]
        sample = {
            "image": {},
            "mask": {},
            "metadata": {
                "id": entry["id"],
                "timestamps": {},
                "date_mask": {},
                "anchors": entry.get("anchors"),
            },
        }
        event_dates = []
        for mod in self.modalities:

            path = str(entry[mod])

            if path.endswith(".tif"):
                sample["image"][mod] = self._load_tif(path)
            else:
                bands, timestamps, cloud_frac, cloud_mask, event_date = self._load_zarr(path)
                sample["image"][mod] = bands
                sample["metadata"]["timestamps"][mod] = timestamps
                sample["metadata"]["date_mask"][mod] = torch.ones(len(timestamps), dtype=torch.bool)
                event_dates.append(event_date)
                if cloud_frac is not None:
                    sample["metadata"]["cloud_frac"] = cloud_frac
                if cloud_mask is not None:
                    sample["cloud_mask"] = cloud_mask

        valid = [d for d in event_dates if d is not None and d != "None"]
        if not valid:
            sample["metadata"]["event_date"] = None
        elif len(set(valid)) == 1:
            sample["metadata"]["event_date"] = str(valid[0])
        else:
            raise ValueError(f"Event dates mismatch across modalities for ID {entry['id']}: {valid}")

        with rasterio.open(entry["MASK"], "r") as f:
            mask = f.read(1)
        sample["mask"] = torch.from_numpy(mask.astype(np.int64))

        sample = select_timesteps(sample, self.temporal_mode)
        if sample is None:
            return self[(idx + 1) % len(self)]
        # After the selection: zero is the first frame the model actually sees.
        # The event day is put on that same axis -- it is a training target, not
        # an input, and there is no global reference date anywhere any more.
        rel, ref = relative_days(sample["metadata"]["timestamps"])
        sample["metadata"]["rel_day"] = rel
        sample["metadata"]["doy"] = day_of_year(sample["metadata"]["timestamps"])
        ed = sample["metadata"]["event_date"]
        sample["metadata"]["event_day"] = (
            None if (ed is None or ref is None) else days_between(ref, ed))
        sample = add_event_targets(sample)
        return self.transforms(sample) if self.transforms else sample


    def print_metadata(self, sample_or_idx):
        if isinstance(sample_or_idx, int):
            sample = self[sample_or_idx]
        else:
            sample = sample_or_idx

        print_metadata(sample, task=self.task)

    def plot(self, sample_or_idx, prediction=None, save_path=None):
        """Plot a sample or batch element. `sample_or_idx` is int (load from disk),
        a single sample dict, or a batch dict with batch_idx=0."""
        if isinstance(sample_or_idx, int):
            sample = self[sample_or_idx]
        else:
            sample = sample_or_idx

        plot_sample(
            sample,
            s2_rgb_indices=(3,2,1),
            task=self.task,
            prediction=prediction,
            save_path=save_path,
        )


class Sen12Landslides(Dataset):
    """Multi-modal dataset for landslide detection."""

    EXCLUDE_VARS = {"MASK", "DEM", "SCL", "spatial_ref"}
    CLOUD_SCL_CLASSES = (8, 9, 10)
    
    def __init__(
        self,
        files: list[dict],
        modalities: list[str],
        transforms: Optional[callable] = None,
        no_data_value: float = -9999.0,
        no_data_replace: float = 0.0,
        temporal_mode: str = "multi-temporal",
        task: Optional[str] = None
    ):
        self.files = files
        self.modalities = [m.upper() for m in modalities]
        self.transforms = transforms
        self.task = task
        self.temporal_mode = temporal_mode
        self.no_data_value = no_data_value
        self.no_data_replace = no_data_replace
        self.exclude_vars = {"MASK", "DEM", "SCL", "spatial_ref"}

    def __len__(self) -> int:
        return len(self.files)
    
    def _load_nc(self, path: str, mod: str, load_dem: bool) -> dict:
        """Open one NetCDF and extract everything available: modality bands, optional DEM,
        mask, event_date, and per-modality temporal fields."""
        with xr.open_dataset(path) as ds:
            ds = ds.sortby("time") if "time" in ds.coords else ds
            T = ds.sizes.get("time", 1)

            # --- Modality bands (T, C, H, W) ---
            ds_img = ds.drop_vars(self.EXCLUDE_VARS, errors="ignore")
            img = xr.concat([ds_img[v] for v in ds_img.data_vars], dim="bands")
            img = img.transpose("time", "bands", "y", "x").values
            bands = torch.from_numpy(clean_nodata(img, self.no_data_value, self.no_data_replace))

            # --- Timestamps ---
            raw_ts = ds["time"].values if "time" in ds.sizes else np.array([0.0])
            timestamps = pd.to_datetime(raw_ts)

            # --- Cloud fraction (S2 only) ---
            cloud_frac = None
            if mod == "S2" and "SCL" in ds:
                cloud = np.isin(ds["SCL"].values, self.CLOUD_SCL_CLASSES)
                cloud_frac = torch.from_numpy(cloud.mean(axis=(1, 2)).astype(np.float32))

            # --- Date mask: post-event index ---
            date_mask = torch.ones(T, dtype=torch.bool)
            # raw = ds.attrs.get("pre_post_dates")
            # if raw is not None:
            #     pre_post = ast.literal_eval(raw) if isinstance(raw, str) else raw
            #     if isinstance(pre_post, dict):
            #         post_idx = pre_post.get("post")
            #         if isinstance(post_idx, (int, np.integer)) and 0 <= post_idx < T:
            #             date_mask[post_idx] = True

            # --- DEM (1, C=1, H, W), optional ---
            dem = None
            if load_dem and "DEM" in ds:
                dem_da = ds["DEM"].isel(time=0) if "time" in ds["DEM"].dims else ds["DEM"]
                arr = clean_nodata(dem_da.transpose("y", "x").values.astype(np.float32), self.no_data_value, self.no_data_replace)
                dem = torch.from_numpy(arr).unsqueeze(0)

            # --- Mask (H, W) ---
            mask = None
            if "MASK" in ds:
                msk = ds["MASK"].isel(time=0) if "time" in ds["MASK"].dims else ds["MASK"]
                mask = torch.from_numpy((msk.transpose("y", "x").values > 0).astype(np.int64))

            # --- Event date (raw string, put on the relative axis downstream) ---
            # Some Sen12Landslides patches carry several comma-separated event dates - Takes the earliest
            raw_evt = ds.attrs.get("event_date")
            dates = [p.strip() for p in str(raw_evt).split(",")] if raw_evt is not None else []
            dates = [p for p in dates if p and p != "None"]
            event_date = min(dates, key=pd.to_datetime) if dates else None

        return {
            "bands": bands,
            "timestamps": timestamps,
            "cloud_frac": cloud_frac,
            "date_mask": date_mask,
            "dem": dem,
            "mask": mask,
            "event_date": event_date,
        }

    def __getitem__(self, idx: int) -> dict:
        entry = self.files[idx]
        temporal_mods = [m for m in self.modalities if m != "DEM"]
        load_dem = "DEM" in self.modalities

        sample = {
            "image": {},
            "mask": {},
            "metadata": {
                "id": entry["id"],
                "timestamps": {},
                "days": {},
                "date_mask": {},
                "anchors": None,
            },
        }

        event_dates = []
        mask_set = dem_set = False

        for mod in temporal_mods:
            data = self._load_nc(entry[mod], mod, load_dem=load_dem and not dem_set)

            sample["image"][mod] = data["bands"]
            sample["metadata"]["timestamps"][mod] = data["timestamps"]
            sample["metadata"]["date_mask"][mod] = data["date_mask"]

            if data["cloud_frac"] is not None:
                sample["metadata"]["cloud_frac"] = data["cloud_frac"]

            if data["dem"] is not None and not dem_set:
                sample["image"]["DEM"] = data["dem"]
                dem_set = True

            if not mask_set and data["mask"] is not None:
                sample["mask"] = data["mask"]
                mask_set = True

            event_dates.append(data["event_date"])

        # Validate event dates across modalities
        valid_dates = [d for d in event_dates if d is not None]
        if not valid_dates:
            sample["metadata"]["event_date"] = None
        elif len(set(valid_dates)) == 1:
            sample["metadata"]["event_date"] = str(valid_dates[0])
        else:
            raise ValueError(f"Event dates mismatch ...")

        sample = select_timesteps(sample, self.temporal_mode)
        if sample is None:
            return self[(idx + 1) % len(self)]
        # After the selection: zero is the first frame the model actually sees.
        # The event day is put on that same axis -- it is a training target, not
        # an input, and there is no global reference date anywhere any more.
        rel, ref = relative_days(sample["metadata"]["timestamps"])
        sample["metadata"]["rel_day"] = rel
        sample["metadata"]["doy"] = day_of_year(sample["metadata"]["timestamps"])
        ed = sample["metadata"]["event_date"]
        sample["metadata"]["event_day"] = (
            None if (ed is None or ref is None) else days_between(ref, ed))
        sample = add_event_targets(sample)
        return self.transforms(sample) if self.transforms else sample

    def print_metadata(self, sample_or_idx):
        if isinstance(sample_or_idx, int):
            sample = self[sample_or_idx]
        else:
            sample = sample_or_idx    
        print_metadata(sample, task=self.task)

    def plot(self, sample_or_idx, prediction=None, save_path=None):
        """Plot a sample or batch element. `sample_or_idx` is int (load from disk),
        a single sample dict, or a batch dict with batch_idx=0."""
        if isinstance(sample_or_idx, int):
            sample = self[sample_or_idx]
        else:
            sample = sample_or_idx

        plot_sample(
            sample,
            s2_rgb_indices=(2,1,0),
            task = "landslide",
            prediction=prediction,
            save_path=save_path,
        )