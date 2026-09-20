import json
import random
import torch
import torch.nn.functional as F
import logging

log = logging.getLogger(__name__)

def _apply_spatial(sample: dict, fn):
    """Apply a spatial op to every modality tensor, the label mask, and -- when
    present -- the per-pixel cloud mask. The cloud mask is the target of the
    reliability head, so it has to follow the exact same geometry as the image;
    leaving it out of the augmentations would silently train the head against
    a flipped target."""
    for mod, x in sample["image"].items():
        sample["image"][mod] = fn(x)
    if isinstance(sample.get("mask"), torch.Tensor):
        sample["mask"] = fn(sample["mask"])
    if isinstance(sample.get("cloud_mask"), torch.Tensor):
        sample["cloud_mask"] = fn(sample["cloud_mask"])
    return sample


class RandomHorizontalFlip:
    """Flip width dimension. Applies to all modality tensors, label and cloud mask."""

    def __init__(self, p: float = 0.5):
        self.p = p

    def __call__(self, sample: dict) -> dict:
        if random.random() >= self.p:
            return sample
        return _apply_spatial(sample, lambda x: x.flip(-1))


class RandomVerticalFlip:
    """Flip height dimension. Applies to all modality tensors, label and cloud mask."""

    def __init__(self, p: float = 0.5):
        self.p = p

    def __call__(self, sample: dict) -> dict:
        if random.random() >= self.p:
            return sample
        return _apply_spatial(sample, lambda x: x.flip(-2))


class RandomRotate90:
    """Rotate by k*90 degrees, k sampled from {1,2,3}."""

    def __init__(self, p: float = 0.5):
        self.p = p

    def __call__(self, sample: dict) -> dict:
        if random.random() >= self.p:
            return sample
        k = random.randint(1, 3)
        return _apply_spatial(sample, lambda x: torch.rot90(x, k, dims=(-2, -1)))


class NormalizeImpactMesh:
    """Normalization for ImpactMesh: stats JSON layout {"means": {mod: [...]}, "stds": {mod: [...]}}.

    All S2L2A variants (clear/cloudy/realistic) share one set of stats — they are the
    same physical reflectance space, so normalization is a single fixed transform.
    """

    def __init__(self, norm_json: str, eps: float = 1e-6):
        with open(norm_json) as f:
            stats = json.load(f)

        self.mean, self.std = {}, {}
        for k, mean_list in stats["means"].items():
            self.mean[k] = torch.tensor(mean_list, dtype=torch.float32).view(1, -1, 1, 1)
            self.std[k] = torch.clamp(
                torch.tensor(stats["stds"][k], dtype=torch.float32), min=eps
            ).view(1, -1, 1, 1)

        if "S2L2A" not in self.mean:
            raise KeyError(f"S2L2A not in {norm_json}; have {list(self.mean)}")

        log.info("Normalization with shared S2L2A stats (all variants).")

    def _key(self, mod: str) -> str:
        # any S2L2A variant normalizes against the shared S2L2A stats;
        # other modalities map to their own stats key directly.
        return "S2L2A" if mod.upper().startswith("S2L2A") else mod.upper()

    def __call__(self, sample: dict) -> dict:
        for mod, x in sample["image"].items():
            key = self._key(mod)
            if key in self.mean:
                sample["image"][mod] = (x - self.mean[key]) / self.std[key]
        return sample


class NormalizeSen12LS:
    """Normalization for Sen12Landslides: stats JSON layout {mod: {"mean": {band: ...}, "std": {band: ...}}}.
    DEM stats are nested inside each modality (identical values); we hoist from the first available."""

    BAND_ORDER = {
        "S1ASC": ["VH", "VV"],
        "S1DSC": ["VH", "VV"],
        "S2": ["B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B11", "B12"],
    }

    def __init__(self, norm_json: str, eps: float = 1e-6):
        with open(norm_json) as f:
            stats = json.load(f)

        self.mean = {}
        self.std = {}

        for mod_upper, bands in self.BAND_ORDER.items():
            mod_lower = mod_upper.lower()
            if mod_lower not in stats:
                continue
            mean = torch.tensor([stats[mod_lower]["mean"][b] for b in bands], dtype=torch.float32)
            std = torch.clamp(
                torch.tensor([stats[mod_lower]["std"][b] for b in bands], dtype=torch.float32),
                min=eps,
            )
            self.mean[mod_upper] = mean.view(1, -1, 1, 1)
            self.std[mod_upper] = std.view(1, -1, 1, 1)

        # DEM: hoist from any modality (stats are identical across them)
        for mod_lower in ("s2", "s1asc", "s1dsc"):
            if mod_lower in stats and "DEM" in stats[mod_lower]["mean"]:
                dem_mean = stats[mod_lower]["mean"]["DEM"]
                dem_std = max(stats[mod_lower]["std"]["DEM"], eps)
                self.mean["DEM"] = torch.tensor([[[[dem_mean]]]], dtype=torch.float32)
                self.std["DEM"] = torch.tensor([[[[dem_std]]]], dtype=torch.float32)
                break

    def __call__(self, sample: dict) -> dict:
        for mod, x in sample["image"].items():
            mod_upper = mod.upper()
            if mod_upper in self.mean:
                sample["image"][mod] = (x - self.mean[mod_upper]) / self.std[mod_upper]
        return sample


class TerrainFeatures:
    """
    Compute terrain features from DEM and replace DEM channels in sample
    Applied as a transform AFTER normalization
    Args:
        features: Which features to include. Any combination of:
            - "dem": raw elevation (unsmoothed)
            - "slope": gradient magnitude  
            - "aspect": gradient direction normalized to [-1, 1]
        dem_key: Key in sample["img"] for DEM modality
        smooth_sigma: Gaussian sigma (in pixels) applied to DEM before
            computing derivatives. Removes quantization stripes from
            30m DEM resampled to 10m grid. Set 0 to disable.
    """
    def __init__(self, features: list[str] = ["slope", "aspect"],
                 dem_key: str = "DEM", smooth_sigma: float = 1.0):
        self.features = [f.lower() for f in features]
        self.dem_key = dem_key
        self.smooth_sigma = smooth_sigma

        # Pre-build Gaussian kernel if smoothing enabled
        if smooth_sigma > 0:
            ks = int(4 * smooth_sigma + 1) | 1  # ensure odd
            ax = torch.arange(ks, dtype=torch.float32) - ks // 2
            kernel_1d = torch.exp(-0.5 * (ax / smooth_sigma) ** 2)
            kernel_2d = kernel_1d[:, None] * kernel_1d[None, :]
            kernel_2d = kernel_2d / kernel_2d.sum()
            # Shape for F.conv2d: [1, 1, ks, ks]
            self.register_buffer_kernel = kernel_2d.unsqueeze(0).unsqueeze(0)
            self.pad_size = ks // 2
        else:
            self.register_buffer_kernel = None

    def _smooth(self, dem: torch.Tensor) -> torch.Tensor:
        """Apply Gaussian smoothing to DEM. Input/output: [B, 1, H, W]."""
        if self.register_buffer_kernel is None:
            return dem
        kernel = self.register_buffer_kernel.to(dem.device, dem.dtype)
        return F.conv2d(
            F.pad(dem, [self.pad_size] * 4, mode='replicate'),
            kernel)

    def __call__(self, sample: dict) -> dict:
        if self.dem_key not in sample.get("image", {}):
            return sample

        dem = sample["image"][self.dem_key]  # [1, 1, H, W] or [1, H, W]
        if dem.dim() == 3:
            dem = dem.unsqueeze(0)  # [1, 1, H, W]

        channels = []

        # Compute derivatives on smoothed DEM
        need_derivs = "slope" in self.features or "aspect" in self.features
        if need_derivs:
            dem_smooth = self._smooth(dem)
            # Central differences, divided by 2 for correct gradient
            dy = (dem_smooth[:, :, 2:, :] - dem_smooth[:, :, :-2, :]) / 2.0
            dx = (dem_smooth[:, :, :, 2:] - dem_smooth[:, :, :, :-2]) / 2.0
            dy = F.pad(dy, (0, 0, 1, 1), mode='replicate')
            dx = F.pad(dx, (1, 1, 0, 0), mode='replicate')

        for feat in self.features:
            if feat == "dem":
                channels.append(dem)  # raw, unsmoothed
            elif feat == "slope":
                channels.append(torch.sqrt(dx**2 + dy**2 + 1e-8))
            elif feat == "aspect":
                channels.append(torch.atan2(dy, dx) / torch.pi)

        # Replace DEM data with selected terrain features
        terrain = torch.cat(channels, dim=1)  # [1, N_feats, H, W]
        sample["image"][self.dem_key] = terrain

        return sample