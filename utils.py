"""Plotting and inspection helpers for a sample produced by ImpactMeshDataset."""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from matplotlib.colors import BoundaryNorm, ListedColormap, to_rgb

STATIC_MODS = {"DEM"}
TERRAIN_CHANNELS = {
    "DEM": [
        ("Elevation", 0, "terrain"),
        ("Slope", 1, "viridis"),
        ("Aspect", 2, "twilight"),
    ],
}
TASK_MASK_COLORS = {
    "flood": {
        -1: "#000000",  # ignore/nodata
         0: "#dddddd",  # background
         1: "#89c4e8",  # permanent water
         2: "#0a4c8c",  # flooded
    },
    "fire": {
        -1: "#000000",  # ignore/nodata
         0: "#dddddd",  # background
         1: "#7a0c0c",  # burned
    },
}
# The class value that counts as "positive" (hazard) per task, used for
# TP/FP/FN scoring only -- not for the mask visualization colors.
POSITIVE_CLASS = {"flood": 2, "fire": 1}


def _normalize_percentile(arr: np.ndarray, lo: float = 2, hi: float = 98) -> np.ndarray:
    p_lo, p_hi = np.percentile(arr, [lo, hi])
    return np.clip((arr - p_lo) / (p_hi - p_lo + 1e-8), 0, 1)


def _render_frame(ax, frame: np.ndarray, s2_rgb_indices: tuple[int, ...], is_s2: bool):
    """Render a (C, H, W) frame: S2 -> RGB via s2_rgb_indices, else first 3 channels or grayscale."""
    c = frame.shape[0]
    if is_s2 and c >= max(s2_rgb_indices) + 1:
        rgb = frame[list(s2_rgb_indices)].transpose(1, 2, 0)
        ax.imshow(_normalize_percentile(rgb))
    elif c >= 3:
        ax.imshow(_normalize_percentile(frame[:3].transpose(1, 2, 0)))
    else:
        ax.imshow(_normalize_percentile(frame[0]), cmap="gray")


def _count_static_cols(image, static_mods, terrain_channels):
    n = 0
    for sm in static_mods:
        spec = terrain_channels.get(sm.upper())
        n += min(len(spec), image[sm].shape[0]) if spec else 1
    return n


def _mask_cmap_norm(task: str | None):
    """Discrete cmap+norm for the mask column. Falls back to binary black/white
    for an unknown task."""
    colors = TASK_MASK_COLORS.get(task)
    if colors is None:
        cmap = ListedColormap([(0, 0, 0, 1), (1, 1, 1, 1)])
        return cmap, BoundaryNorm(np.arange(-0.5, 2.5, 1), cmap.N)
    keys = sorted(colors)
    cmap = ListedColormap([colors[k] for k in keys])
    return cmap, BoundaryNorm(np.arange(keys[0] - 0.5, keys[-1] + 1.5, 1), cmap.N)


def print_metadata(sample: dict, task: str | None = None):
    """Print a short summary of one sample: shapes, timestamps, mask stats."""
    meta = sample["metadata"]
    print("=" * 70)
    print(f"  ID:         {meta.get('id', 'unknown')}")
    print(f"  Task:       {(task or 'unknown').upper()}")
    print(f"  Event date: {meta.get('event_date', 'None')}")
    print("=" * 70)

    mask = sample.get("mask")
    if isinstance(mask, torch.Tensor):
        pos = (mask > 0).float().mean().item()
        print(f"  Mask:   shape={tuple(mask.shape)}  pos_frac={pos:.3f}")

    print("\n  Image modalities:")
    for mod, x in sample.get("image", {}).items():
        nan = torch.isnan(x).sum().item() if isinstance(x, torch.Tensor) else 0
        print(f"    {mod:8s} shape={tuple(x.shape)}  mean={x.mean():+.3f}  std={x.std():.3f}  nan={nan}")

    timestamps = meta.get("timestamps", {})
    if timestamps:
        print("\n  Per-modality timestamps:")
        for mod, ts in timestamps.items():
            ts = pd.to_datetime(ts)
            print(f"    {mod}: {ts[0].strftime('%Y-%m-%d')} -> {ts[-1].strftime('%Y-%m-%d')}  (T={len(ts)})")

    print("=" * 70)


def plot_sample(
    sample: dict,
    s2_rgb_indices: tuple[int, ...] = (3, 2, 1),
    task: str = "flood",
    prediction: torch.Tensor | np.ndarray | None = None,
    save_path: str | Path | None = None,
):
    """Plot one sample: one row per temporal modality, one column per timestep,
    plus terrain feature columns, the mask, and (if given) the prediction and
    a TP/FP/FN overlay.

    Args:
        sample: dict with "image", "mask", "metadata", as returned by ImpactMeshDataset.
        s2_rgb_indices: band indices for S2 true-color (default: B04, B03, B02).
        task: "flood" or "fire" -- selects the mask color scheme.
        prediction: optional (H, W) or (C, H, W) tensor/array of predicted class labels.
        save_path: if given, save the figure there instead of showing it.
    """
    image = sample["image"]
    mask = sample["mask"].cpu().numpy() if isinstance(sample["mask"], torch.Tensor) else sample["mask"]
    meta = sample["metadata"]

    event_date_raw = meta.get("event_date")
    event_date = pd.to_datetime(event_date_raw) if event_date_raw else None

    temporal_mods = [k for k in image if k.upper() not in STATIC_MODS]
    static_mods = [k for k in image if k.upper() in STATIC_MODS]

    has_pred = prediction is not None
    if has_pred:
        pred = prediction.cpu().numpy() if isinstance(prediction, torch.Tensor) else prediction
        if pred.ndim == 3:
            pred = pred[0] if pred.shape[0] == 1 else pred.argmax(0)
    else:
        pred = None

    T = max((image[mod].shape[0] for mod in temporal_mods), default=1)
    n_static_cols = _count_static_cols(image, static_mods, TERRAIN_CHANNELS)
    n_cols = T + n_static_cols + 1 + (2 if has_pred else 0)
    n_rows = max(len(temporal_mods), 1)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(2.0 * n_cols, 2.6 * n_rows))
    if n_rows == 1:
        axes = axes.reshape(1, -1)

    mask_cmap, mask_norm = _mask_cmap_norm(task)
    positive_class = POSITIVE_CLASS.get(task, 1)

    for row, mod in enumerate(temporal_mods):
        img = image[mod].cpu().numpy()
        T_mod = img.shape[0]
        timestamps = meta.get("timestamps", {}).get(mod)
        is_s2 = mod.upper().startswith("S2")

        for t in range(T):
            ax = axes[row, t]
            if t >= T_mod:
                ax.axis("off")
                continue
            _render_frame(ax, img[t], s2_rgb_indices, is_s2)
            ax.axis("off")

            title = pd.Timestamp(timestamps[t]).strftime("%Y-%m-%d") if timestamps is not None else ""
            ax.set_title(title, fontsize=7)

        col = T
        for sm in static_mods:
            sdata = image[sm].cpu().numpy()  # (C, H, W)
            channels_spec = TERRAIN_CHANNELS.get(sm.upper())
            if channels_spec is None:
                ax = axes[row, col]
                _render_frame(ax, sdata, s2_rgb_indices, is_s2=False)
                if row == 0:
                    ax.set_title(sm, fontsize=7, fontweight="bold")
                ax.axis("off")
                col += 1
            else:
                for name, ch_idx, cmap in channels_spec:
                    if ch_idx >= sdata.shape[0]:
                        continue
                    ax = axes[row, col]
                    ax.imshow(_normalize_percentile(sdata[ch_idx]), cmap=cmap)
                    if row == 0:
                        ax.set_title(name, fontsize=7, fontweight="bold")
                    ax.axis("off")
                    col += 1

        axes[row, col].imshow(mask, cmap=mask_cmap, norm=mask_norm, interpolation="nearest")
        if row == 0:
            axes[row, col].set_title("Mask", fontsize=7, fontweight="bold")
        axes[row, col].axis("off")
        col += 1

        if has_pred:
            axes[row, col].imshow(pred, cmap=mask_cmap, norm=mask_norm, interpolation="nearest")
            if row == 0:
                axes[row, col].set_title("Prediction", fontsize=7, fontweight="bold")
            axes[row, col].axis("off")
            col += 1

            ignore = mask == -1
            gt_pos = mask == positive_class
            pred_pos = pred == positive_class
            bg_color = np.array(to_rgb(TASK_MASK_COLORS.get(task, {}).get(0, "#dddddd")))
            ignore_color = np.array(to_rgb(TASK_MASK_COLORS.get(task, {}).get(-1, "#000000")))

            tpfpfn = np.tile(bg_color, (*pred.shape, 1))
            tpfpfn[pred_pos & gt_pos] = [0, 1, 0]        # TP
            tpfpfn[pred_pos & ~gt_pos & ~ignore] = [1, 0, 0]  # FP
            tpfpfn[~pred_pos & gt_pos] = [0, 0, 1]       # FN
            tpfpfn[ignore] = ignore_color
            axes[row, col].imshow(tpfpfn, interpolation="nearest")
            if row == 0:
                axes[row, col].set_title("TP/FP/FN", fontsize=7, fontweight="bold")
            axes[row, col].axis("off")
            col += 1

        axes[row, 0].text(-0.15, 0.5, f"{mod}\n(T={T_mod})", transform=axes[row, 0].transAxes,
                           fontsize=9, fontweight="bold", va="center", ha="right", rotation=90)

    title_parts = [str(meta.get("id", "unknown"))]
    if event_date is not None:
        title_parts.append(f"event={event_date.strftime('%Y-%m-%d')}")
    fig.suptitle(" | ".join(title_parts), fontsize=11, fontweight="bold", y=0.99)
    plt.tight_layout()

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()
