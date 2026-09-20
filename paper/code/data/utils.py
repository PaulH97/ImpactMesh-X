from torch.utils.data import default_collate
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.colors import ListedColormap, BoundaryNorm, to_rgb
import numpy as np
import pandas as pd
import torch
from pathlib import Path

STATIC_MODS = {"DEM"}
TERRAIN_CHANNELS = {
    "DEM": [
        ("Elevation", 0, "terrain"),
        ("Slope",     1, "viridis"),
        ("Aspect",    2, "twilight"),
    ],
}
TASK_MASK_COLORS = {
    "flood": {
        -1: "#000000",  # ignore/nodata
         0: "#dddddd",  # background
         1: "#89c4e8",  # permanent water (lighter blue)
         2: "#0a4c8c",  # flooded (darker blue)
    },
    "fire":  {
        -1: "#000000",  # ignore/nodata
         0: "#dddddd",  # background
         1: "#7a0c0c",  # burned (dark red)
    },
    "landslide": {
        -1: "#000000",  # ignore/nodata
         0: "#dddddd",  # background
         1: "#d99a2b",  # landslide (ochre/brown)
    },
}

# Which class value counts as the "positive" (hazard) class per task.
# Used for TP/FP/FN scoring — NOT for the mask visualization colors.
POSITIVE_CLASS = {
    "flood": 2,
    "fire":  1,
    "landslide": 1
}

# Minimum distance to the event for the two outer slots. Mirrors the
# ImpactMesh convention, where pre_month sits 45-47 days before and post_event
# 26-32 days after the event date (medians).
MONTH_GAP_DAYS = 30

# The four slots of the original ImpactMesh series, in temporal order, each with
# the window it accepts relative to the event day. Every temporal mode is a
# subset of these, which is what makes the modes nested (1 subset 2 subset 4
# subset 16) and the ladder comparable.
_SLOTS = {
    "pre_month":  lambda off: off <= -MONTH_GAP_DAYS,
    "pre":        lambda off: off < 0,
    "post":       lambda off: off >= 0,
    "post_month": lambda off: off >= MONTH_GAP_DAYS,
}
# Per slot, the chain of windows to try: its own first, then the relaxed one,
# finally the whole series. `None` means "no condition".
_RELAX = {
    "pre_month":  ["pre_month", "pre", None],
    "pre":        ["pre", None],
    "post":       ["post", None],
    "post_month": ["post_month", "post", None],
}
TEMPORAL_SLOTS = {
    "single-post":   ["post"],
    "bi-temporal":   ["pre", "post"],
    "quad-temporal": ["pre_month", "pre", "post", "post_month"],
    "octo-temporal": ["pre_month", "pre", "post", "post_month"],
}
# Frames a mode takes ON TOP of its named slots, picked by `_fill_spread`.
# octo-temporal is quad-temporal plus four frames chosen purely for temporal
# coverage; starting from the quad set is what keeps the ladder nested,
# 1 subset 2 subset 4 subset 8 subset 16.
TEMPORAL_FILL = {"octo-temporal": 4}
# Order of assignment, not of the time axis: the two slots the annotation hangs
# on reach for their anchor first, then the outer slots fill up from the rest.
_PICK_ORDER = ["post", "pre", "pre_month", "post_month"]


def clean_nodata(data: np.ndarray, nodata: float | None, replace: float = 0.0) -> np.ndarray:
    """Replace non-finite values (NaN/inf) and an optional per-source sentinel.
    Covers: ImpactMesh S2 zarr (sentinel -9999, from _FillValue attr),
    ImpactMesh S1 zarr (NaN only, no sentinel), DEM tif (f.nodata tag),
    Sen12LS NetCDF (xarray-decoded NaN + residual sentinel)."""
    if data.dtype != np.float32:
        data = data.astype(np.float32, copy=False)
    invalid = ~np.isfinite(data)
    if nodata is not None and np.isfinite(nodata):
        invalid |= data == nodata
    if invalid.any():
        data[invalid] = replace
    return data


def _normalize_percentile(arr: np.ndarray, lo: float = 2, hi: float = 98) -> np.ndarray:
    p_lo, p_hi = np.percentile(arr, [lo, hi])
    return np.clip((arr - p_lo) / (p_hi - p_lo + 1e-8), 0, 1)


def _render_frame(ax, frame: np.ndarray, s2_rgb_indices: tuple[int, ...], is_s2: bool):
    """Render a [C, H, W] frame: S2 → RGB via s2_rgb_indices, else first 3 ch or grayscale."""
    C = frame.shape[0]
    if is_s2 and C >= max(s2_rgb_indices) + 1:
        rgb = frame[list(s2_rgb_indices)].transpose(1, 2, 0)
        ax.imshow(_normalize_percentile(rgb))
    elif C >= 3:
        rgb = frame[:3].transpose(1, 2, 0)
        ax.imshow(_normalize_percentile(rgb))
    else:
        ax.imshow(_normalize_percentile(frame[0]), cmap="gray")


def _count_static_cols(image, static_mods, terrain_channels):
    """How many columns the static section will take."""
    n = 0
    for sm in static_mods:
        spec = terrain_channels.get(sm.upper())
        if spec is None:
            n += 1
        else:
            n += min(len(spec), image[sm].shape[1])
    return n


def _mask_cmap_norm(task: str | None):
    """Discrete cmap+norm for the mask column, driven by TASK_MASK_COLORS[task].
    Falls back to a black/white binary for unknown tasks (e.g. landslides).
    """
    colors = TASK_MASK_COLORS.get(task)
    if colors is None:
        cmap = ListedColormap([(0, 0, 0, 1), (1, 1, 1, 1)])
        norm = BoundaryNorm(np.arange(-0.5, 2.5, 1), cmap.N)
        return cmap, norm
    keys = sorted(colors)
    cmap = ListedColormap([colors[k] for k in keys])
    norm = BoundaryNorm(np.arange(keys[0] - 0.5, keys[-1] + 1.5, 1), cmap.N)
    return cmap, norm


def _naive(timestamps) -> pd.DatetimeIndex:
    """Timestamps as a tz-naive DatetimeIndex."""
    ts = pd.DatetimeIndex(pd.to_datetime(timestamps))
    return ts.tz_localize(None) if ts.tz is not None else ts


def days_between(ref, when) -> float:
    """Signed days from `ref` to `when`."""
    return float((pd.Timestamp(when) - pd.Timestamp(ref)) / np.timedelta64(1, "D"))


def relative_days(timestamps_by_mod: dict) -> tuple[dict, pd.Timestamp | None]:
    """Acquisition positions for the temporal encoder, in days, and their origin.

        p_t = d_t - min_m d_0^(m)

    One rule, evaluated per sample. This is the only temporal encoding in the
    pipeline; everything downstream -- the positional encoder, the event target,
    the event head's output axis -- is expressed on it.

    One origin per sample, shared across modalities: the S1/S2 difference stays
    the real day offset, which mid fusion needs to order the sensors and early
    fusion needs to interpolate between them. A per-modality zero point would
    put both sensors at 0 and erase that offset.

    Call this AFTER the frames have been selected -- the origin must be the
    first frame the model actually sees. Taken from the full stack instead, the
    reduced regimes would inherit an offset inside the event-centred window,
    and that offset is exactly the event information we are removing.

    Two references that suggest themselves were measured and dropped:

    * Day of year breaks at the turn of the year. 88 % of the flood series and
      66 % of the fire series run past one, and day of year then decreases over
      the sequence; in 43 % / 34 % of samples a pair of frames more than 300
      days apart lands less than 30 apart, so the sinusoidal encoder would put
      two observations a year apart almost on top of each other.
    * Absolute days since a dataset-wide reference date are effectively an ID
      of the individual event, which the network can memorise.

    Relative days are monotone, bounded by the window length, identically
    distributed over train/val/test, and contain neither the absolute date nor
    the event date. They behave the same on the fixed ~360-day ImpactMesh
    windows and on the multi-year Sen12Landslides windows.

    NaT is interpolated over the sequence; a series that is invalid throughout
    falls back to zeros and no origin.
    """
    per_mod = {m: _naive(ts) for m, ts in timestamps_by_mod.items()}
    first = [ts[~ts.isna()].min() for ts in per_mod.values() if (~ts.isna()).any()]
    if not first:
        return {m: torch.zeros(len(ts), dtype=torch.float32) for m, ts in per_mod.items()}, None
    ref = min(first)

    out = {}
    for m, ts in per_mod.items():
        d = pd.Series(np.asarray((ts - ref) / np.timedelta64(1, "D"), dtype=np.float64))
        d = d.interpolate(method="linear", limit_direction="both").fillna(0.0)
        out[m] = torch.from_numpy(d.to_numpy(dtype=np.float32))
    return out, ref


def day_of_year(timestamps_by_mod: dict) -> dict:
    """Day of year (1..366) per acquisition, per modality.

    The reference U-TAE and TSViT use in their published form, kept as the
    second temporal encoding so the paper can report both. Unlike
    `relative_days` it carries the season an acquisition was taken in --
    phenology for fire, the wet season for flood -- but it has a known defect
    on this data and the comparison has to state it: day of year is cyclic, and
    88 % of the flood series and 66 % of the fire series run past a new year,
    where it starts decreasing over the sequence. In 43 % / 34 % of samples a
    pair of frames more than 300 days apart lands less than 30 apart in day of
    year, so the sinusoidal encoder puts two observations a year apart almost
    on top of each other. No exact collisions occur -- no series here exceeds
    365 days -- but the distances are wrong for the majority of samples.

    Like all positional fields, derive this from the SELECTED timestamps.

    NaT is interpolated over the sequence; a series that is invalid throughout
    falls back to zeros.
    """
    out = {}
    for m, ts in timestamps_by_mod.items():
        ts = _naive(ts)
        doy = np.full(len(ts), np.nan, dtype=np.float64)
        ok = ~ts.isna()
        if ok.any():
            doy[ok] = np.asarray(ts[ok].dayofyear, dtype=np.float64)
        doy = pd.Series(doy).interpolate(method="linear", limit_direction="both").fillna(0.0)
        out[m] = torch.from_numpy(doy.to_numpy(dtype=np.float32))
    return out


def _anchor_frame_idx(timestamps, anchor_date) -> int | None:
    """Index of the frame in `timestamps` that lands on the same calendar day as
    `anchor_date`, or None if there's no such frame (or no anchor_date)."""
    if anchor_date is None or (np.isscalar(anchor_date) and pd.isna(anchor_date)):
        return None
    target = pd.Timestamp(anchor_date).normalize()
    ts = pd.DatetimeIndex(timestamps).normalize()
    hit = np.where(ts == target)[0]
    return int(hit[0]) if len(hit) else None


def _pick_slot(slot, offset, score, used):
    """Frame index for one slot: its own window first, then relaxed.

    The relaxation is why every rung of the ladder runs on the same set of
    samples. Without it a sample that has, say, no pre-event acquisition at all
    drops out of bi-temporal and quad but stays in single-post and
    multi-temporal -- and then the table compares temporal regimes on different
    test sets. This is not hypothetical: 16.7 % of the Sen12Landslides S1 series
    have no frame before the event, 11.2 % of the S2 series none 30 days before.

    Relaxation happens in two steps: drop the month boundary first (the slot
    falls back to its inner counterpart), then the event boundary. A frame that
    is already taken is never picked twice.
    """
    for window in _RELAX[slot]:
        keep = _SLOTS[window](offset) if window else torch.ones_like(offset, dtype=torch.bool)
        cand = [i for i in torch.where(keep)[0].tolist() if i not in used]
        if cand:
            return min(cand, key=lambda i: float(score[i]))
    return None


def _fill_spread(offset, score, used, k) -> list[int]:
    """`k` more frame indices, chosen for temporal coverage. MUTATES `used`.

    Farthest-point sampling: each step takes the unused frame whose nearest
    already-selected frame is furthest away, so the widest uncovered stretch of
    the series is closed first. Seeded with the frames the named slots picked,
    which is what makes the result a superset of the shorter rung.

    Deliberately blind to the pre/post side of the event -- the distance is
    |offset_i - offset_j|, so the sign never enters. A series whose event sits
    late (measured: 3 pre / 13 post occurs in 7 of 60 IM-Fire test patches) is
    sampled where its frames actually are instead of being forced to a 50/50
    split. Measured over 60 test patches per hazard, this cuts the distance from
    the worst-covered acquisition to its nearest selected frame from 131 d to
    25 d (IM-Fire) and 144 d to 30 d (IM-Flood) against quad-temporal.

    Cloud only breaks ties, it never drives the choice: the full series takes
    every frame including the occluded ones, so a rung that preferred clear
    frames would confound "fewer frames" with "better frames" -- and since
    `score` carries no cloud term for S1, it would do so for S2 only.

    Ties fall back to `score`, then to the frame index, so the selection is
    deterministic -- it runs per `__getitem__`, and train and test have to see
    the same frames every epoch.
    """
    picks = []
    for _ in range(k):
        cand = [i for i in range(len(offset)) if i not in used]
        if not cand:
            break
        best = min(cand, key=lambda i: (
            -min(abs(float(offset[i]) - float(offset[j])) for j in used),
            float(score[i]),
            i,
        ))
        used.add(best)
        picks.append(best)
    return picks


def select_timesteps(sample: dict, mode: str) -> dict:
    """Select timesteps, independently per temporal modality.

    S1 and S2 revisit on different schedules, so a fused S12 sample needs its
    own pre/post pick for each. The modes are subsets of the four-part original
    ImpactMesh series:

        single-post   = {event_window}                                     (1)
        bi-temporal   = {pre_event, event_window}                          (2)
        quad-temporal = {pre_month, pre_event, event_window, post_event}   (4)
        octo-temporal = quad-temporal + 4 frames for temporal coverage     (8)
        multi-temporal= the full series                                    (16)

    For ImpactMesh, `sample["metadata"]["anchors"]` -- when present -- carries
    the acquisition dates the ImpactMesh team anchored the annotation to (the
    image the mask was drawn from, and their chosen references). All four slots
    prefer those exact dates over the cloud/date heuristic, so quad-temporal
    reproduces the original series frame by frame: measured, the four original
    acquisitions sit in 100 % of the 16-frame stacks (S1 and S2-clear; on
    cloudy/realistic they are different acquisitions and the heuristic takes
    over). Falls back to the heuristic when no anchor is available
    (Sen12Landslides has none) or the anchor date is not among this sample's
    loaded frames.

    The event date is used HERE, to pick frames, and nowhere else -- it never
    reaches the model. See `relative_days` for what the model is given instead.
    """
    if mode == "multi-temporal":
        return sample
    if mode not in TEMPORAL_SLOTS:
        raise ValueError(f"Unknown temporal mode: {mode}")
    slots = TEMPORAL_SLOTS[mode]
    n_fill = TEMPORAL_FILL.get(mode, 0)
    n_frames = len(slots) + n_fill

    sid = sample["metadata"].get("id")

    event_date = sample["metadata"].get("event_date")
    if event_date is None or str(event_date) == "None":
        print(f"[skip] {sid}: missing event_date")
        return None
    event = pd.Timestamp(event_date)

    ts_by_mod = sample["metadata"]["timestamps"]
    temporal_mods = [k for k, v in ts_by_mod.items() if len(v) > 1]
    if not temporal_mods:
        print(f"[skip] {sid}: no temporal modality with T>1")
        return None
    anchors_by_mod = sample["metadata"].get("anchors") or {}

    for mod in temporal_mods:
        timestamps = ts_by_mod[mod]
        anchors = anchors_by_mod.get(mod, {})

        if len(timestamps) < n_frames:
            print(f"[skip] {sid}: only {len(timestamps)} frames for {mode} ({mod})")
            return None

        offset = torch.from_numpy(np.asarray(
            (_naive(timestamps) - event) / np.timedelta64(1, "D"), dtype=np.float32))

        cloud = sample["metadata"].get("cloud_frac") if "S2" in mod.upper() else None
        cloud = cloud if cloud is not None else torch.zeros_like(offset)
        score = cloud + 0.01 * offset.abs()

        chosen, used, source = {}, set(), {}
        for slot in [s for s in _PICK_ORDER if s in slots]:
            i = _anchor_frame_idx(timestamps, anchors.get(slot))
            source[slot] = "anchor" if (i is not None and i not in used) else "heuristic"
            if i is None or i in used:
                i = _pick_slot(slot, offset, score, used)
            if i is None:
                print(f"[skip] {sid}: no frame left for slot {slot} ({mod})")
                return None
            chosen[slot] = i
            used.add(i)

        # Rungs that take more than their named slots close the widest gaps in
        # what the slots left uncovered. Ranked by the order they were picked,
        # so the audit below shows which gap each one went to.
        for rank, i in enumerate(_fill_spread(offset, score, used, n_fill), start=1):
            chosen[f"spread_{rank}"] = i
            source[f"spread_{rank}"] = "spread"

        idx = torch.tensor(sorted(chosen.values()), dtype=torch.long)

        # Record of the selection, per modality: which slot got which date, and
        # whether it came from the ImpactMesh anchor or from the heuristic.
        # Without it the finished batch no longer shows whether quad-temporal
        # really is the original series -- and that is the claim the 1/2/4/16
        # comparison rests on. Sen12Landslides has no anchors, so everything
        # reads "heuristic" there, which the paper's table has to state.
        sample["metadata"].setdefault("selected_frames", {})[mod] = {
            slot: {"date": str(pd.Timestamp(timestamps[i]).date()),
                   "offset_days": round(float(offset[i]), 1),
                   "source": source[slot]}
            for slot, i in sorted(chosen.items(), key=lambda kv: kv[1])
        }

        sample["image"][mod] = sample["image"][mod][idx]
        sample["metadata"]["timestamps"][mod] = timestamps[idx.numpy()]
        # No positional field is subset here: `rel_day` is derived from the
        # SELECTED timestamps afterwards (see relative_days), so there is
        # nothing that could fall out of sync with the frames.
        if mod in sample["metadata"].get("date_mask", {}):
            sample["metadata"]["date_mask"][mod] = sample["metadata"]["date_mask"][mod][idx]
        if "S2" in mod.upper() and "cloud_frac" in sample["metadata"]:
            sample["metadata"]["cloud_frac"] = sample["metadata"]["cloud_frac"][idx]
        # The per-pixel mask is tied to the same timesteps as the S2 image.
        if "S2" in mod.upper() and "cloud_mask" in sample:
            sample["cloud_mask"] = sample["cloud_mask"][idx]

        n = sample["image"][mod].shape[0]
        assert n == n_frames, f"{sid}/{mod}: {n} frames for mode {mode} ({n_frames} expected)"
        got = len(sample["metadata"]["timestamps"][mod])
        assert got == n, f"{sid}/{mod}: timestamps has {got} entries, image has {n}"

    return sample


def add_event_targets(sample: dict) -> dict:
    """Attach the temporal event target: per modality, is frame t after the event?

    Runs on the same relative axis as the model input: `rel_day` and
    `event_day` are both counted from the sample's first frame, so the target
    is a plain threshold and the error the metrics report stays in days.

    The target is a STEP function `(T,)` in {0, 1}, not a one-hot on the first
    post-event frame. Three reasons:

    1. It stays defined when the event falls outside the loaded window (all-zero
       or all-one), which one-hot does not -- and any windowing/jitter produces
       exactly that case.
    2. The loss is not carried by a single positive among 15-16 frames.
    3. The quantity we actually report -- WHEN did it happen, in days -- is the
       transition point of the step, which is well defined even if no single
       frame is classified correctly.

    `event_valid` is 0.0 for samples without an event date (Sen12Landslides has
    them) or where the event lies outside the observed window on one side; the
    loss and the metrics skip those instead of training on a degenerate target.

    Nothing here is fed back into the input.
    """
    meta = sample["metadata"]
    event_day = meta.get("event_day")
    rel_by_mod = meta["rel_day"]

    step, first_post, valid = {}, {}, {}
    for mod, rel in rel_by_mod.items():
        rel = rel.float()
        if event_day is None:
            step[mod] = torch.zeros_like(rel)
            first_post[mod] = torch.tensor(-1, dtype=torch.long)
            valid[mod] = torch.tensor(0.0)
            continue
        post = rel >= float(event_day)
        step[mod] = post.float()
        # -1 marks "no post-event frame in this window"; the metrics drop those
        # rather than silently scoring the clamp.
        idx = int(torch.nonzero(post)[0]) if bool(post.any()) else -1
        first_post[mod] = torch.tensor(idx, dtype=torch.long)
        # Both sides must be observed, otherwise the transition is not bracketed
        # and "localisation error in days" has no finite ground truth.
        valid[mod] = torch.tensor(float(bool(post.any()) and bool((~post).any())))

    sample["event_step"] = step
    sample["event_first_post"] = first_post
    sample["event_valid"] = valid
    sample["event_day"] = torch.tensor(
        float("nan") if event_day is None else float(event_day))
    return sample


def metadata_collate(batch: list[dict]) -> dict:
    """Collate that pops `metadata` and lists it instead of stacking."""
    metadata = [b.pop("metadata") for b in batch]
    out = default_collate(batch)
    out["metadata"] = metadata
    return out


def print_metadata(sample: dict, task: str | None = None):
    meta = sample["metadata"]

    task_str = task.upper() if task else "UNKNOWN"
    print("=" * 70)
    print(f"  ID:         {meta.get('id', 'unknown')}")
    print(f"  Task:       {task_str}")
    print(f"  Event date: {meta.get('event_date', 'None')}")
    print("=" * 70)

    mask = sample.get("mask")
    if isinstance(mask, torch.Tensor):
        pos = (mask > 0).float().mean().item()
        print(f"  Mask:   shape={tuple(mask.shape)}  pos_frac={pos:.3f}")

    print(f"\n  Image modalities:")
    for mod, x in sample.get("image", {}).items():
        nan = torch.isnan(x).sum().item() if isinstance(x, torch.Tensor) else 0
        print(f"    {mod:8s} shape={tuple(x.shape)}  "
              f"mean={x.mean():+.3f}  std={x.std():.3f}  nan={nan}")

    timestamps = meta.get("timestamps", {})
    days = meta.get("days", {})
    if timestamps or days:
        print(f"\n  Per-modality temporal info:")
        for mod in sorted(set(timestamps.keys()) | set(days.keys())):
            print(f"    {mod}:")
            if mod in timestamps:
                ts = pd.to_datetime(timestamps[mod])
                print(f"      timestamps: {ts[0].strftime('%Y-%m-%d')} → "
                      f"{ts[-1].strftime('%Y-%m-%d')}  (T={len(ts)})")
            if mod in days:
                d = days[mod]
                vals = list(d.values()) if isinstance(d, dict) else d
                d_arr = vals.cpu().numpy() if isinstance(vals, torch.Tensor) else np.asarray(vals)
                rounded = [round(float(v)) for v in d_arr.flat]
                print(f"      days:       {rounded}")

    if "cloud_frac" in meta:
        cf = meta["cloud_frac"]
        cf_arr = cf.cpu().numpy() if isinstance(cf, torch.Tensor) else np.asarray(cf)
        clear = (cf_arr < 0.1).sum()
        print(f"\n  Cloud fraction (S2):")
        print(f"    mean={cf_arr.mean():.3f}  max={cf_arr.max():.3f}  "
              f"clear (<0.1) = {clear}/{len(cf_arr)}")
        print(f"    per-timestep: {[f'{x:.2f}' for x in cf_arr]}")

    if "rel_day" in meta:
        rd = meta["rel_day"]
        print(f"\n  Relative days:")
        if isinstance(rd, dict):
            for mod, d in rd.items():
                vals = list(d.values()) if isinstance(d, dict) else d
                d_arr = vals.cpu().numpy() if isinstance(vals, torch.Tensor) else np.asarray(vals)
                rounded = [round(float(v)) for v in d_arr.flat]
                print(f"    {mod}: {rounded}")
        else:
            d_arr = rd.cpu().numpy() if isinstance(rd, torch.Tensor) else np.asarray(rd)
            rounded = [round(float(v)) for v in d_arr.flat]
            print(f"    {rounded}")

    if "doy" in meta:
        doy = meta["doy"]
        print(f"\n  Day of year:")
        if isinstance(doy, dict):
            for mod, d in doy.items():
                vals = list(d.values()) if isinstance(d, dict) else d
                d_arr = vals.cpu().numpy() if isinstance(vals, torch.Tensor) else np.asarray(vals)
                rounded = [round(float(v)) for v in d_arr.flat]
                print(f"    {mod}: {rounded}")
        else:
            d_arr = doy.cpu().numpy() if isinstance(doy, torch.Tensor) else np.asarray(doy)
            rounded = [round(float(v)) for v in d_arr.flat]
            print(f"    {rounded}")

    other = set(meta.keys()) - {"id", "event_date", "timestamps", "days", "cloud_frac", "doy", "rel_day"}
    if other:
        print(f"\n  Other metadata keys: {sorted(other)}")

    print("=" * 70)

# TODO Update prediction to 3 classes in case of the flood
def plot_sample(
    sample: dict,
    s2_rgb_indices: tuple[int, ...] = (2, 1, 0),
    task: str = "flood",
    prediction: torch.Tensor | np.ndarray | None = None,
    save_path: str | Path | None = None,
):
    """Plot one sample with all timesteps per modality.

    Columns: T temporal frames + N terrain feature cols + Mask [+ Prediction + TP/FP/FN].

    Mask and Prediction share the same discrete task-specific colors
    (TASK_MASK_COLORS[task]) — for flood this distinguishes permanent water (1)
    from flooded (2) in both columns, so the two are directly comparable by eye.
    TP/FP/FN uses POSITIVE_CLASS[task] to identify the hazard class in both mask
    and prediction (not a hardcoded class-1 check), so it scores correctly even
    when the hazard is class 2 (flood) and class 1 is a separate non-hazard
    class (permanent water).

    Args:
        sample: Dict with "image", "mask", "metadata".
        s2_rgb_indices: Channel indices for S2 RGB.
        task: One of the keys in TASK_MASK_COLORS (e.g. "flood", "fire"). Unknown
            values fall back to a binary mask visualization.
        prediction: Optional [H, W] or [C, H, W] tensor. If [C, H, W] with C>1, argmax
            is taken; if [1, H, W] the leading dim is squeezed. Values are assumed
            to be class labels in the same range as the mask (e.g. flood: 0/1/2).
        save_path: If provided, save and close. Else show.
    """
    image = sample["image"]
    mask = sample["mask"].cpu().numpy() if isinstance(sample["mask"], torch.Tensor) else sample["mask"]
    meta = sample["metadata"]

    event_date_raw = meta.get("event_date")
    event_date = pd.to_datetime(event_date_raw) if event_date_raw else None

    temporal_mods = [k for k in image if k.upper() not in STATIC_MODS]
    static_mods = [k for k in image if k.upper() in STATIC_MODS]

    # Prediction handling
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
    positive_class = POSITIVE_CLASS.get(task, 1)  # fallback: binary tasks use 1

    for row, mod in enumerate(temporal_mods):
        img = image[mod].cpu().numpy()
        T_mod = img.shape[0]
        timestamps = meta.get("timestamps", {}).get(mod)
        days = meta.get("days", {}).get(mod)
        cloud = meta.get("cloud_frac") if mod.upper().startswith("S2") else None
        is_s2 = mod.upper().startswith("S2")

        # Find first post-event timestep for highlighting
        first_post = None
        if event_date is not None and timestamps is not None:
            ts = pd.to_datetime(timestamps)
            post = np.where(ts > event_date)[0]
            first_post = post[0] if len(post) else None

        # Temporal frames
        for t in range(T):
            ax = axes[row, t]
            if t >= T_mod:
                ax.axis("off")
                continue

            _render_frame(ax, img[t], s2_rgb_indices, is_s2)
            ax.axis("off")

            lines = []
            if timestamps is not None:
                lines.append(pd.Timestamp(timestamps[t]).strftime("%Y-%m-%d"))
            if days is not None:
                lines.append(f"d={float(days[t]):.0f}")
            if cloud is not None:
                cf = cloud[t].item() if isinstance(cloud, torch.Tensor) else float(cloud[t])
                if cf > 0:
                    lines.append(f"☁{cf:.0%}")

            if t == first_post:
                ax.add_patch(Rectangle(
                    (0, 0), 1, 1, transform=ax.transAxes,
                    linewidth=3, edgecolor="red", facecolor="none",
                    clip_on=False, zorder=100,
                ))
                ax.set_title("\n".join(lines), fontsize=6, color="red", fontweight="bold")
            else:
                ax.set_title("\n".join(lines), fontsize=6)

        # Static modalities — one column per terrain channel
        col = T
        for sm in static_mods:
            sdata = image[sm].cpu().numpy()  # [1, C, H, W]
            channels_spec = TERRAIN_CHANNELS.get(sm.upper())

            if channels_spec is None:
                ax = axes[row, col]
                _render_frame(ax, sdata[0], s2_rgb_indices, is_s2=False)
                if row == 0:
                    ax.set_title(sm, fontsize=7, fontweight="bold")
                ax.axis("off")
                col += 1
            else:
                for name, ch_idx, cmap in channels_spec:
                    if ch_idx >= sdata.shape[1]:
                        continue
                    ax = axes[row, col]
                    ax.imshow(_normalize_percentile(sdata[0, ch_idx]), cmap=cmap)
                    if row == 0:
                        ax.set_title(name, fontsize=7, fontweight="bold")
                    ax.axis("off")
                    col += 1

        # Mask — discrete task colors (raw class values)
        axes[row, col].imshow(mask, cmap=mask_cmap, norm=mask_norm, interpolation="nearest")
        if row == 0:
            axes[row, col].set_title("Mask", fontsize=7, fontweight="bold")
        axes[row, col].axis("off")
        col += 1

        # Prediction columns (only if provided)
        if has_pred:
            # Same discrete cmap as the Mask column -- for flood this renders
            # predicted background/water/flood in the exact colors used for the
            # ground truth, instead of collapsing water+flood into one "positive" color.
            axes[row, col].imshow(pred, cmap=mask_cmap, norm=mask_norm, interpolation="nearest")
            if row == 0:
                axes[row, col].set_title("Prediction", fontsize=7, fontweight="bold")
            axes[row, col].axis("off")
            col += 1

            # TP/FP/FN — hazard class defined by POSITIVE_CLASS[task], compared
            # against the prediction's own positive class (not a hardcoded ==1,
            # which would silently miscount "predicted water" as a flood hit).
            # Background (TN) uses the task's background color rather than black,
            # so it reads as "correctly empty" instead of "no data"; ignore_index=-1
            # pixels are excluded from scoring and shown in the task's ignore color,
            # matching the Mask column.
            ignore = (mask == -1)
            gt_pos = (mask == positive_class)
            pred_pos = (pred == positive_class)

            bg_color = np.array(to_rgb(TASK_MASK_COLORS.get(task, {}).get(0, "#dddddd")))
            ignore_color = np.array(to_rgb(TASK_MASK_COLORS.get(task, {}).get(-1, "#000000")))

            tpfpfn = np.tile(bg_color, (*pred.shape, 1))
            tpfpfn[pred_pos & gt_pos]            = [0, 1, 0]  # TP green
            tpfpfn[pred_pos & ~gt_pos & ~ignore] = [1, 0, 0]  # FP red
            tpfpfn[~pred_pos & gt_pos]           = [0, 0, 1]  # FN blue
            tpfpfn[ignore]                       = ignore_color
            axes[row, col].imshow(tpfpfn, interpolation="nearest")
            if row == 0:
                axes[row, col].set_title("TP/FP/FN", fontsize=7, fontweight="bold")
            axes[row, col].axis("off")
            col += 1

        # Row label
        axes[row, 0].text(
            -0.15, 0.5, f"{mod}\n(T={T_mod})",
            transform=axes[row, 0].transAxes,
            fontsize=9, fontweight="bold", va="center", ha="right", rotation=90,
        )

    sid = meta.get("id", "unknown")
    title_parts = [f"{sid}"]
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