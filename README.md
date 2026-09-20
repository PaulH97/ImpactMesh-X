# ImpactMesh-X

ImpactMesh-X extends the original ImpactMesh dataset along two axes: **time** and **robustness**. It benchmarks hazard mapping models on long-term multi-temporal context and atmospheric occlusion.

**Dataset**:
🔗 [https://huggingface.co/datasets/paulhoehn/ImpactMesh-X](https://huggingface.co/datasets/paulhoehn/ImpactMesh-X)

---

## Setup

```bash
# 1. Clone code repo
git clone https://github.com/PaulH97/ImpactMesh-X.git
cd ImpactMesh-X

# 2. Install HF CLI
pip install --upgrade huggingface_hub

# 3. Authenticate (only first time)
hf auth login  # paste your token from https://huggingface.co/settings/tokens

# 4. Pull one hazard into a local folder, e.g. Flood
mkdir -p data
hf download paulhoehn/ImpactMesh-X --repo-type dataset --local-dir data --include "Flood/*"
```

Each modality folder ships as one or more `.tar` archives -- unpack them before running the code:

```bash
# from repo root, for each modality archive under data/<hazard>/:
for archive in data/Flood/*/*.tar data/Flood/*/*/*.tar; do
  [ -f "$archive" ] && tar -xf "$archive" -C "$(dirname "$archive")" && rm "$archive"
done
```

## Data

Each hazard (`Flood`, `Fire`) is a flat folder of per-patch files. Positive and hazard-free negative samples are mixed together, distinguished by the `pool` column in the metadata parquet. Every patch carries 16 timesteps per sensor across a ±180-day window around the event date. `S2L2A` ships three cloud-cover variants of the same reflectance (`clear`/`cloudy`/`realistic`) for robustness studies.

| Hazard | Patches (total) | Positive | Negative | CEMS events | Event dates |
|---|---|---|---|---|---|
| Flood | 9,723 | 4,410 | 5,313 | 191 | 2016 – 2026 |
| Fire  | 10,168 | 4,801 | 5,367 | 215 | 2016 – 2026 |

Official split (positive samples only, see `splits/`):

| Hazard | Train | Val | Test |
|---|---|---|---|
| Flood | 3,510 | 459 | 441 |
| Fire  | 3,433 | 648 | 720 |

```
ImpactMesh-X/<hazard>/
├── S1RTC/                                          # Sentinel-1 RTC, all timesteps
│   └── {patch_id}_S1RTC.zarr.zip
├── S2L2A/                                          # Sentinel-2 L2A, all timesteps
│   ├── clear/{patch_id}_S2L2A.zarr.zip             # cloud-free reference
│   ├── cloudy/{patch_id}_S2L2A.zarr.zip            # raw, with real cloud cover
│   └── realistic/{patch_id}_S2L2A.zarr.zip         # clear + synthetic clouds
├── DEM/                                            # static terrain
│   └── {patch_id}_DEM.tif                          # elevation, slope, aspect
├── MASK/                                           # annotations
│   └── {patch_id}_annotation_{flood,wildfire}.tif  # code's task="fire" reads the "wildfire" mask
└── {hazard}_patch_meta.parquet                     # per-patch metadata + `pool` column
```

## Quickstart

```bash
pip install -r requirements.txt
python example.py --root /path/to/ImpactMesh-X/Flood --task flood
```

`dataset.py`, `datamodule.py`, `transforms.py`, `utils.py` are plain PyTorch/Lightning, no config framework. They return the full 16-timestep stack per modality, subsample downstream if you need fewer.

`splits/` carries the official train/val/test split (based on original ImpactMesh split) and per-band normalization stats used for every hazard-mapping result in the paper -- **positive samples only**. Negative samples have no predefined split, build your own if you need one.

```python
import json
from dataset import ImpactMeshDataset
from datamodule import ImpactMeshDataModule
from transforms import Compose, Normalize, RandomHorizontalFlip, RandomVerticalFlip

ds = ImpactMeshDataset(root="/path/to/ImpactMesh-X/Flood", task="flood", pool="positive")
sample = ds[0]  # {"image": {...}, "mask": ..., "metadata": {...}}

# official split (positive samples only) + normalization stats, shipped in splits/
with open("splits/flood_splits.json") as f:
    split = json.load(f)

dm = ImpactMeshDataModule(
    root="/path/to/ImpactMesh-X/Flood", task="flood", pool="positive",
    train_ids=split["train"], val_ids=split["val"], test_ids=split["test"],
    train_transforms=Compose([
        RandomHorizontalFlip(), RandomVerticalFlip(),
        Normalize("splits/flood_norm.json"),
    ]),
)
```

See `example.py` for a runnable CLI script (split, batch, saved sample plot), or `visualize_sample.ipynb` for the same flow interactively.

## Reproducing the paper

`paper/` holds the Hydra configs, training/prediction scripts, and model code behind every paper result (454 runs, 112 config combinations). It builds on the dataset code above, adding event-anchor-aware timestep selection, the U-TAE fusion model with early/mid/late/decision fusion, and baselines (ConvLSTM, TSViT, UNet3D).

```bash
cd paper
pip install -r requirements.txt
pip install -e .   # installs paper/code as the `fusion` package used by configs and scripts
python code/pipeline/train.py experiment=fusion/mid dataset=im_flood_s12
python code/pipeline/predict.py --model_dir <path to a finished run>
```
