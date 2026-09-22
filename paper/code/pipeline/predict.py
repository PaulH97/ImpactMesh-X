import argparse
import torch
import gc
import traceback
import numpy as np
from pathlib import Path
from lightning import seed_everything, Trainer
from omegaconf import OmegaConf, open_dict
from torch.utils.data import DataLoader, Subset
import json
import random
import shutil
from hydra.utils import instantiate, get_class
import inspect

from fusion.data.utils import metadata_collate

import fusion.resolvers  # noqa: F401  -- registers the OmegaConf resolvers


def instantiate_compatible(node, **kwargs):
    """Like hydra.utils.instantiate, but drops config keys the *current*
    target's __init__ no longer accepts (with a printed note) instead of
    raising a TypeError.

    Meant for reloading an old run's frozen .hydra/config.yaml (predict.py,
    analysis scripts), where the constructor signature may have changed since
    that run was launched -- e.g. a param removed after the config snapshot
    was written. train.py should keep calling instantiate() directly so a
    real config typo on a *fresh* run still fails loudly.

    Only handles pure removals: a renamed param is a different key and gets
    dropped like any other stale one, not remapped.
    """
    target = get_class(node["_target_"])
    params = inspect.signature(target.__init__).parameters
    if any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values()):
        return instantiate(node, **kwargs)  # __init__(**kwargs) already accepts anything

    accepted = set(params) - {"self"}
    stale = [k for k in node.keys() if k != "_target_" and k not in accepted]
    if not stale:
        return instantiate(node, **kwargs)

    print(f"[instantiate_compatible] {node['_target_']}: dropping stale config keys {stale} "
          f"(not in current __init__)")
    # Resolve interpolations (e.g. dataset_cfg: ${dataset}) *before* dropping keys --
    # node is still attached to its parent here, so cross-references into the rest
    # of the loaded config still work; a detached copy (e.g. masked_copy) would break them.
    resolved = OmegaConf.to_container(node, resolve=True)
    for k in stale:
        resolved.pop(k, None)
    return instantiate(resolved, **kwargs)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", type=str, required=True)
    parser.add_argument("--n_vis", type=int, default=10)
    parser.add_argument(
        "--test_s2_version", type=str, default=None, choices=["clear", "cloudy", "realistic"],
        help="Test the run's checkpoint against a different S2 variant than it was "
             "trained on (experiment 4, test-time shift axis). Results go to "
             "test_metrics_<version>.json / predictions_<version>/ so the "
             "matched-condition results of the run stay untouched.")
    parser.add_argument("--force", action="store_true",
                        help="Recompute even if the metrics file already exists.")
    parser.add_argument(
        "--save_all_predictions", action="store_true",
        help="Also run inference over the *entire* test split (not just n_vis) and dump "
             "every sample's predicted class-label map to a single predictions_full<suffix>.npz "
             "in model_dir (keys = sample id, values = uint8 [H, W] argmax labels). Meant to "
             "front-load GPU inference for figure notebooks (e.g. predictions.ipynb's appendix "
             "candidate browser), which can then just load the cached array instead of "
             "reloading the checkpoint and re-running inference on CPU per sample.")
    args = parser.parse_args()

    model_dir = Path(args.model_dir)
    cfg = OmegaConf.load(model_dir / ".hydra" / "config.yaml")
    trained_version = cfg.dataset.get("s2_version", None)
    shift = args.test_s2_version is not None and args.test_s2_version != trained_version
    if args.test_s2_version is not None:
        if trained_version is None:
            raise SystemExit(f"[PREDICT] {model_dir.name} has no dataset.s2_version -- "
                             "--test_s2_version only applies to ImpactMesh long-version runs")
        cfg.dataset.s2_version = args.test_s2_version

    suffix = f"_{args.test_s2_version}" if shift else ""
    metrics_path = model_dir / f"test_metrics{suffix}.json"
    vis_dir = model_dir / f"predictions{suffix}"
    full_preds_path = model_dir / f"predictions_full{suffix}.npz"

    metrics_done = metrics_path.exists()
    full_preds_done = full_preds_path.exists() if args.save_all_predictions else True
    if metrics_done and full_preds_done and not args.force:
        print(f"[PREDICT] Test metrics already exist → {metrics_path}")
        return

    if shift:
        print(f"[PREDICT] Test-time shift: trained on '{trained_version}' → "
              f"testing on '{args.test_s2_version}'")

    with open_dict(cfg.datamodule):
        cfg.datamodule.pop("batch_size", None)
        cfg.datamodule.pop("num_workers", None)

    try:
        torch.set_float32_matmul_precision("high")
        seed_everything(cfg.seed, workers=True)

        is_terratorch = cfg.lit_module.get("model_factory", None) is not None

        # Build model + load checkpoint
        if is_terratorch:
            lit_module = instantiate_compatible(cfg.lit_module, _convert_="all", _recursive_=False)
        else:
            model = instantiate_compatible(cfg.model.instance, _convert_="all")
            lit_module = instantiate_compatible(cfg.lit_module, net=model, _convert_="all", _recursive_=False)

        datamodule = instantiate_compatible(cfg.datamodule)

        ckpt_path = model_dir / "checkpoints" / "best.ckpt"
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        missing, unexpected = lit_module.load_state_dict(ckpt["state_dict"], strict=False)
        if missing:
            print(f"[PREDICT] Missing keys ({len(missing)}): {missing[:5]}{'...' if len(missing) > 5 else ''}")
        if unexpected:
            print(f"[PREDICT] Unexpected keys ({len(unexpected)}): {unexpected[:5]}{'...' if len(unexpected) > 5 else ''}")

        # ----- 1. Global test metrics -----
        device = "cuda" if torch.cuda.is_available() else "cpu"
        datamodule.setup(stage="test")

        trainer = Trainer(accelerator=device, devices=1, logger=False)
        if hasattr(lit_module, "per_sample_path"):
            lit_module.per_sample_path = model_dir / f"test_per_sample{suffix}.csv"
        test_results = trainer.test(lit_module, datamodule=datamodule, verbose=False)
        metrics = {k: float(v) for k, v in test_results[0].items()}
        metrics_with_meta = {
            "exp_tag": cfg.get("exp_tag", model_dir.name),
            "dataset": cfg.dataset.name,
            "model": cfg.model.get("name", "unknown"),
            "seed": cfg.seed,
            "checkpoint": str(ckpt_path),
            "trained_s2_version": trained_version,
            "tested_s2_version": cfg.dataset.get("s2_version", None),
            "metrics": metrics,
        }
        with open(metrics_path, "w") as f:
            json.dump(metrics_with_meta, f, indent=2)
        print(f"[PREDICT] Saved test metrics (recomputed from best.ckpt) → {metrics_path}")

        # ----- 2. Plot samples -----
        # Predict only on the n_vis chosen samples (not the full test set) -- a
        # Subset + single batch is enough since these are just for visualization.
        test_dataset = datamodule.test_dataset
        n_test = len(test_dataset)
        chosen = random.Random(cfg.seed).sample(range(n_test), k=min(args.n_vis, n_test))

        vis_loader = DataLoader(Subset(test_dataset, chosen), batch_size=len(chosen), collate_fn=metadata_collate)
        vis_out = trainer.predict(lit_module, dataloaders=vis_loader, return_predictions=True)
        probs = torch.cat([o["probs"] for o in vis_out]).cpu()

        if vis_dir.exists():
            shutil.rmtree(vis_dir)
        vis_dir.mkdir(parents=True, exist_ok=True)

        for i, idx in enumerate(chosen):
            sample = test_dataset[idx]
            sid = sample["metadata"]["id"]
            test_dataset.plot(sample, prediction=probs[i], save_path=vis_dir / f"{sid}.png")

        print(f"[PREDICT] Saved {len(chosen)} predictions → {vis_dir}")

        # ----- 3. Full-test-split prediction cache (optional) -----
        # Runs inference once over every test sample (on GPU, batched) and caches the
        # argmax class-label map per sample id in one compressed .npz -- so figure
        # notebooks that need many different samples (e.g. picking appendix candidates)
        # can load predictions instead of re-running the model per sample on CPU.
        if args.save_all_predictions:
            full_loader = DataLoader(
                test_dataset, batch_size=16, shuffle=False,
                num_workers=4, collate_fn=metadata_collate,
            )
            full_out = trainer.predict(lit_module, dataloaders=full_loader, return_predictions=True)

            ids = [f["id"] for f in test_dataset.files]
            pred_by_id = {}
            offset = 0
            for batch_out in full_out:
                batch_preds = batch_out["preds"].cpu().numpy().astype("uint8")
                for pred in batch_preds:
                    pred_by_id[ids[offset]] = pred
                    offset += 1
            assert offset == len(ids), f"expected {len(ids)} predictions, got {offset}"

            np.savez_compressed(full_preds_path, **pred_by_id)
            print(f"[PREDICT] Saved {len(pred_by_id)} full-test-split predictions → {full_preds_path}")

    except Exception:
        traceback.print_exc()
    finally:
        torch.cuda.empty_cache()
        gc.collect()

if __name__ == "__main__":
    main()