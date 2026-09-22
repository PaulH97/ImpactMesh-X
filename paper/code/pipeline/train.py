import hydra
import torch
import gc
import logging
import traceback
import wandb
from hydra.utils import instantiate
from lightning import seed_everything
from hydra.core.hydra_config import HydraConfig
from pathlib import Path
import warnings

import fusion.resolvers  # noqa: F401  -- registers the OmegaConf resolvers

warnings.filterwarnings("ignore", message="Grad strides do not match bucket view strides")
warnings.filterwarnings("ignore", message="The epoch parameter in `scheduler.step\\(\\)` was not necessary")


def check_effective_batch(cfg) -> None:
    """Abort before training if the effective batch drifts from `effective_batch_size`.

    The effective batch is assembled across three config layers -- `datamodule.batch_size`,
    `trainer.accumulate_grad_batches` (which by default reads `dataset.accum_steps`), and
    the device count -- and the benchmark configs override the middle one per architecture.
    A drift there produces runs that train fine but are no longer comparable to the rest of
    the table, which is exactly the kind of thing that only surfaces once the numbers look
    odd. Set `effective_batch_size=null` to bypass.

    Raises SystemExit rather than a plain exception: `main` swallows Exception, so anything
    else would be logged and still exit 0.
    """
    target = cfg.get("effective_batch_size", None)
    if target is None:
        return
    if cfg.trainer.get("fast_dev_run", False):
        return

    devices = cfg.trainer.get("devices", 1)
    if not isinstance(devices, int) or devices < 1:
        logging.warning(
            f"Skipping effective-batch check: trainer.devices={devices!r} is not a fixed "
            f"count, so the effective batch cannot be determined from the config."
        )
        return

    batch = cfg.datamodule.batch_size
    accum = cfg.trainer.get("accumulate_grad_batches", 1) or 1
    effective = batch * accum * devices
    logging.info(
        f"Effective batch = {batch} (batch_size) x {accum} (accumulate) x {devices} "
        f"(devices) = {effective}"
    )
    if effective != target:
        raise SystemExit(
            f"Effective batch is {effective}, expected {target}: batch_size={batch}, "
            f"accumulate_grad_batches={accum}, devices={devices}. Fix the experiment or "
            f"dataset config, or set effective_batch_size=null to bypass deliberately."
        )


@hydra.main(config_path="../../configs", config_name="config", version_base=None)
def main(cfg):
    check_effective_batch(cfg)
    try:
        torch.set_float32_matmul_precision("high")
        seed_everything(cfg.seed, workers=True)

        is_terratorch = cfg.lit_module.get("model_factory", None) is not None

        datamodule = instantiate(cfg.datamodule)

        if is_terratorch:
            lit_module = instantiate(cfg.lit_module, _convert_="all", _recursive_=False)
        else:
            model = instantiate(cfg.model.instance, _convert_="all")
            lit_module = instantiate(cfg.lit_module, net=model, _convert_="all", _recursive_=False)

        trainer = instantiate(cfg.trainer, _convert_="all")

        if cfg.get("sanity_check", False):
            datamodule.setup()
            if trainer.global_rank == 0:
                output_dir = Path(HydraConfig.get().runtime.output_dir) / "sanity_check"
                output_dir.mkdir(parents=True, exist_ok=True)

                n_samples = 10
                ds = datamodule.train_dataset
                ds.print_metadata(0)

                for i in range(min(n_samples, len(ds))):
                    sample = ds[i]
                    sid = sample["metadata"].get("id", f"sample_{i:04d}")
                    ds.plot(sample, save_path=output_dir / f"{sid}.png")

                logging.info(f"Saved {min(n_samples, len(ds))} sanity samples → {output_dir}")

            if torch.distributed.is_initialized():
                torch.distributed.barrier()

        resume_from = cfg.get("resume_from", None)
        if resume_from:
            logging.info(f"Resuming from checkpoint → {resume_from}")

        trainer.fit(lit_module, datamodule=datamodule, ckpt_path=resume_from)

        if hasattr(lit_module, "per_sample_path"):
            lit_module.per_sample_path = Path(HydraConfig.get().runtime.output_dir) / "test_per_sample.csv"

        test_ckpt_path = None if trainer.fast_dev_run else "best" # for fast_dev_run
        trainer.test(lit_module, datamodule=datamodule, ckpt_path=test_ckpt_path)
        logging.info("Finished run")

    except Exception as e:
        traceback.print_exc()
        logging.error(f"Training crashed: {e}")
    finally:
        wandb.finish()
        torch.cuda.empty_cache()
        gc.collect()


if __name__ == "__main__":
    main()