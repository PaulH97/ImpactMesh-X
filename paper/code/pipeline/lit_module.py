import csv
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
from lightning.pytorch import LightningModule
from hydra.utils import instantiate
from torchmetrics import MetricCollection
from torchmetrics.classification import (
    MulticlassAccuracy,
    MulticlassJaccardIndex,
    MulticlassF1Score,
    MulticlassPrecision,
    MulticlassRecall,
)
import segmentation_models_pytorch as smp


class DiceCELoss(nn.Module):
    def __init__(self, dice_weight=0.5, ce_weight=0.5, ignore_index=-1, dice_classes=None):
        super().__init__()
        self.dice_weight = dice_weight
        self.ce_weight = ce_weight
        self.dice = smp.losses.DiceLoss(
            mode="multiclass",
            ignore_index=ignore_index,
            classes=dice_classes,  # None = all classes (default), [1] = foreground only
        )
        self.ce = nn.CrossEntropyLoss(ignore_index=ignore_index)

    def forward(self, pred, target):
        return self.dice_weight * self.dice(pred, target) + self.ce_weight * self.ce(pred, target)

class SingleClassIoU(MulticlassJaccardIndex):
    """Returns IoU for a single class as a scalar."""
    def __init__(self, num_classes, class_idx, **kwargs):
        super().__init__(num_classes=num_classes, average=None, **kwargs)
        self.class_idx = class_idx

    def compute(self):
        return super().compute()[self.class_idx]

class SingleClassAccuracy(MulticlassAccuracy):
    def __init__(self, num_classes, class_idx, **kwargs):
        super().__init__(num_classes=num_classes, average=None, **kwargs)
        self.class_idx = class_idx

    def compute(self):
        return super().compute()[self.class_idx]

# ═════════════════════════════════════════════════════════════════════════
# LIGHTNING MODULE
# ═════════════════════════════════════════════════════════════════════════

class LitSegmentation(LightningModule):
    """Segmentation module with per-class IoU/accuracy metrics, generalized to any num_classes."""

    def __init__(
        self,
        net: torch.nn.Module,
        num_classes: int = 2,
        loss: dict = None,
        optimizer: dict = None,
        scheduler: dict = None,
        ignore_index: int = -1,
        monitor_metric: str = "val/mIoU",
        compile: bool = False,
        seed: int = 42,
        reliability_loss_weight: float = 0.0,
        warmup_epochs: int = 0,
        warmup_start_factor: float = 0.01,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=['net'])
        self.net = torch.compile(net) if compile else net
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        # Gated fusion only: supervises the reliability head against the per-pixel
        # cloud mask. 0.0 leaves the head fully self-supervised (it then has to
        # discover occlusion from the segmentation loss alone). The mask is a
        # TRAINING signal only -- nothing downstream needs it at inference.
        self.reliability_loss_weight = reliability_loss_weight

        self.loss_fn = instantiate(loss)

        # ── Metrics (identical to TerraTorch configure_metrics, plus per-class IoU/Accuracy) ──
        metric_dict = {
            "mIoU": MulticlassJaccardIndex(
                num_classes=num_classes, ignore_index=ignore_index, average="macro"),
            "mIoU_Micro": MulticlassJaccardIndex(
                num_classes=num_classes, ignore_index=ignore_index, average="micro"),
            "F1_Score": MulticlassF1Score(
                num_classes=num_classes, ignore_index=ignore_index, average="macro"),
            "Accuracy": MulticlassAccuracy(
                num_classes=num_classes, ignore_index=ignore_index, average="macro"),
            "Pixel_Accuracy": MulticlassAccuracy(
                num_classes=num_classes, ignore_index=ignore_index, average="micro"),
            "Precision": MulticlassPrecision(
                num_classes=num_classes, ignore_index=ignore_index, average="macro"),
            "Recall": MulticlassRecall(
                num_classes=num_classes, ignore_index=ignore_index, average="macro"),
        }
        for c in range(num_classes):
            metric_dict[f"IoU_{c}"] = SingleClassIoU(
                num_classes=num_classes, class_idx=c, ignore_index=ignore_index)
            metric_dict[f"Class_Accuracy_{c}"] = SingleClassAccuracy(
                num_classes=num_classes, class_idx=c, ignore_index=ignore_index)
        metrics = MetricCollection(metric_dict)
        self.train_metrics = metrics.clone(prefix="train/")
        self.val_metrics = metrics.clone(prefix="val/")
        self.test_metrics = metrics.clone(prefix="test/")

        # Per-sample dump of the test run. The path is set by train.py/predict.py;
        # if it stays None, `on_test_epoch_end` falls back to trainer.log_dir.
        self.per_sample_path: str | Path | None = None
        self._test_rows: list[dict] = []
        self._warned_missing_id = False

    def forward(self, x):
        return self.net(x)

    def _reliability_loss(self, reliability_logits: dict, cloud_mask: torch.Tensor):
        """BCE of the per-pixel, per-timestep reliability head against the observed
        cloud mask.

        The head predicts at the encoder's deepest resolution (H/8) while the mask
        is full resolution, so the mask is average-pooled down: the target is the
        occluded FRACTION of each cell in [0, 1], not a hard label, and BCE with a
        soft target is the right fit. Reliability is the complement of occlusion,
        hence `1 - cloud`.
        """
        total, n = 0.0, 0
        for logits in reliability_logits.values():          # (B, T, h, w)
            if logits.shape[1] != cloud_mask.shape[1]:      # T mismatch -> not this modality's mask
                continue
            target = F.adaptive_avg_pool2d(cloud_mask, logits.shape[-2:])
            total = total + F.binary_cross_entropy_with_logits(
                logits, (1.0 - target).clamp(0.0, 1.0))
            n += 1
        return total / n if n else None

    def _shared_step(self, batch, stage, metrics):
        inputs = batch["image"]
        targets = batch["mask"].long()
        metadata = batch.get("metadata", None)

        out = self.net(inputs, metadata=metadata)
        logits = out["segmentation"]
        loss = self.loss_fn(logits, targets)

        preds = logits.argmax(dim=1)
        bs = targets.shape[0]

        rel_logits = out.get("reliability_logits")
        cloud_mask = batch.get("cloud_mask")
        if self.reliability_loss_weight > 0 and rel_logits and cloud_mask is not None:
            aux = self._reliability_loss(rel_logits, cloud_mask.float())
            if aux is not None:
                loss = loss + self.reliability_loss_weight * aux
                self.log(f"{stage}/reliability_loss", aux, batch_size=bs,
                         on_step=False, on_epoch=True, sync_dist=False)

        metrics.update(preds, targets)
        self.log(f"{stage}/loss", loss, batch_size=bs, on_step=False, on_epoch=True, prog_bar=True, sync_dist=False)
        self.log_dict(metrics, batch_size=bs, on_step=False, on_epoch=True, sync_dist=False)

        if stage == "test":
            self._collect_per_sample(preds, targets, metadata)

        return loss

    def training_step(self, batch, batch_idx):
        return self._shared_step(batch, "train", self.train_metrics)

    def validation_step(self, batch, batch_idx):
        return self._shared_step(batch, "val", self.val_metrics)

    def on_validation_epoch_end(self):
        metrics = self.trainer.callback_metrics
        keys = ["val/loss", "val/mIoU", "val/IoU_1", "val/IoU_2", "val/F1_Score"]
        parts = [f"{k.split('/')[1]}={metrics[k].item():.4f}" for k in keys if k in metrics]
        self.print(f"[epoch {self.current_epoch}] " + " | ".join(parts))

    # ── Per-sample metrics ───────────────────────────────────────────────
    # The reported test mIoU is a POOLED estimator: MulticlassJaccardIndex sums
    # the confusion over the whole test set and only then forms per-class IoU
    # and its mean. A confidence interval for it (bootstrap over AOIs -- the 441
    # flood test patches spread over only 61 AOIs, which is the effective sample
    # size) has to rebuild the same estimator. That is why
    # this stores the sufficient statistics per sample rather than the finished
    # per-sample IoU: drawing a resample, summing the counts within it, then
    # forming the metric reproduces the table number exactly. A mean over
    # per-sample IoUs would not.
    #
    # The full one-vs-rest confusion per class is stored -- tp/fp/fn/tn -- rather
    # than just intersection and union. That lets any metric in the table be
    # rebuilt from the dump afterwards, without rerunning the test:
    #   IoU_c       = tp / (tp + fp + fn)
    #   Precision_c = tp / (tp + fp)          Recall_c = tp / (tp + fn)
    #   F1_c        = 2tp / (2tp + fp + fn)   Accuracy_c = (tp + tn) / n_valid
    # tp + fp + fn + tn == n_valid holds per class; ignore_index pixels are not
    # counted in any of the four numbers.
    #
    # Everything else (AOI, event, split, cloud fraction) hangs off patch_id and
    # is joined in from flood_patch_meta*.parquet -- none of it is duplicated here.

    def _collect_per_sample(self, preds, targets, metadata):
        valid = targets != self.ignore_index
        n_valid = valid.flatten(1).sum(1)

        counts = {}
        for c in range(self.num_classes):
            tgt_c = (targets == c) & valid
            prd_c = (preds == c) & valid
            tp = (tgt_c & prd_c).flatten(1).sum(1)
            union = (tgt_c | prd_c).flatten(1).sum(1)
            counts[c] = (
                tp,
                prd_c.flatten(1).sum(1) - tp,   # FP: predicted, not GT
                tgt_c.flatten(1).sum(1) - tp,   # FN: GT, not predicted
                n_valid - union,                # TN: neither
            )

        for i in range(targets.shape[0]):
            sid = None
            if metadata is not None and i < len(metadata):
                sid = metadata[i].get("id")
            if sid is None and not self._warned_missing_id:
                # The dump is only useful if it can be joined to the patch
                # metadata by id -- a silent fallback key would be exactly the
                # kind of bug that only surfaces at join time.
                self._warned_missing_id = True
                self.print(
                    "WARNING: test sample without metadata id, falling back to a "
                    "positional key -- those rows cannot be joined to the patch metadata."
                )
            row = {"id": sid if sid is not None else f"sample_{len(self._test_rows):05d}",
                   "n_valid": int(n_valid[i])}
            ious = []
            for c, (tp, fp, fn, tn) in counts.items():
                t, p, n = int(tp[i]), int(fp[i]), int(fn[i])
                row[f"tp_{c}"] = t
                row[f"fp_{c}"] = p
                row[f"fn_{c}"] = n
                row[f"tn_{c}"] = int(tn[i])
                if t + p + n > 0:
                    ious.append(t / (t + p + n))
            # Convenience column for quick inspection and sorting by outliers --
            # NOT the table's estimator (see comment above). Classes with no
            # pixels in either prediction or GT are left out.
            row["mIoU_sample"] = round(sum(ious) / len(ious), 6) if ious else ""
            self._test_rows.append(row)

    def on_test_epoch_start(self):
        self._test_rows = []
        self._warned_missing_id = False

    def on_test_epoch_end(self):
        if not self._test_rows:
            return

        path = self.per_sample_path
        if path is None:
            log_dir = getattr(self.trainer, "log_dir", None)
            path = Path(log_dir or ".") / "test_per_sample.csv"
        path = Path(path)
        # Under DDP each rank only sees its shard of the test set -- one file per
        # rank instead of a silently partial dump. Everything currently runs with
        # devices=1, so this branch is pure safeguarding.
        if self.trainer.world_size > 1:
            path = path.with_name(f"{path.stem}_rank{self.trainer.global_rank}{path.suffix}")

        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(self._test_rows[0]))
            writer.writeheader()
            writer.writerows(self._test_rows)
        self.print(f"Saved per-sample test metrics ({len(self._test_rows)} samples) → {path}")
        self._test_rows = []

    def test_step(self, batch, batch_idx, dataloader_idx=0):
        return self._shared_step(batch, "test", self.test_metrics)

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        inputs = batch["image"]
        targets = batch["mask"]
        metadata = batch.get("metadata", None)
        logits = self.net(inputs, metadata=metadata)["segmentation"]
        preds = logits.argmax(dim=1)
        probs = F.softmax(logits, dim=1)
        return {"preds": preds, "targets": targets, "probs": probs}

    def configure_optimizers(self):
        optimizer = instantiate(self.hparams.optimizer, params=self.parameters())
        if self.hparams.scheduler:
            sched_cfg = dict(self.hparams.scheduler)
            warmup_epochs = int(self.hparams.warmup_epochs)

            # The warmup eats into the run, so the cosine has to be shortened by the
            # same amount -- otherwise it never reaches eta_min.
            if warmup_epochs > 0 and "T_max" in sched_cfg:
                sched_cfg["T_max"] = max(1, int(sched_cfg["T_max"]) - warmup_epochs)

            scheduler = instantiate(sched_cfg, optimizer=optimizer)

            if warmup_epochs > 0:
                warmup = torch.optim.lr_scheduler.LinearLR(
                    optimizer,
                    start_factor=self.hparams.warmup_start_factor,
                    end_factor=1.0,
                    total_iters=warmup_epochs,
                )
                scheduler = torch.optim.lr_scheduler.SequentialLR(
                    optimizer,
                    schedulers=[warmup, scheduler],
                    milestones=[warmup_epochs],
                )

            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": self.hparams.monitor_metric,
                    "interval": "epoch",
                    "frequency": 1,
                },
            }
        return optimizer