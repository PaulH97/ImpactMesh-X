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
        event_loss_weight: float = 0.0,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=['net'])
        self.net = torch.compile(net) if compile else net
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        # Weight of the temporal event-localisation task. 0.0 disables it, which
        # is what every segmentation-only run in the benchmark tables uses; the
        # head then either does not exist or receives no gradient.
        self.event_loss_weight = event_loss_weight
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

        # Per-Sample-Dump des Testlaufs. Der Pfad wird von train.py/predict.py gesetzt;
        # bleibt er None, faellt `on_test_epoch_end` auf trainer.log_dir zurueck.
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

    # ── Temporale Event-Lokalisierung ────────────────────────────────────
    # Das Ziel ist eine Stufenfunktion je Modalitaet (siehe add_event_targets).
    #
    # Berichtet wird in ZWEI Einheiten, und die Reihenfolge ist Absicht:
    #
    # 1. Fehler in FRAMES, auf der eigenen Zeitachse der Modalitaet. Das ist die
    #    Groesse, die das Modell ueberhaupt entscheiden kann -- es waehlt einen
    #    Uebergang zwischen zwei Aufnahmen, mehr Aufloesung existiert nicht.
    # 2. Fehler in TAGEN, auf der gemeinsamen Achse aller Modalitaeten.
    #
    # Warum nicht nur Tage: bei IM-Flood liegt der Median-Abstand zweier Aufnahmen
    # bei 20-24 Tagen. Selbst ein PERFEKT getroffener Frame landet im Mittel ~6
    # Tage neben dem wahren Datum, weil der Schaetzer die Mitte des Brackets ist.
    # "MAE 62 Tage" liest sich wie ein kaputtes Modell und heisst in Wahrheit
    # "2-3 Frames daneben". Umgekehrt ist `<=7d` auf dieser Achse STRENGER als
    # "richtiger Frame" und damit als Hauptzahl irrefuehrend.
    #
    # Warum die Tage trotzdem bleiben: nur dort kann Fusion ueberhaupt gewinnen.
    # Die Frames beider Sensoren werden auf eine gemeinsame, zeitlich sortierte
    # Achse gelegt; die ist doppelt so dicht, also kann der Uebergang feiner
    # lokalisiert werden. Ein Index auf dieser Mischachse waere dagegen nicht mit
    # dem Index einer Einzelmodalitaet vergleichbar -- 32 statt 16 Positionen.

    @staticmethod
    def _transition(probs: torch.Tensor, days: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """(B, T) post-event probabilities + (B, T) days -> (predicted day, predicted index).

        Reads the transition as the midpoint between the last frame below 0.5 and
        the first frame at or above it, scanning from the END so a single early
        false positive does not drag the estimate to the front of the series.

        Der Index ist der erste Post-Frame und laeuft bewusst von 0 bis T, nicht
        bis T-1: sagt das Modell fuer KEINEN Frame post-event vorher, ist die
        ehrliche Aussage "das Ereignis liegt hinter der Reihe", nicht "im letzten
        Frame". Der frueher hier stehende Clamp auf T-1 hat genau diesen Fall
        still auf den letzten Termin gelegt und damit einen Teil des beobachteten
        Spaet-Bias erzeugt; der Tag wird jetzt entsprechend ueber das Reihenende
        hinaus extrapoliert (letzter Termin + ein halber Median-Abstand).
        """
        post = probs >= 0.5
        T = probs.shape[1]
        # Index of the first frame from which the series stays post-event.
        rev_all_post = torch.flip(torch.cumprod(torch.flip(post.long(), [1]), dim=1), [1])
        has = rev_all_post.any(dim=1)
        first = torch.where(has, rev_all_post.argmax(dim=1),
                            torch.full_like(has, T, dtype=torch.long))

        at = first.clamp(max=T - 1)
        prev = (first - 1).clamp(min=0, max=T - 1)
        d_at = days.gather(1, at[:, None]).squeeze(1)
        d_prev = days.gather(1, prev[:, None]).squeeze(1)
        gap = ((days[:, 1:] - days[:, :-1]).median(dim=1).values if T > 1
               else torch.zeros_like(d_at))

        day = torch.where(
            first == 0, d_at,                        # Ereignis am oder vor dem ersten Termin
            torch.where(first >= T, d_at + 0.5 * gap,  # Ereignis hinter der Reihe
                        0.5 * (d_at + d_prev)))
        return day, first.float()

    @staticmethod
    def _monotonicity(probs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """(B, T) -> (Anzahl 0/1-Wechsel, kumulierter Rueckwaertsschritt).

        Nichts am Kopf garantiert, dass die Post-Event-Wahrscheinlichkeit ueber T
        steigt -- das Ziel ist eine Stufe, die Vorhersage muss keine sein.
        `_transition` repariert das nachtraeglich, indem es nur den LETZTEN
        zusammenhaengenden Post-Block akzeptiert. Ob dieser Scan ein harmloser
        Schoenheitsfehler oder die halbe Antwort ist, entscheidet sich daran, wie
        oft die Reihe ueberhaupt nicht-monoton ist -- deshalb wird es gemessen und
        nicht angenommen:

          flips > 1   die binaere Reihe wechselt mehr als einmal, es gibt also
                      mehrere Kandidaten-Uebergaenge und der Scan WAEHLT einen aus.
          down_var    Summe aller Rueckwaertsschritte von p. 0 = perfekt monoton,
                      unabhaengig von der Schwelle und damit die feinere Zahl.
        """
        post = probs >= 0.5
        flips = (post[:, 1:] != post[:, :-1]).sum(dim=1)
        down_var = F.relu(probs[:, :-1] - probs[:, 1:]).sum(dim=1)
        return flips, down_var

    def _event_loss_and_stats(self, out, batch, stage, bs):
        """BCE on the step function + localisation error in days. Returns the loss."""
        logits = out.get("event_logits")
        if not logits:
            return None
        target = batch.get("event_step")
        valid = batch.get("event_valid")
        if target is None or valid is None:
            return None

        first_post = batch.get("event_first_post") or {}

        total, n = 0.0, 0
        per_mod = {}
        for m, lg in logits.items():
            if m not in target:
                continue
            v = valid[m].to(lg.device)                       # (B,)
            if v.sum() == 0:
                continue
            tgt = target[m].to(lg.device).float()
            per_sample = F.binary_cross_entropy_with_logits(lg, tgt, reduction="none").mean(dim=1)
            total = total + (per_sample * v).sum() / v.sum()
            n += 1
            per_mod[m] = (torch.sigmoid(lg), out["event_days"][m].to(lg.device).float(), v)
        if n == 0:
            return None
        loss = total / n

        with torch.no_grad():
            # ── Tage, gemeinsame Achse ──
            # Frames aller Sensoren aneinanderhaengen und nach Datum sortieren.
            # Bei einer Einzelmodalitaet ist das ein No-op, bei S12 ist es das
            # dichtere Gitter, auf dem das Fusionsargument beruht.
            p = torch.cat([v[0] for v in per_mod.values()], dim=1)
            d = torch.cat([v[1] for v in per_mod.values()], dim=1)
            order = d.argsort(dim=1)
            pred_day, _ = self._transition(p.gather(1, order), d.gather(1, order))
            true_day = batch["event_day"].to(pred_day.device).float()
            any_valid = torch.stack([v[2] for v in per_mod.values()], 0).amax(0)
            keep = (any_valid > 0) & torch.isfinite(true_day)

            # ── Frames + Monotonie, je Modalitaet auf ihrer eigenen Achse ──
            rows = {}
            idx_err_all, idx_keep_all, flips_all, down_all = [], [], [], []
            for m, (p_m, d_m, v_m) in per_mod.items():
                _, pred_idx = self._transition(p_m, d_m)
                flips, down_var = self._monotonicity(p_m)
                k_m = v_m > 0
                true_idx = first_post.get(m)
                if true_idx is None:
                    err_m = torch.full_like(pred_idx, float("nan"))
                    k_m = torch.zeros_like(k_m)
                else:
                    err_m = pred_idx - true_idx.to(pred_idx.device).float()
                rows[m] = (pred_idx, err_m, k_m, flips, down_var)
                idx_err_all.append(err_m)
                idx_keep_all.append(k_m)
                flips_all.append(flips)
                down_all.append(down_var)

            idx_err = torch.cat(idx_err_all)
            idx_keep = torch.cat(idx_keep_all)
            flips = torch.cat(flips_all)
            down_var = torch.cat(down_all)

            # Per-Sample-Dump, damit sich das Modell gegen die ungelernten
            # Baselines (fusion.analysis.event_baselines) auf DENSELBEN Patches
            # und mit denselben Statistiken vergleichen laesst. Der Epochen-
            # Mittelwert im Logger reicht dafuer nicht: die Tabelle berichtet
            # Median und Trefferquoten, und beide sind aus einem Mittel nicht
            # rekonstruierbar. Der Join laeuft ueber die patch id.
            if stage == "test":
                meta = batch.get("metadata") or []
                for i in range(pred_day.shape[0]):
                    sid = meta[i].get("id") if i < len(meta) else None
                    row = {
                        "id": sid if sid is not None else f"sample_{len(self._test_event_rows):05d}",
                        "pred_event_day": round(float(pred_day[i]), 3),
                        "true_event_day": ("" if not torch.isfinite(true_day[i])
                                           else round(float(true_day[i]), 3)),
                        "error_days": ("" if not bool(keep[i])
                                       else round(float(pred_day[i] - true_day[i]), 3)),
                        "valid": int(bool(keep[i])),
                    }
                    for m, (pred_idx, err_m, k_m, fl, dv) in rows.items():
                        row[f"pred_idx_{m}"] = int(pred_idx[i])
                        row[f"idx_err_{m}"] = ("" if not bool(k_m[i]) else int(err_m[i]))
                        row[f"flips_{m}"] = int(fl[i])
                        row[f"down_var_{m}"] = round(float(dv[i]), 4)
                        row[f"valid_{m}"] = int(bool(k_m[i]))
                    self._test_event_rows.append(row)

            log = lambda k, v: self.log(f"{stage}/{k}", v, batch_size=bs, on_step=False,
                                        on_epoch=True, sync_dist=False)

            if keep.any():
                signed = pred_day[keep] - true_day[keep]
                err = signed.abs()
                log("event_MAE_days", err.mean())
                log("event_within_7d", (err <= 7).float().mean())
                log("event_within_14d", (err <= 14).float().mean())
                log("event_bias_days", signed.mean())

            if idx_keep.any():
                signed_i = idx_err[idx_keep]
                err_i = signed_i.abs()
                log("event_idx_MAE", err_i.mean())
                log("event_idx_exact", (err_i == 0).float().mean())
                log("event_idx_within1", (err_i <= 1).float().mean())
                log("event_idx_within2", (err_i <= 2).float().mean())
                log("event_idx_bias", signed_i.mean())

            # Ueber ALLE Samples, nicht nur die mit beidseitig beobachtetem
            # Uebergang: ob die Reihe monoton ist, ist eine Eigenschaft der
            # Vorhersage und haengt nicht am Ground Truth.
            log("event_nonmono_frac", (flips > 1).float().mean())
            log("event_down_var", down_var.mean())

        self.log(f"{stage}/event_loss", loss, batch_size=bs,
                 on_step=False, on_epoch=True, sync_dist=False)
        return loss

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

        if self.event_loss_weight > 0:
            ev = self._event_loss_and_stats(out, batch, stage, bs)
            if ev is not None:
                loss = loss + self.event_loss_weight * ev

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

    # ── Per-Sample-Metriken ──────────────────────────────────────────────
    # Die gemeldete Test-mIoU ist ein GEPOOLTER Schaetzer: MulticlassJaccardIndex
    # summiert die Konfusion ueber den ganzen Testsatz und bildet erst danach
    # IoU je Klasse und deren Mittel. Ein Konfidenzintervall dazu (Bootstrap ueber
    # AOIs -- die 441 Flood-Testpatches verteilen sich auf nur 61 AOIs, das ist
    # die effektive Stichprobe) muss denselben Schaetzer nachbauen. Deshalb landen
    # hier je Sample die suffizienten Statistiken und nicht die fertige
    # Per-Sample-IoU: Resample ziehen, Zaehlungen darin aufsummieren, dann die
    # Metrik bilden reproduziert exakt die Tabellenzahl. Ein Mittel ueber
    # Per-Sample-IoUs taete das nicht.
    #
    # Gespeichert wird die volle One-vs-Rest-Konfusion je Klasse -- tp/fp/fn/tn --
    # statt nur Schnitt und Vereinigung. Damit laesst sich jede Metrik der Tabelle
    # nachtraeglich aus dem Dump bauen, ohne den Testlauf zu wiederholen:
    #   IoU_c       = tp / (tp + fp + fn)
    #   Precision_c = tp / (tp + fp)          Recall_c = tp / (tp + fn)
    #   F1_c        = 2tp / (2tp + fp + fn)   Accuracy_c = (tp + tn) / n_valid
    # Es gilt tp + fp + fn + tn == n_valid je Klasse; ignore_index-Pixel sind in
    # keiner der vier Zahlen enthalten.
    #
    # Alles Weitere (AOI, Event, Split, Wolkenanteil) haengt am patch_id und kommt
    # per Join aus flood_patch_meta*.parquet -- hier wird nichts davon dupliziert.

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
                prd_c.flatten(1).sum(1) - tp,   # FP: vorhergesagt, aber nicht GT
                tgt_c.flatten(1).sum(1) - tp,   # FN: GT, aber nicht vorhergesagt
                n_valid - union,                # TN: in keinem von beiden
            )

        for i in range(targets.shape[0]):
            sid = None
            if metadata is not None and i < len(metadata):
                sid = metadata[i].get("id")
            if sid is None and not self._warned_missing_id:
                # Der Dump ist nur brauchbar, wenn er sich per id an die
                # Patch-Metadaten joinen laesst -- ein stiller Ersatzschluessel
                # waere genau die Art Fehler, die erst beim Join auffaellt.
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
            # Bequemlichkeitsspalte fuer schnelle Sichtung und zum Sortieren nach
            # Ausreissern -- NICHT der Schaetzer der Tabelle (s. Kommentar oben).
            # Klassen ohne Pixel in Prediction und GT bleiben aussen vor.
            row["mIoU_sample"] = round(sum(ious) / len(ious), 6) if ious else ""
            self._test_rows.append(row)

    def on_test_epoch_start(self):
        self._test_rows = []
        self._test_event_rows = []
        self._warned_missing_id = False

    def _write_event_dump(self, seg_path: Path):
        rows = getattr(self, "_test_event_rows", None)
        if not rows:
            return
        path = seg_path.with_name(seg_path.name.replace("per_sample", "event_per_sample"))
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        n_valid = sum(r["valid"] for r in rows)
        self.print(f"Saved per-sample event localisation ({n_valid}/{len(rows)} valid) → {path}")
        self._test_event_rows = []

    def on_test_epoch_end(self):
        if not self._test_rows:
            return

        path = self.per_sample_path
        if path is None:
            log_dir = getattr(self.trainer, "log_dir", None)
            path = Path(log_dir or ".") / "test_per_sample.csv"
        path = Path(path)
        # Bei DDP sieht jeder Rank nur seinen Shard des Testsatzes -- eine Datei je
        # Rank statt eines stillen Teildumps. Aktuell laeuft alles mit devices=1,
        # der Zweig ist reine Absicherung.
        if self.trainer.world_size > 1:
            path = path.with_name(f"{path.stem}_rank{self.trainer.global_rank}{path.suffix}")

        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(self._test_rows[0]))
            writer.writeheader()
            writer.writerows(self._test_rows)
        self.print(f"Saved per-sample test metrics ({len(self._test_rows)} samples) → {path}")
        self._test_rows = []
        self._write_event_dump(path)

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