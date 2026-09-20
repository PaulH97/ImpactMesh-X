from typing import Optional
import torch
from lightning.pytorch import Callback, LightningModule, Trainer


class EMA(Callback):
    """Exponential moving average of the network weights.

    Averaging the trajectory of the last few epochs instead of taking whatever
    point SGD happened to land on removes a good chunk of the seed-to-seed
    scatter, which is what the paper tables are bottlenecked on. Cost is one
    extra copy of the weights on the GPU and no extra compute worth measuring.

    Swap window
    -----------
    The EMA weights are swapped into the live module at the start of validation
    and swapped back out at the start of the next training epoch. That window is
    deliberately wider than validation itself: `ModelCheckpoint` writes
    `best.ckpt` (on validation end) and `last.ckpt` (on train-epoch end) inside
    it, so both files hold the EMA weights and `trainer.test(ckpt_path="best")`
    evaluates them. `val/*` metrics are EMA metrics too, so the checkpoint
    monitor selects on the same weights it saves. `train/*` metrics are not --
    training always runs on the raw weights.

    The averaging covers `net` only, never the LightningModule itself: the
    torchmetrics objects live in the module's `state_dict` as well, and blending
    accumulator states across epochs would quietly corrupt every metric.

    Resuming
    --------
    A resumed run restarts from the EMA weights rather than the raw ones, since
    the checkpoint's `state_dict` is what Lightning restores. The EMA buffer
    itself is carried over in the callback state, so the average is not reset.
    """

    def __init__(self, decay: float = 0.999, warmup_updates: int = 10):
        super().__init__()
        if not 0.0 <= decay < 1.0:
            raise ValueError(f"decay must be in [0, 1), got {decay}")
        # decay=0 turns the callback into a no-op, so an EMA/no-EMA ablation is a
        # `callbacks.2.decay=0` override rather than a second callbacks config.
        self.enabled = decay > 0.0
        self.decay = decay
        self.warmup_updates = warmup_updates
        self.shadow: Optional[dict] = None
        self.num_updates = 0
        self._backup: Optional[dict] = None
        self._last_global_step = -1

    # ── helpers ──────────────────────────────────────────────────────────────

    @staticmethod
    def _target(pl_module: LightningModule) -> torch.nn.Module:
        for attr in ("net", "model"):
            net = getattr(pl_module, attr, None)
            if isinstance(net, torch.nn.Module):
                return net
        raise AttributeError(
            f"{type(pl_module).__name__} exposes neither `net` nor `model`; EMA does not "
            f"know which submodule to average. Averaging the LightningModule itself is not "
            f"an option -- it would blend the torchmetrics states too."
        )

    @property
    def _swapped(self) -> bool:
        return self._backup is not None

    @torch.no_grad()
    def _swap_in(self, pl_module: LightningModule) -> None:
        net = self._target(pl_module)
        self._backup = {k: v.detach().clone() for k, v in net.state_dict().items()}
        net.load_state_dict(self.shadow)

    @torch.no_grad()
    def _swap_out(self, pl_module: LightningModule) -> None:
        self._target(pl_module).load_state_dict(self._backup)
        self._backup = None

    # ── averaging ────────────────────────────────────────────────────────────

    def on_train_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        if not self.enabled:
            return
        if self.shadow is None:
            self.shadow = {
                k: v.detach().clone().float() if v.is_floating_point() else v.detach().clone()
                for k, v in self._target(pl_module).state_dict().items()
            }
        else:  # resumed: the buffer came back through load_state_dict on the wrong device
            device = next(self._target(pl_module).parameters()).device
            self.shadow = {k: v.to(device) for k, v in self.shadow.items()}

    @torch.no_grad()
    def on_train_batch_end(self, trainer: Trainer, pl_module: LightningModule, *args) -> None:
        # Keyed on global_step, not on the batch: with gradient accumulation the
        # weights do not move between micro-batches, and accum_steps differs across
        # datasets (1 for Sen12Landslides, 2 for ImpactMesh). Counting batches would
        # give those two a different effective averaging window.
        if not self.enabled or trainer.global_step == self._last_global_step:
            return
        self._last_global_step = trainer.global_step
        self.num_updates += 1

        # Ramp the decay in, otherwise the first epochs stay pinned to the random init.
        decay = min(self.decay, (1 + self.num_updates) / (self.warmup_updates + self.num_updates))

        for k, v in self._target(pl_module).state_dict().items():
            shadow = self.shadow[k]
            if shadow.is_floating_point():
                shadow.mul_(decay).add_(v.detach().to(shadow.dtype), alpha=1.0 - decay)
            else:
                shadow.copy_(v.detach())  # e.g. num_batches_tracked

    # ── swap window ──────────────────────────────────────────────────────────

    def on_validation_epoch_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        if self.shadow is not None and not trainer.sanity_checking and not self._swapped:
            self._swap_in(pl_module)

    def on_train_epoch_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        if self._swapped:
            self._swap_out(pl_module)

    def on_train_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        # Leave the EMA weights in place so an in-memory `trainer.test(...)` with no
        # ckpt_path (fast_dev_run) evaluates the same weights the checkpoints hold.
        if self.shadow is not None and not self._swapped:
            self._swap_in(pl_module)
        self._backup = None

    # ── checkpointing ────────────────────────────────────────────────────────

    def state_dict(self) -> dict:
        return {
            "decay": self.decay,
            "num_updates": self.num_updates,
            "shadow": None if self.shadow is None else {k: v.cpu() for k, v in self.shadow.items()},
        }

    def load_state_dict(self, state_dict: dict) -> None:
        self.decay = state_dict.get("decay", self.decay)
        self.num_updates = state_dict.get("num_updates", 0)
        self.shadow = state_dict.get("shadow", None)
