import torch
import warnings
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm.auto import tqdm
from .trainer import Trainer

try:
    from kan.LBFGS import LBFGS as PyKANLBFGS
except Exception:
    PyKANLBFGS = None


class KANTrainer(Trainer):
    """
    Trainer for GKAN with configurable hybrid optimizer strategies:
      - "two_phase":   Adam warmup -> LBFGS
      - "alternating": Adam exploration <-> LBFGS convergence

    Extra parameters
    ----------------
    optimizer_mode : str
        Hybrid schedule mode: "two_phase" or "alternating".
    adam_warmup_epochs : int
        Number of epochs to train with Adam before switching to LBFGS.
        Set to 0 to skip warmup and use LBFGS from the start.
        Used only when optimizer_mode="two_phase".
    alternating_adam_epochs : int
        Number of Adam epochs to run per exploration window when
        optimizer_mode="alternating".
    lbfgs_rise_tol : float
        Absolute tolerance for rise detection in alternating mode. Switch from
        LBFGS to Adam when current epoch loss > previous LBFGS epoch loss + tol.
    lbfgs_lr : float
        Learning rate for LBFGS (should almost always be 1.0)
    lbfgs_impl : str
        LBFGS backend to use: "torch" or "pykan".
    lbfgs_tolerance_ys : float
        Curvature threshold used by pykan LBFGS when deciding whether to
        update inverse-Hessian history. Ignored for torch LBFGS.
    adam_lr : float
        Learning rate for Adam warmup phase
    grid_update_freq : int
        Update grids every N epochs (0 = disabled)
    grid_update_warmup : int
        Don't update grids before this epoch
    max_grid_updates : int
        Stop updating grids after this many updates
    """

    def __init__(self, model, train_loader, val_loader=None,
                 lbfgs_train_loader=None,
                 optimizer_mode: str = 'two_phase',
                 lbfgs_lr: float = 1.0,
                 lbfgs_max_iter: int = 10,
                 lbfgs_max_eval: int | None = None,
                 lbfgs_line_search_fn: str | None = 'strong_wolfe',
                 lbfgs_impl: str = 'torch',
                 lbfgs_tolerance_ys: float = 1e-32,
                 adam_lr: float = 1e-3,
                 adam_warmup_epochs: int = 0,
                 alternating_adam_epochs: int = 4,
                 lbfgs_rise_tol: float = 0.0,
                 grid_update_freq: int = 10,
                 grid_update_warmup: int = 5,
                 max_grid_updates: int = 4,
                 scheduler=None,
                 **kwargs):
        super().__init__(model, train_loader, val_loader, **kwargs)

        # Allow different loaders/batch sizes for warmup (Adam) vs LBFGS phase.
        self.adam_train_loader = train_loader
        self.lbfgs_train_loader = (
            lbfgs_train_loader if lbfgs_train_loader is not None else train_loader
        )

        self.adam_warmup_epochs = int(adam_warmup_epochs)
        if self.adam_warmup_epochs < 0:
            raise ValueError(
                f"adam_warmup_epochs must be >= 0 (got {self.adam_warmup_epochs})."
            )

        optimizer_mode = str(optimizer_mode).strip().lower()
        if optimizer_mode not in ('two_phase', 'alternating'):
            raise ValueError(
                f"optimizer_mode must be 'two_phase' or 'alternating' "
                f"(got {optimizer_mode!r})."
            )
        self.optimizer_mode = optimizer_mode

        self.alternating_adam_epochs = int(alternating_adam_epochs)
        if self.alternating_adam_epochs <= 0:
            raise ValueError(
                f"alternating_adam_epochs must be > 0 "
                f"(got {self.alternating_adam_epochs})."
            )

        self.lbfgs_rise_tol = float(lbfgs_rise_tol)
        if self.lbfgs_rise_tol < 0:
            raise ValueError(
                f"lbfgs_rise_tol must be >= 0 (got {self.lbfgs_rise_tol})."
            )

        if isinstance(lbfgs_line_search_fn, str):
            stripped = lbfgs_line_search_fn.strip().lower()
            if stripped in ('', 'none', 'null'):
                lbfgs_line_search_fn = None

        lbfgs_impl = str(lbfgs_impl).strip().lower()
        if lbfgs_impl not in ('torch', 'pykan'):
            raise ValueError(
                f"lbfgs_impl must be 'torch' or 'pykan' (got {lbfgs_impl!r})."
            )
        lbfgs_tolerance_ys = float(lbfgs_tolerance_ys)
        if lbfgs_tolerance_ys < 0:
            raise ValueError(
                f"lbfgs_tolerance_ys must be >= 0 (got {lbfgs_tolerance_ys})."
            )

        self._lbfgs_impl = lbfgs_impl

        # Build both optimizers upfront — we swap self.optimizer between them.
        # Adam-phase updates use AdamW by default.
        self._adam_optimizer = torch.optim.AdamW(
            model.parameters(), lr=adam_lr
        )
        if self._lbfgs_impl == 'pykan':
            if PyKANLBFGS is None:
                warnings.warn(
                    "lbfgs_impl='pykan' requested but kan.LBFGS is unavailable; "
                    "falling back to torch.optim.LBFGS."
                )
                self._lbfgs_impl = 'torch'
            else:
                self._lbfgs_optimizer = PyKANLBFGS(
                    model.parameters(),
                    lr=lbfgs_lr,
                    max_iter=lbfgs_max_iter,
                    max_eval=lbfgs_max_eval,
                    history_size=20,
                    line_search_fn=lbfgs_line_search_fn,
                    tolerance_ys=lbfgs_tolerance_ys,
                )

        if self._lbfgs_impl == 'torch':
            self._lbfgs_optimizer = torch.optim.LBFGS(
                model.parameters(),
                lr=lbfgs_lr,
                max_iter=lbfgs_max_iter,
                max_eval=lbfgs_max_eval,
                history_size=20,
                line_search_fn=lbfgs_line_search_fn,
            )

        if self.optimizer_mode == 'alternating':
            self.optimizer = self._adam_optimizer
            self._using_lbfgs = False
            self._adam_epochs_remaining = self.alternating_adam_epochs
        else:
            # Start with Adam if warmup is requested, otherwise LBFGS immediately.
            self.optimizer = (
                self._adam_optimizer if self.adam_warmup_epochs > 0 else self._lbfgs_optimizer
            )
            self._using_lbfgs = self.adam_warmup_epochs == 0
            self._adam_epochs_remaining = self.adam_warmup_epochs

        self._last_lbfgs_epoch_loss: float | None = None

        if self.optimizer_mode == 'alternating':
            if scheduler is None:
                # Keep Adam decay active across alternating cycles.
                self.scheduler = ReduceLROnPlateau(
                    self._adam_optimizer,
                    mode="min",
                    factor=0.5,
                    patience=5,
                    threshold=1e-4,
                    min_lr=1e-5,
                    cooldown=1,
                )
            else:
                self.scheduler = scheduler
        elif scheduler is None and self.adam_warmup_epochs > 0:
            # Use a simple default scheduler during the Adam warmup phase.
            self.scheduler = ReduceLROnPlateau(
                self._adam_optimizer,
                mode="min",
                factor=0.5,
                patience=5,
                threshold=1e-4,
                min_lr=1e-5,
                cooldown=1,
            )
        else:
            self.scheduler = scheduler
        self.lamb              = 0.0
        self.grid_update_freq  = grid_update_freq
        self.grid_update_warmup = grid_update_warmup
        self.max_grid_updates  = max_grid_updates
        self._n_grid_updates   = 0
        self._current_gradient_clip: float | None = None

    @staticmethod
    def _grad_l2_norm(parameters) -> float:
        total_sq = 0.0
        found_grad = False
        for param in parameters:
            if param.grad is None:
                continue
            grad = param.grad.detach()
            total_sq += float(torch.sum(grad * grad).item())
            found_grad = True
        if not found_grad:
            return 0.0
        return total_sq ** 0.5

    @staticmethod
    def _parameter_delta_l2(parameters_before, parameters_after) -> float:
        return sum(
            float(torch.sum((after.detach() - before) ** 2).item())
            for before, after in zip(parameters_before, parameters_after)
        ) ** 0.5

    def _switch_to_lbfgs(self, epoch: int, reason: str):
        if self._using_lbfgs:
            return
        tqdm.write(f"  Epoch {epoch}: switching Adam -> LBFGS ({reason})")
        self.optimizer = self._lbfgs_optimizer
        self._using_lbfgs = True
        # In alternating mode, keep Adam scheduler state for future return phases.
        if self.optimizer_mode != 'alternating':
            self.scheduler = None
        self._last_lbfgs_epoch_loss = None

    def _switch_to_adam(self, epoch: int, reason: str, n_epochs: int):
        if not self._using_lbfgs:
            self._adam_epochs_remaining = n_epochs
            return
        plural = "s" if n_epochs != 1 else ""
        tqdm.write(
            f"  Epoch {epoch}: switching LBFGS -> Adam "
            f"({reason}; explore for {n_epochs} epoch{plural})"
        )
        # Reset AdamW moments on re-entry so stale momentum from previous
        # Adam windows cannot pull parameters away from the new LBFGS basin.
        if hasattr(self._adam_optimizer, "state"):
            self._adam_optimizer.state.clear()
        self.optimizer = self._adam_optimizer
        self._using_lbfgs = False
        self._adam_epochs_remaining = n_epochs
        self._last_lbfgs_epoch_loss = None

    def _maybe_switch_to_lbfgs(self, epoch: int):
        """Switch from Adam to LBFGS once warmup is complete."""
        if self.optimizer_mode != 'two_phase':
            return
        if not self._using_lbfgs and epoch >= self.adam_warmup_epochs:
            self._switch_to_lbfgs(
                epoch,
                reason=f"warmup complete after {self.adam_warmup_epochs} epochs",
            )

    def _should_step_scheduler(self) -> bool:
        if self.optimizer_mode == 'alternating':
            return not self._using_lbfgs
        return True

    @property
    def current_phase(self) -> str:
        return "LBFGS" if self._using_lbfgs else "Adam"

    def _active_train_loader(self):
        return self.lbfgs_train_loader if self._using_lbfgs else self.adam_train_loader

    def _on_epoch_start(self, epoch: int):
        # Phase switch check comes first
        self._maybe_switch_to_lbfgs(epoch)
        # Also gate on Adam warmup so first grid update is not on switch epoch.
        if self.optimizer_mode == 'alternating':
            grid_warmup_epoch = self.grid_update_warmup
        else:
            grid_warmup_epoch = max(self.grid_update_warmup, self.adam_warmup_epochs)

        # Grid updates only during LBFGS phase — noisy Adam gradients
        # make grid updates unreliable during warmup
        if (self._using_lbfgs
                and self.grid_update_freq > 0
                and epoch > grid_warmup_epoch
                and epoch % self.grid_update_freq == 0
                and self._n_grid_updates < self.max_grid_updates
                and hasattr(self.model, 'update_grids')):
            tqdm.write(f"  Updating KAN grids (update #{self._n_grid_updates + 1})...")
            self.model.update_grids(self.lbfgs_train_loader, device=self.device)
            # Reset LBFGS history after grid changes to avoid stale curvature info
            if hasattr(self.optimizer, 'state'):
                self.optimizer.state.clear()
            self._n_grid_updates += 1

    def _on_epoch_end(self, epoch: int, train_loss: float, val_loss: float):
        del val_loss
        if self.optimizer_mode != 'alternating':
            return

        if self._using_lbfgs:
            if self._last_lbfgs_epoch_loss is not None:
                previous_loss = self._last_lbfgs_epoch_loss
                current_loss = float(train_loss)
                if current_loss > previous_loss + self.lbfgs_rise_tol:
                    self._switch_to_adam(
                        epoch,
                        reason=(
                            "LBFGS loss increased "
                            f"({previous_loss:.3e} -> {current_loss:.3e}, "
                            f"tol={self.lbfgs_rise_tol:.1e})"
                        ),
                        n_epochs=self.alternating_adam_epochs,
                    )
                    return

            self._last_lbfgs_epoch_loss = float(train_loss)
            return

        self._adam_epochs_remaining -= 1
        if self._adam_epochs_remaining <= 0:
            self._switch_to_lbfgs(
                epoch,
                reason=(
                    "Adam exploration window complete "
                    f"({self.alternating_adam_epochs} epochs)"
                ),
            )

    def _train_step(self, batch, augment: bool = False,
                    augmentation_scale: float = 3.0,
                    square_loss: bool = False) -> float:
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        if self._using_lbfgs:
            params_before = [p.detach().clone() for p in trainable_params]
            closure_calls = 0
            closure_grad_norm = 0.0

            def closure():
                nonlocal closure_calls, closure_grad_norm
                self.optimizer.zero_grad()
                loss = self.model.loss(batch, augment=augment,
                                       augmentation=augmentation_scale,
                                       lamb=self.lamb,
                                       square=square_loss)
                loss.backward()
                closure_calls += 1
                closure_grad_norm = self._grad_l2_norm(trainable_params)
                return loss
            # LBFGS returns the pre-update closure loss; recompute post-step for logging
            self.optimizer.step(closure)
            param_delta = self._parameter_delta_l2(params_before, trainable_params)

            self._set_step_metrics(
                opt=f"{self._lbfgs_impl}-lbfgs",
                gnorm=closure_grad_norm,
                dparam=param_delta,
                evals=closure_calls,
            )

            with torch.no_grad():
                loss = self.model.loss(
                    batch,
                    augment=augment,
                    augmentation=augmentation_scale,
                    lamb=self.lamb,
                    square=square_loss,
                )
            return float(loss.item())
        else:
            # Standard Adam step
            self.optimizer.zero_grad()
            loss = self.model.loss(batch, augment=augment,
                                   augmentation=augmentation_scale,
                                   lamb=self.lamb,
                                   square=square_loss)
            loss.backward()
            grad_norm = self._grad_l2_norm(trainable_params)
            if self._current_gradient_clip is not None:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self._current_gradient_clip
                )
            self.optimizer.step()
            self._set_step_metrics(opt="adamw", gnorm=grad_norm)
            return loss.item()

    def train_epoch(self, augment=False, augmentation_scale=3.0,
                   gradient_clip=None, square_loss=None):
        # Only honor gradient clipping during Adam warmup; LBFGS does its own line search
        self._current_gradient_clip = None
        if gradient_clip is not None and self._using_lbfgs:
            warnings.warn("gradient_clip is not supported with LBFGS and will be ignored.")
            gradient_clip = None
        else:
            self._current_gradient_clip = gradient_clip

        original_loader = self.train_loader
        self.train_loader = self._active_train_loader()
        try:
            return super().train_epoch(
                augment=augment,
                augmentation_scale=augmentation_scale,
                gradient_clip=gradient_clip,  # passes through for Adam phase
                square_loss=square_loss,
            )
        finally:
            self.train_loader = original_loader