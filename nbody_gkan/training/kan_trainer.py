import torch
import warnings
from collections.abc import Sequence
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
from .trainer import Trainer

try:
    from kan.LBFGS import LBFGS as PyKANLBFGS
except Exception:
    PyKANLBFGS = None


class KANTrainer(Trainer):
    """
    Trainer for GKAN with a two-phase optimizer strategy:
      Phase 1 (warmup): AdamW for stable initialization
      Phase 2:          LBFGS for fine-grained convergence

    Extra parameters
    ----------------
    adam_warmup_epochs : int
        Number of epochs to train with Adam before switching to LBFGS.
        Set to 0 to skip warmup and use LBFGS from the start.
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
                 lbfgs_lr: float = 1.0,
                 lbfgs_max_iter: int = 10,
                 lbfgs_max_eval: int | None = None,
                 lbfgs_line_search_fn: str | None = 'strong_wolfe',
                 lbfgs_impl: str = 'torch',
                 lbfgs_tolerance_ys: float = 1e-32,
                 adam_lr: float = 1e-3,
                 adam_warmup_epochs: int = 0,
                 grid_update_freq: int = 10,
                 grid_update_warmup: int = 5,
                 max_grid_updates: int = 4,
                 lamb_schedule: list[float] | None = None,
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

        # Start with Adam if warmup is requested, otherwise LBFGS immediately.
        self.optimizer = (
            self._adam_optimizer if self.adam_warmup_epochs > 0 else self._lbfgs_optimizer
        )
        self._using_lbfgs = self.adam_warmup_epochs == 0

        if scheduler is None and self.adam_warmup_epochs > 0:
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
        self._lamb_schedule = self._coerce_lamb_schedule(lamb_schedule)
        self._scheduled_total_epochs: int | None = None
        self._sparsity_stage_idx: int | None = None

    @staticmethod
    def _coerce_lamb_schedule(
        lamb_schedule: list[float] | Sequence[float] | None,
    ) -> tuple[float, ...] | None:
        if lamb_schedule is None:
            return None
        if isinstance(lamb_schedule, (str, bytes)):
            raise ValueError(
                "lamb_schedule must be a sequence of floats, not a string."
            )
        if not isinstance(lamb_schedule, Sequence):
            raise ValueError("lamb_schedule must be a sequence of floats.")

        parsed: list[float] = []
        for i, value in enumerate(lamb_schedule):
            try:
                weight = float(value)
            except Exception as exc:
                raise ValueError(
                    f"lamb_schedule[{i}] must be numeric, got {value!r}."
                ) from exc
            if weight < 0.0:
                raise ValueError(
                    f"lamb_schedule[{i}] must be >= 0, got {weight}."
                )
            parsed.append(weight)

        if not parsed:
            return None
        return tuple(parsed)

    def _apply_lamb_schedule(self, epoch: int):
        if not self._lamb_schedule:
            return
        if self._scheduled_total_epochs is None or self._scheduled_total_epochs <= 0:
            return

        n_levels = len(self._lamb_schedule)
        stage_idx = min(
            (epoch * n_levels) // self._scheduled_total_epochs,
            n_levels - 1,
        )
        self.lamb = float(self._lamb_schedule[stage_idx])

        if self._sparsity_stage_idx != stage_idx:
            self._sparsity_stage_idx = stage_idx
            tqdm.write(
                f"  Epoch {epoch}: lamb schedule -> {self.lamb:.2e} "
                f"({stage_idx + 1}/{n_levels})"
            )

    def train(
        self,
        n_epochs: int,
        augment: bool = False,
        augmentation_scale: float = 3.0,
        gradient_clip: float | None = None,
        square_loss: bool = False,
        save_every: int = 10,
        log_every: int = 1,
    ):
        self._scheduled_total_epochs = int(n_epochs)
        self._sparsity_stage_idx = None
        try:
            return super().train(
                n_epochs=n_epochs,
                augment=augment,
                augmentation_scale=augmentation_scale,
                gradient_clip=gradient_clip,
                square_loss=square_loss,
                save_every=save_every,
                log_every=log_every,
            )
        finally:
            self._scheduled_total_epochs = None
            self._sparsity_stage_idx = None

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
        self.scheduler = None

    def _maybe_switch_to_lbfgs(self, epoch: int):
        """Switch from Adam to LBFGS once warmup is complete."""
        if not self._using_lbfgs and epoch >= self.adam_warmup_epochs:
            self._switch_to_lbfgs(
                epoch,
                reason=f"warmup complete after {self.adam_warmup_epochs} epochs",
            )

    @property
    def current_phase(self) -> str:
        return "LBFGS" if self._using_lbfgs else "Adam"

    def _active_train_loader(self):
        return self.lbfgs_train_loader if self._using_lbfgs else self.adam_train_loader

    def _on_epoch_start(self, epoch: int):
        self._apply_lamb_schedule(epoch)

        # Phase switch check comes first
        self._maybe_switch_to_lbfgs(epoch)
        # Also gate on Adam warmup so first grid update is not on switch epoch.
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