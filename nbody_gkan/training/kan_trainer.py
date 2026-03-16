import torch
import warnings
from tqdm import tqdm
from kan.LBFGS import LBFGS
from .trainer import Trainer


class KANTrainer(Trainer):
    """
    Trainer for GKAN with a two-phase optimizer strategy:
      Phase 1 (warmup): Adam for stable initialization
      Phase 2:          LBFGS for fine-grained convergence

    Extra parameters
    ----------------
    adam_warmup_epochs : int
        Number of epochs to train with Adam before switching to LBFGS.
        Set to 0 to skip warmup and use LBFGS from the start.
    lbfgs_lr : float
        Learning rate for LBFGS (should almost always be 1.0)
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
                 lbfgs_lr: float = 1.0,
                 adam_lr: float = 1e-3,
                 adam_warmup_epochs: int = 0,
                 grid_update_freq: int = 10,
                 grid_update_warmup: int = 5,
                 max_grid_updates: int = 4,
                 scheduler=None,
                 **kwargs):
        super().__init__(model, train_loader, val_loader, **kwargs)

        self.adam_warmup_epochs = adam_warmup_epochs

        # Build both optimizers upfront — we swap self.optimizer between them
        self._adam_optimizer = torch.optim.Adam(
            model.parameters(), lr=adam_lr
        )
        # self._lbfgs_optimizer = LBFGS(          # pykan's LBFGS — handles minibatch
        #     model.parameters(), lr=lbfgs_lr,    # noise and nonsmooth splines better
        #     max_iter=10, history_size=20,
        #     line_search_fn='strong_wolfe',
        # )
        self._lbfgs_optimizer = torch.optim.LBFGS(
            model.parameters(), lr=lbfgs_lr,
            max_iter=10, history_size=20,
            line_search_fn='strong_wolfe',
        )

        # Start with Adam if warmup is requested, otherwise LBFGS immediately
        self.optimizer = self._adam_optimizer if adam_warmup_epochs > 0 else self._lbfgs_optimizer
        self._using_lbfgs = adam_warmup_epochs == 0

        self.scheduler         = scheduler
        self.lamb              = 0.0
        self.grid_update_freq  = grid_update_freq
        self.grid_update_warmup = grid_update_warmup
        self.max_grid_updates  = max_grid_updates
        self._n_grid_updates   = 0


    def _maybe_switch_to_lbfgs(self, epoch: int):
        """Switch from Adam to LBFGS once warmup is complete."""
        if not self._using_lbfgs and epoch >= self.adam_warmup_epochs:
            tqdm.write(
                f"  Epoch {epoch}: switching from Adam to LBFGS "
                f"(warmup complete after {self.adam_warmup_epochs} epochs)"
            )
            self.optimizer    = self._lbfgs_optimizer
            self._using_lbfgs = True

    @property
    def current_phase(self) -> str:
        return "LBFGS" if self._using_lbfgs else "Adam"

    def _on_epoch_start(self, epoch: int):
        # Phase switch check comes first
        self._maybe_switch_to_lbfgs(epoch)

        # Grid updates only during LBFGS phase — noisy Adam gradients
        # make grid updates unreliable during warmup
        if (self._using_lbfgs
                and self.grid_update_freq > 0
                and epoch > self.grid_update_warmup
                and epoch % self.grid_update_freq == 0
                and self._n_grid_updates < self.max_grid_updates
                and hasattr(self.model, 'update_grids')):
            tqdm.write(f"  Updating KAN grids (update #{self._n_grid_updates + 1})...")
            self.model.update_grids(self.train_loader, device=self.device)
            self._n_grid_updates += 1

    def _train_step(self, batch, augment: bool = False,
                    augmentation_scale: float = 3.0) -> float:
        if self._using_lbfgs:
            def closure():
                self.optimizer.zero_grad()
                loss = self.model.loss(batch, augment=augment,
                                       augmentation=augmentation_scale,
                                       lamb=self.lamb)
                loss.backward()
                return loss
            return self.optimizer.step(closure).item()
        else:
            # Standard Adam step
            self.optimizer.zero_grad()
            loss = self.model.loss(batch, augment=augment,
                                   augmentation=augmentation_scale,
                                   lamb=self.lamb)
            loss.backward()
            self.optimizer.step()
            return loss.item()

    def train_epoch(self, augment=False, augmentation_scale=3.0, gradient_clip=None):
        if gradient_clip is not None and self._using_lbfgs:
            warnings.warn("gradient_clip is not supported with LBFGS and will be ignored.")
            gradient_clip = None
        return super().train_epoch(
            augment=augment,
            augmentation_scale=augmentation_scale,
            gradient_clip=gradient_clip,  # passes through for Adam phase
        )