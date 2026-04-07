import torch
from torch.optim import Adam
from tqdm import tqdm
from .trainer import Trainer

class GNNTrainer(Trainer):
    """Trainer for standard GNN with Adam optimizer."""

    def __init__(self, model, train_loader, val_loader=None,
                 lr: float = 1e-3, weight_decay: float = 0.0,
                 gradient_clip: float | None = None,
                 scheduler: torch.optim.lr_scheduler._LRScheduler | None = None,
                 **kwargs):
        super().__init__(model, train_loader, val_loader, **kwargs)
        self.optimizer     = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        self.scheduler     = scheduler
        self.lamb          = 0.0
        self.gradient_clip = gradient_clip

    def _train_step(self, batch, augment: bool = False, augmentation_scale: float = 3.0,
                    square_loss: bool = False) -> float:
        self.optimizer.zero_grad()
        loss = self.model.loss(batch, augment=augment, augmentation=augmentation_scale,
                               lamb=self.lamb, square=square_loss)
        loss.backward()
        if self.gradient_clip is not None:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip)
        self.optimizer.step()
        return loss.item()