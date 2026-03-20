from typing import Optional

import torch
from torch_geometric.loader import DataLoader
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tqdm import tqdm

class Trainer:
    def __init__(
            self,
            model: torch.nn.Module,
            train_loader: DataLoader,
            val_loader: DataLoader | None = None,
            device: torch.device | None = None,
            checkpoint_dir: Path | str | None = None,
    ):
        self.model        = model
        self.train_loader = train_loader
        self.val_loader   = val_loader
        self.device       = device or torch.device("cpu")
        self.model.to(self.device)

        self.checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir else Path("checkpoints")
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.history = {"train": [], "val": [], "lr": []}
        self.epoch         = 0
        self.best_val_loss = float("inf")

        # subclasses must set these
        self.optimizer: torch.optim.Optimizer
        self.scheduler: torch.optim.lr_scheduler._LRScheduler | None = None

    def _train_step(self, batch, augment: bool = False,
                    augmentation_scale: float = 3.0) -> float:
        """
        Single optimizer step. Override in subclasses that need
        closure-based optimizers (e.g. LBFGS).
        """
        raise NotImplementedError

    def _on_epoch_start(self, epoch: int):
        """Hook called at the start of each epoch. Override for custom logic."""
        pass

    def train_epoch(
            self,
            augment: bool = True,
            augmentation_scale: float = 3.0,
            gradient_clip: float | None = None,
    ) -> float:
        self.model.train()
        total_loss = 0.0
        n_samples  = 0

        pbar = tqdm(self.train_loader, desc=f"Epoch {self.epoch}", leave=False)
        for i, batch in enumerate(pbar):
            batch      = batch.to(self.device)
            loss_value = self._train_step(batch, augment=augment, augmentation_scale=augmentation_scale)

            if self.scheduler is not None and isinstance(self.scheduler, OneCycleLR):
                self.scheduler.step()

            batch_size  = batch.num_graphs if hasattr(batch, "num_graphs") else 1
            total_loss += loss_value * batch_size
            n_samples  += batch_size

            if i % 10 == 0:
                pbar.set_postfix(loss=f"{loss_value:.4e}")

        return total_loss / n_samples

    @torch.no_grad()
    def validate(self) -> float:
        if self.val_loader is None:
            return 0.0
        self.model.eval()
        total_loss = 0.0
        n_samples  = 0
        for batch in self.val_loader:
            batch       = batch.to(self.device)
            loss        = self.model.loss(batch, augment=False)
            batch_size  = batch.num_graphs if hasattr(batch, "num_graphs") else 1
            total_loss += loss.item() * batch_size
            n_samples  += batch_size
        return total_loss / n_samples

    def train(
            self,
            n_epochs: int,
            augment: bool = False,
            augmentation_scale: float = 3.0,
            gradient_clip: float | None = None,
            save_every: int = 10,
            log_every: int = 1,
    ):
        print(f"Training {self.model.__class__.__name__} "
              f"| {sum(p.numel() for p in self.model.parameters()):,} params "
              f"| {self.device}")

        for epoch in range(n_epochs):
            self.epoch = epoch
            self._on_epoch_start(epoch)

            train_loss = self.train_epoch(
                augment=augment,
                augmentation_scale=augmentation_scale,
                gradient_clip=gradient_clip,
            )
            val_loss   = self.validate()

            if self.scheduler is not None and not isinstance(
                self.scheduler, OneCycleLR
            ):
                self.scheduler.step()

            current_lr = self.optimizer.param_groups[0]["lr"]
            self.history["train"].append(train_loss)
            self.history["val"].append(val_loss)
            self.history["lr"].append(current_lr)

            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.save_checkpoint("best.pt")

            if epoch % save_every == 0 or epoch == n_epochs - 1:
                self.save_checkpoint(f"epoch_{epoch:03d}.pt")

            if epoch % log_every == 0:
                tqdm.write(
                    f"Epoch {epoch:3d} | "
                    f"Train={train_loss:.4e} | "
                    f"Val={val_loss:.4e} | "
                    f"LR={current_lr:.2e}"
                )

        print(f"Done. Best val loss: {self.best_val_loss:.6f}")

    def save_checkpoint(self, filename: str):
        if self.checkpoint_dir is None:
            return
            
        torch.save({
            "epoch":               self.epoch,
            "model_state_dict":    self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict() if self.scheduler else None,
            "history":             self.history,
            "best_val_loss":       self.best_val_loss,
        }, self.checkpoint_dir / filename)

    def load_checkpoint(self, checkpoint_path: str | Path):
        ckpt = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        if self.scheduler and ckpt["scheduler_state_dict"]:
            self.scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        self.epoch         = ckpt["epoch"]
        self.history       = ckpt["history"]
        self.best_val_loss = ckpt["best_val_loss"]
        print(f"Loaded checkpoint from epoch {self.epoch}")

    def save_history(self, filename: str = "history.json"):
        with open(self.checkpoint_dir / filename, "w") as f:
            json.dump(self.history, f, indent=2)


