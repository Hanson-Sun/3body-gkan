"""
Training loop with checkpointing and logging.
"""

import json
from pathlib import Path
from typing import Optional

import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import OneCycleLR, CosineAnnealingLR, StepLR
from torch.utils.data import DataLoader
from tqdm import tqdm


class Trainer:
    """
    Trainer for Graph-KAN and baseline GNN models.

    Handles:
    - Training loop with progress bars
    - Learning rate scheduling
    - Position augmentation
    - Checkpointing
    - Metric logging

    Parameters
    ----------
    model : nn.Module
        The model to train (must have a .loss() method)
    train_loader : DataLoader
        Training data loader
    val_loader : DataLoader, optional
        Validation data loader
    optimizer : torch.optim.Optimizer
        Optimizer (required)
    scheduler : torch.optim.lr_scheduler, optional
        Learning rate scheduler
    device : torch.device, optional
        Device to train on
    checkpoint_dir : Path or str, optional
        Directory for saving checkpoints
    """

    def __init__(
            self,
            model: torch.nn.Module,
            train_loader: DataLoader,
            val_loader: Optional[DataLoader] = None,
            optimizer: torch.optim.Optimizer = None,
            scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
            device: Optional[torch.device] = None,
            checkpoint_dir: Optional[Path | str] = None,
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device or torch.device("cpu")
        self.model.to(self.device)

        if optimizer is None:
            raise ValueError("optimizer is required. Pass an optimizer instance to Trainer.")
        self.optimizer = optimizer
        self.scheduler = scheduler

        self.checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir else Path("checkpoints")
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.history = {
            "train_loss": [],
            "val_loss": [],
            "lr": [],
        }

        self.epoch = 0
        self.best_val_loss = float("inf")

    def train_epoch(
            self,
            augment: bool = True,
            augmentation_scale: float = 3.0,
            gradient_clip: Optional[float] = None,
    ) -> float:
        """
        Train for one epoch.

        Parameters
        ----------
        augment : bool, optional (default=True)
            Apply position augmentation
        augmentation_scale : float, optional (default=3.0)
            Scale of augmentation noise
        gradient_clip : float, optional
            Gradient clipping value (None to disable)

        Returns
        -------
        float
            Average training loss
        """
        self.model.train()
        total_loss = 0.0
        n_samples = 0

        pbar = tqdm(self.train_loader, desc=f"Epoch {self.epoch}", leave=False)

        for batch in pbar:
            batch = batch.to(self.device)

            self.optimizer.zero_grad()

            # Compute loss using the model's loss method
            loss = self.model.loss(
                batch, augment=augment, augmentation=augmentation_scale
            )

            loss.backward()

            # Gradient clipping
            if gradient_clip is not None:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), gradient_clip)

            self.optimizer.step()

            # Step scheduler if it needs per-batch stepping (OneCycleLR)
            if self.scheduler is not None and isinstance(
                    self.scheduler, OneCycleLR
            ):
                self.scheduler.step()

            # Accumulate loss
            batch_size = batch.num_graphs if hasattr(batch, "num_graphs") else 1
            total_loss += loss.item() * batch_size
            n_samples += batch_size

            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        avg_loss = total_loss / n_samples
        return avg_loss

    @torch.no_grad()
    def validate(self) -> float:
        """
        Validate the model.

        Returns
        -------
        float
            Average validation loss
        """
        if self.val_loader is None:
            return 0.0

        self.model.eval()
        total_loss = 0.0
        n_samples = 0

        for batch in self.val_loader:
            batch = batch.to(self.device)

            # No augmentation during validation
            loss = self.model.loss(batch, augment=False)

            batch_size = batch.num_graphs if hasattr(batch, "num_graphs") else 1
            total_loss += loss.item() * batch_size
            n_samples += batch_size

        avg_loss = total_loss / n_samples
        return avg_loss

    def train(
            self,
            n_epochs: int,
            augment: bool = True,
            augmentation_scale: float = 3.0,
            gradient_clip: Optional[float] = None,
            save_every: int = 10,
            log_every: int = 1,
    ):
        """
        Full training loop.

        Parameters
        ----------
        n_epochs : int
            Number of epochs to train
        augment : bool, optional (default=True)
            Apply position augmentation
        augmentation_scale : float, optional (default=3.0)
            Scale of augmentation noise
        gradient_clip : float, optional
            Gradient clipping value
        save_every : int, optional (default=10)
            Save checkpoint every k epochs
        log_every : int, optional (default=1)
            Log metrics every k epochs
        """
        print(f"Training for {n_epochs} epochs on {self.device}")
        print(f"Model: {self.model.__class__.__name__}")
        print(f"Parameters: {sum(p.numel() for p in self.model.parameters()):,}")

        for epoch in range(n_epochs):
            self.epoch = epoch

            # Train
            train_loss = self.train_epoch(
                augment=augment,
                augmentation_scale=augmentation_scale,
                gradient_clip=gradient_clip,
            )

            # Validate
            val_loss = self.validate()

            # Update scheduler (only for epoch-based schedulers, not OneCycleLR)
            if self.scheduler is not None and not isinstance(
                    self.scheduler, OneCycleLR
            ):
                self.scheduler.step()

            # Log
            current_lr = self.optimizer.param_groups[0]["lr"]
            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_loss)
            self.history["lr"].append(current_lr)

            if epoch % log_every == 0:
                print(
                    f"Epoch {epoch:3d} | "
                    f"Train Loss: {train_loss:.6f} | "
                    f"Val Loss: {val_loss:.6f} | "
                    f"LR: {current_lr:.6f}"
                )

            # Save checkpoint
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.save_checkpoint("best.pt")

            if epoch % save_every == 0 or epoch == n_epochs - 1:
                self.save_checkpoint(f"epoch_{epoch:03d}.pt")

        print(f"\nTraining complete. Best validation loss: {self.best_val_loss:.6f}")

    def save_checkpoint(self, filename: str):
        """
        Save model checkpoint.

        Parameters
        ----------
        filename : str
            Checkpoint filename
        """
        checkpoint_path = self.checkpoint_dir / filename
        torch.save(
            {
                "epoch": self.epoch,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "scheduler_state_dict": (
                    self.scheduler.state_dict() if self.scheduler else None
                ),
                "history": self.history,
                "best_val_loss": self.best_val_loss,
            },
            checkpoint_path,
        )
        print(f"Checkpoint saved: {checkpoint_path}")

    def load_checkpoint(self, checkpoint_path: str | Path):
        """
        Load model checkpoint.

        Parameters
        ----------
        checkpoint_path : str or Path
            Path to checkpoint file
        """
        checkpoint_path = Path(checkpoint_path)
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        if self.scheduler and checkpoint["scheduler_state_dict"]:
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        self.epoch = checkpoint["epoch"]
        self.history = checkpoint["history"]
        self.best_val_loss = checkpoint["best_val_loss"]

        print(f"Loaded checkpoint from epoch {self.epoch}")

    def save_history(self, filename: str = "history.json"):
        """Save training history to JSON."""
        history_path = self.checkpoint_dir / filename
        with open(history_path, "w") as f:
            json.dump(self.history, f, indent=2)
        print(f"Training history saved: {history_path}")


def create_optimizer(
        model: torch.nn.Module,
        learning_rate: float = 1e-3,
        weight_decay: float = 0.0,
) -> torch.optim.Optimizer:
    """
    Create Adam optimizer.

    Parameters
    ----------
    model : nn.Module
        Model to optimize
    learning_rate : float, optional (default=1e-3)
        Learning rate
    weight_decay : float, optional (default=0.0)
        L2 regularization weight

    Returns
    -------
    torch.optim.Optimizer
        Adam optimizer
    """
    return Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)


def create_scheduler(
        optimizer: torch.optim.Optimizer,
        scheduler_type: str,
        n_epochs: int,
        steps_per_epoch: int,
        max_lr: float = 5e-3,
        **kwargs,
) -> Optional[torch.optim.lr_scheduler._LRScheduler]:
    """
    Create learning rate scheduler.

    Parameters
    ----------
    optimizer : torch.optim.Optimizer
        Optimizer
    scheduler_type : str
        Scheduler type: 'onecycle', 'cosine', 'step', or 'none'
    n_epochs : int
        Total number of epochs
    steps_per_epoch : int
        Number of optimization steps per epoch
    max_lr : float, optional (default=5e-3)
        Maximum learning rate for OneCycleLR
    **kwargs
        Additional scheduler arguments

    Returns
    -------
    torch.optim.lr_scheduler._LRScheduler or None
        Scheduler instance or None
    """
    if scheduler_type == "none":
        return None

    elif scheduler_type == "onecycle":
        return OneCycleLR(
            optimizer,
            max_lr=max_lr,
            epochs=n_epochs,
            steps_per_epoch=steps_per_epoch,
            **kwargs,
        )

    elif scheduler_type == "cosine":
        return CosineAnnealingLR(optimizer, T_max=n_epochs, **kwargs)

    elif scheduler_type == "step":
        step_size = kwargs.get("step_size", n_epochs // 3)
        gamma = kwargs.get("gamma", 0.1)
        return StepLR(optimizer, step_size=step_size, gamma=gamma)

    else:
        raise ValueError(
            f"Unknown scheduler type: {scheduler_type}. "
            f"Choose from: 'onecycle', 'cosine', 'step', 'none'"
        )
