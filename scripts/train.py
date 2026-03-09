"""
Main training script for Graph-KAN and baseline models.

Usage:
    python scripts/train.py --model graph_kan --epochs 100
    python scripts/train.py --model baseline_gnn --data data/train.npz
"""

import argparse
from pathlib import Path

import torch
from torch_geometric.loader import DataLoader

from nbody_gkan.data.dataset import NBodyDataset
from nbody_gkan.device import get_device
from nbody_gkan.models import OrdinaryGraphKAN, OGN
from nbody_gkan.training.trainer import Trainer, create_optimizer, create_scheduler


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train Graph-KAN or baseline GNN")

    # Model selection
    parser.add_argument(
        "--model",
        type=str,
        default="graph_kan",
        choices=["graph_kan", "baseline_gnn"],
        help="Model type to train",
    )

    # Data
    parser.add_argument(
        "--train_data", type=str, default="data/train.npz", help="Training data path"
    )
    parser.add_argument(
        "--val_data", type=str, default="data/val.npz", help="Validation data path"
    )

    # Model hyperparameters
    parser.add_argument(
        "--msg_dim", type=int, default=100, help="Message dimension"
    )
    parser.add_argument(
        "--hidden", type=int, default=300, help="Hidden layer dimension"
    )
    parser.add_argument(
        "--grid_size", type=int, default=5, help="KAN grid size (Graph-KAN only)"
    )
    parser.add_argument(
        "--spline_order", type=int, default=3, help="KAN spline order (Graph-KAN only)"
    )

    # Training
    parser.add_argument(
        "--batch_size", type=int, default=32, help="Batch size"
    )
    parser.add_argument(
        "--epochs", type=int, default=100, help="Number of epochs"
    )
    parser.add_argument(
        "--lr", type=float, default=1e-3, help="Learning rate"
    )
    parser.add_argument(
        "--max_lr", type=float, default=5e-3, help="Max LR for OneCycleLR"
    )
    parser.add_argument(
        "--weight_decay", type=float, default=0.0, help="Weight decay"
    )
    parser.add_argument(
        "--scheduler",
        type=str,
        default="onecycle",
        choices=["onecycle", "cosine", "step", "none"],
        help="LR scheduler",
    )
    parser.add_argument(
        "--augment",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use position augmentation (use --no-augment to disable)"
    )
    parser.add_argument(
        "--augmentation_scale", type=float, default=3.0, help="Augmentation scale"
    )
    parser.add_argument(
        "--gradient_clip", type=float, default=1.0, help="Gradient clipping"
    )
    parser.add_argument(
        "--grid_update_freq", type=int, default=10,
        help="Update KAN grids every N epochs (0=disabled, Graph-KAN only)"
    )

    # Checkpointing
    parser.add_argument(
        "--checkpoint_dir", type=str, default="checkpoints", help="Checkpoint directory"
    )
    parser.add_argument(
        "--save_every", type=int, default=10, help="Save checkpoint every N epochs"
    )
    parser.add_argument(
        "--log_every", type=int, default=1, help="Log every N epochs"
    )

    # Device
    parser.add_argument(
        "--device", type=str, default=None, help="Device (cuda/mps/cpu, None=auto)"
    )

    return parser.parse_args()


def main():
    """Main training function."""
    args = parse_args()

    # Device
    if args.device:
        device = torch.device(args.device)
    else:
        device = get_device()
    print(f"Using device: {device}")

    # Load datasets
    print(f"Loading training data from {args.train_data}")
    train_dataset = NBodyDataset(args.train_data)

    print(f"Loading validation data from {args.val_data}")
    val_dataset = NBodyDataset(args.val_data)

    print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")

    # Data loaders
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0
    )

    # Model parameters
    n = train_dataset.n  # Number of bodies
    dim = train_dataset.dim  # Spatial dimension
    n_features = 2 * dim + 1  # [pos, vel, mass]

    edge_index = train_dataset.edge_index

    # Create model
    # Both models use configurable architecture: 4 layers, default 300 hidden (matches original OGN)
    if args.model == "graph_kan":
        print(f"Creating Graph-KAN model (4 layers, {args.hidden} hidden)")
        model = OrdinaryGraphKAN(
            n_f=n_features,
            msg_dim=args.msg_dim,
            ndim=dim,
            edge_index=edge_index,
            hidden=args.hidden,
            grid_size=args.grid_size,
            spline_order=args.spline_order,
            aggr="add",
        )
    elif args.model == "baseline_gnn":
        print(f"Creating baseline GNN model (4 layers, {args.hidden} hidden)")
        model = OGN(
            n_f=n_features,
            msg_dim=args.msg_dim,
            ndim=dim,
            edge_index=edge_index,
            hidden=args.hidden,
            aggr="add",
        )
    else:
        raise ValueError(f"Unknown model type: {args.model}")

    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Optimizer
    optimizer = create_optimizer(
        model, learning_rate=args.lr, weight_decay=args.weight_decay
    )

    # Scheduler
    scheduler = create_scheduler(
        optimizer,
        scheduler_type=args.scheduler,
        n_epochs=args.epochs,
        steps_per_epoch=len(train_loader),
        max_lr=args.max_lr,
    )

    # Checkpoint directory
    checkpoint_dir = Path(args.checkpoint_dir) / args.model
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        checkpoint_dir=checkpoint_dir,
    )

    # Grid update info (Graph-KAN only)
    if args.model == "graph_kan":
        if args.grid_update_freq > 0:
            print(f"Grid updates enabled every {args.grid_update_freq} epochs")
        else:
            print("Grid updates disabled (use --grid_update_freq N to enable)")

    # Train
    trainer.train(
        n_epochs=args.epochs,
        augment=args.augment,
        augmentation_scale=args.augmentation_scale,
        gradient_clip=args.gradient_clip,
        save_every=args.save_every,
        log_every=args.log_every,
        grid_update_freq=args.grid_update_freq,
    )

    # Save history
    trainer.save_history()

    print(f"\nTraining complete! Checkpoints saved to {checkpoint_dir}")


if __name__ == "__main__":
    main()
