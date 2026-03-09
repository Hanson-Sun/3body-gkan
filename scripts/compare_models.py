"""
Compare trained models on the SAME test trajectories.

This script loads two trained models (baseline and Graph-KAN) and evaluates
them on identical test cases for fair comparison.

Usage:
    # Compare two trained models
    python scripts/compare_models.py \
        --baseline_checkpoint checkpoints/baseline_gnn/best.pt \
        --kan_checkpoint checkpoints/graph_kan/best.pt

    # Or train first then compare
    python scripts/compare_models.py --train --epochs 10
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch_geometric.loader import DataLoader

from nbody_gkan.data.dataset import NBodyDataset
from nbody_gkan.device import get_device
from nbody_gkan.models import OGN, OrdinaryGraphKAN
from nbody_gkan.training.trainer import Trainer, create_optimizer, create_scheduler

# Import from train_and_visualize
import sys
sys.path.append(str(Path(__file__).parent))
from train_and_visualize import (
    rollout_trajectory,
    visualize_trajectory,
    compute_rollout_errors,
    plot_rollout_errors,
    extract_kan_edge_functions,
    visualize_kan_edge_functions,
)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Compare baseline and Graph-KAN models")

    # Model checkpoints
    parser.add_argument(
        "--baseline_checkpoint",
        type=str,
        default=None,
        help="Baseline GNN checkpoint path",
    )
    parser.add_argument(
        "--kan_checkpoint", type=str, default=None, help="Graph-KAN checkpoint path"
    )

    # Or train models
    parser.add_argument(
        "--train", action="store_true", help="Train models before comparison"
    )
    parser.add_argument("--epochs", type=int, default=10, help="Training epochs (if --train)")

    # Data
    parser.add_argument(
        "--train_data", type=str, default="data/train.npz", help="Training data path"
    )
    parser.add_argument(
        "--val_data", type=str, default="data/val.npz", help="Validation data path"
    )

    # Model hyperparameters
    parser.add_argument("--msg_dim", type=int, default=100, help="Message dimension")
    parser.add_argument("--hidden", type=int, default=300, help="Hidden dimension")
    parser.add_argument("--grid_size", type=int, default=5, help="KAN grid size")
    parser.add_argument("--spline_order", type=int, default=3, help="KAN spline order")

    # Training (if --train)
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")

    # Rollout
    parser.add_argument("--rollout_steps", type=int, default=100, help="Rollout steps")
    parser.add_argument("--rollout_dt", type=float, default=0.01, help="Rollout dt")
    parser.add_argument(
        "--test_seed", type=int, default=42, help="Seed for test trajectory selection"
    )
    parser.add_argument(
        "--n_test_trajectories", type=int, default=3, help="Number of test trajectories"
    )
    parser.add_argument(
        "--live_visualization", action="store_true", help="Show live visualization"
    )

    # Output
    parser.add_argument(
        "--output_dir", type=str, default="results_comparison", help="Output directory"
    )
    parser.add_argument(
        "--checkpoint_dir", type=str, default="checkpoints", help="Checkpoint directory"
    )

    # Device
    parser.add_argument("--device", type=str, default=None, help="Device")

    return parser.parse_args()


def train_model(model, train_loader, val_loader, optimizer, scheduler, device, epochs, checkpoint_dir):
    """Train a model."""
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        checkpoint_dir=checkpoint_dir,
    )

    trainer.train(
        n_epochs=epochs,
        augment=True,
        augmentation_scale=3.0,
        gradient_clip=1.0,
        save_every=max(1, epochs // 5),
        log_every=1,
    )

    return trainer.model


def main():
    """Main function."""
    args = parse_args()

    # Device
    device = torch.device(args.device) if args.device else get_device()
    print(f"Using device: {device}")

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load datasets
    print(f"\nLoading data...")
    train_dataset = NBodyDataset(args.train_data)
    val_dataset = NBodyDataset(args.val_data)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    n = train_dataset.n
    dim = train_dataset.dim
    n_features = 2 * dim + 1
    edge_index = train_dataset.edge_index

    print(f"Dataset: {n} bodies in {dim}D")
    print(f"Training samples: {len(train_dataset)}, Validation samples: {len(val_dataset)}")

    # Get test trajectories (ONCE, with fixed seed)
    print(f"\nSelecting test trajectories (seed={args.test_seed})...")
    np.random.seed(args.test_seed)
    test_indices = np.random.choice(
        len(val_dataset), size=args.n_test_trajectories, replace=False
    )
    print(f"Test indices: {test_indices}")

    # Save test indices for reproducibility
    np.save(output_dir / "test_indices.npy", test_indices)

    # Create or load models
    print(f"\n{'='*60}")
    print(f"Setting up models...")
    print(f"{'='*60}")

    # Baseline GNN
    baseline_model = OGN(
        n_f=n_features,
        msg_dim=args.msg_dim,
        ndim=dim,
        edge_index=edge_index,
        hidden=args.hidden,
    )

    if args.train or args.baseline_checkpoint is None:
        print(f"\nTraining baseline GNN...")
        baseline_optimizer = create_optimizer(baseline_model, learning_rate=args.lr)
        baseline_scheduler = create_scheduler(
            baseline_optimizer,
            scheduler_type="onecycle",
            n_epochs=args.epochs,
            steps_per_epoch=len(train_loader),
            max_lr=5e-3,
        )
        baseline_checkpoint_dir = Path(args.checkpoint_dir) / "baseline_gnn_comparison"
        baseline_model = train_model(
            baseline_model,
            train_loader,
            val_loader,
            baseline_optimizer,
            baseline_scheduler,
            device,
            args.epochs,
            baseline_checkpoint_dir,
        )
        args.baseline_checkpoint = str(baseline_checkpoint_dir / "best.pt")
    else:
        print(f"\nLoading baseline GNN from {args.baseline_checkpoint}")
        checkpoint = torch.load(args.baseline_checkpoint, map_location=device)
        baseline_model.load_state_dict(checkpoint["model_state_dict"])

    baseline_model.to(device)
    baseline_model.eval()

    # Graph-KAN
    kan_model = OrdinaryGraphKAN(
        n_f=n_features,
        msg_dim=args.msg_dim,
        ndim=dim,
        edge_index=edge_index,
        hidden=args.hidden,
        grid_size=args.grid_size,
        spline_order=args.spline_order,
    )

    if args.train or args.kan_checkpoint is None:
        print(f"\nTraining Graph-KAN...")
        kan_optimizer = create_optimizer(kan_model, learning_rate=args.lr)
        kan_scheduler = create_scheduler(
            kan_optimizer,
            scheduler_type="onecycle",
            n_epochs=args.epochs,
            steps_per_epoch=len(train_loader),
            max_lr=5e-3,
        )
        kan_checkpoint_dir = Path(args.checkpoint_dir) / "graph_kan_comparison"
        kan_model = train_model(
            kan_model,
            train_loader,
            val_loader,
            kan_optimizer,
            kan_scheduler,
            device,
            args.epochs,
            kan_checkpoint_dir,
        )
        args.kan_checkpoint = str(kan_checkpoint_dir / "best.pt")
    else:
        print(f"\nLoading Graph-KAN from {args.kan_checkpoint}")
        checkpoint = torch.load(args.kan_checkpoint, map_location=device)
        kan_model.load_state_dict(checkpoint["model_state_dict"])

    kan_model.to(device)
    kan_model.eval()

    print(f"\nBaseline parameters: {sum(p.numel() for p in baseline_model.parameters()):,}")
    print(f"Graph-KAN parameters: {sum(p.numel() for p in kan_model.parameters()):,}")

    # Evaluate both models on the SAME test trajectories
    print(f"\n{'='*60}")
    print(f"Evaluating both models on identical test trajectories")
    print(f"{'='*60}")

    all_baseline_errors = []
    all_kan_errors = []

    for i, test_idx in enumerate(test_indices):
        print(f"\n[Test trajectory {i+1}/{len(test_indices)}] Index: {test_idx}")

        sample = val_dataset[test_idx]
        initial_state = sample.x

        # Baseline rollout
        print(f"  Baseline GNN rollout...")
        baseline_trajectory = rollout_trajectory(
            model=baseline_model,
            initial_state=initial_state,
            edge_index=edge_index,
            n_steps=args.rollout_steps,
            dt=args.rollout_dt,
            device=device,
            masses=train_dataset.masses,
            G=train_dataset.G,
            softening=train_dataset.softening,
            visualize=args.live_visualization,
            model_name="Baseline GNN",
        )

        baseline_errors = compute_rollout_errors(baseline_trajectory)
        all_baseline_errors.append(baseline_errors)

        # Graph-KAN rollout (SAME initial state!)
        print(f"  Graph-KAN rollout...")
        kan_trajectory = rollout_trajectory(
            model=kan_model,
            initial_state=initial_state,
            edge_index=edge_index,
            n_steps=args.rollout_steps,
            dt=args.rollout_dt,
            device=device,
            masses=train_dataset.masses,
            G=train_dataset.G,
            softening=train_dataset.softening,
            visualize=args.live_visualization,
            model_name="Graph-KAN",
        )

        kan_errors = compute_rollout_errors(kan_trajectory)
        all_kan_errors.append(kan_errors)

        # Visualize both trajectories
        baseline_save_path = output_dir / f"baseline_trajectory_{i+1}.png"
        visualize_trajectory(
            baseline_trajectory, f"Baseline GNN - Test {i+1}", baseline_save_path
        )

        kan_save_path = output_dir / f"kan_trajectory_{i+1}.png"
        visualize_trajectory(kan_trajectory, f"Graph-KAN - Test {i+1}", kan_save_path)

        # Print comparison
        baseline_final_pos_err = baseline_errors["position_rmse"][-1]
        kan_final_pos_err = kan_errors["position_rmse"][-1]
        improvement = (baseline_final_pos_err - kan_final_pos_err) / baseline_final_pos_err * 100

        print(f"  Baseline final position RMSE: {baseline_final_pos_err:.6f}")
        print(f"  Graph-KAN final position RMSE: {kan_final_pos_err:.6f}")
        print(f"  Improvement: {improvement:+.1f}%")

    # Plot comparison
    print(f"\n{'='*60}")
    print(f"Creating comparison plots...")
    print(f"{'='*60}")

    # Combined error plot
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    for i, (baseline_err, kan_err) in enumerate(zip(all_baseline_errors, all_kan_errors)):
        timesteps = np.arange(len(baseline_err["position_rmse"]))

        # Position RMSE
        axes[0, 0].plot(
            timesteps, baseline_err["position_rmse"], "r-", alpha=0.7, label=f"Baseline {i+1}"
        )
        axes[0, 0].plot(
            timesteps, kan_err["position_rmse"], "b-", alpha=0.7, label=f"Graph-KAN {i+1}"
        )

        # Velocity RMSE
        axes[0, 1].plot(
            timesteps, baseline_err["velocity_rmse"], "r-", alpha=0.7, label=f"Baseline {i+1}"
        )
        axes[0, 1].plot(
            timesteps, kan_err["velocity_rmse"], "b-", alpha=0.7, label=f"Graph-KAN {i+1}"
        )

    axes[0, 0].set_xlabel("Timestep")
    axes[0, 0].set_ylabel("Position RMSE")
    axes[0, 0].set_title("Position Error Comparison")
    axes[0, 0].legend()
    axes[0, 0].grid(alpha=0.3)

    axes[0, 1].set_xlabel("Timestep")
    axes[0, 1].set_ylabel("Velocity RMSE")
    axes[0, 1].set_title("Velocity Error Comparison")
    axes[0, 1].legend()
    axes[0, 1].grid(alpha=0.3)

    # Average errors
    avg_baseline_pos = np.mean([e["position_rmse"] for e in all_baseline_errors], axis=0)
    avg_kan_pos = np.mean([e["position_rmse"] for e in all_kan_errors], axis=0)
    avg_baseline_vel = np.mean([e["velocity_rmse"] for e in all_baseline_errors], axis=0)
    avg_kan_vel = np.mean([e["velocity_rmse"] for e in all_kan_errors], axis=0)

    timesteps = np.arange(len(avg_baseline_pos))
    axes[1, 0].plot(timesteps, avg_baseline_pos, "r-", linewidth=2, label="Baseline (avg)")
    axes[1, 0].plot(timesteps, avg_kan_pos, "b-", linewidth=2, label="Graph-KAN (avg)")
    axes[1, 0].set_xlabel("Timestep")
    axes[1, 0].set_ylabel("Position RMSE")
    axes[1, 0].set_title("Average Position Error")
    axes[1, 0].legend()
    axes[1, 0].grid(alpha=0.3)

    axes[1, 1].plot(timesteps, avg_baseline_vel, "r-", linewidth=2, label="Baseline (avg)")
    axes[1, 1].plot(timesteps, avg_kan_vel, "b-", linewidth=2, label="Graph-KAN (avg)")
    axes[1, 1].set_xlabel("Timestep")
    axes[1, 1].set_ylabel("Velocity RMSE")
    axes[1, 1].set_title("Average Velocity Error")
    axes[1, 1].legend()
    axes[1, 1].grid(alpha=0.3)

    plt.suptitle("Model Comparison: Baseline GNN vs Graph-KAN", fontsize=14, fontweight="bold")
    plt.tight_layout()

    comparison_path = output_dir / "model_comparison.png"
    plt.savefig(comparison_path, dpi=150, bbox_inches="tight")
    print(f"Saved comparison plot: {comparison_path}")
    plt.close()

    # Extract KAN edge functions
    print(f"\nExtracting KAN edge functions...")
    kan_data = extract_kan_edge_functions(
        model=kan_model, edge_index=edge_index, n_samples=5000, device=device
    )
    visualize_kan_edge_functions(kan_data, output_dir)

    # Summary
    print(f"\n{'='*60}")
    print(f"Comparison Complete!")
    print(f"{'='*60}")
    print(f"\nOutput directory: {output_dir}")
    print(f"  - model_comparison.png: Side-by-side error comparison")
    print(f"  - baseline_trajectory_*.png: Baseline predictions")
    print(f"  - kan_trajectory_*.png: Graph-KAN predictions")
    print(f"  - kan_edge_functions.png: Learned KAN functions")
    print(f"\nFinal position RMSE (averaged):")
    print(f"  Baseline: {avg_baseline_pos[-1]:.6f}")
    print(f"  Graph-KAN: {avg_kan_pos[-1]:.6f}")
    improvement = (avg_baseline_pos[-1] - avg_kan_pos[-1]) / avg_baseline_pos[-1] * 100
    print(f"  Improvement: {improvement:+.1f}%")


if __name__ == "__main__":
    main()
