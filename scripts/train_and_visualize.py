"""
Train model, perform forward inference, and visualize learned dynamics.

This script:
1. Trains a model (baseline GNN or Graph-KAN) for a specified number of epochs
2. Performs rollout inference using the learned model
3. Visualizes predicted vs ground truth trajectories
4. Extracts and visualizes learned edge functions (for Graph-KAN)

Usage:
    # Train Graph-KAN and visualize
    python scripts/train_and_visualize.py --model graph_kan --epochs 10

    # Train baseline GNN
    python scripts/train_and_visualize.py --model baseline_gnn --epochs 10

    # Use pre-trained model
    python scripts/train_and_visualize.py --model graph_kan --load_checkpoint checkpoints/graph_kan/best.pt
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from tqdm import tqdm

from nbody_gkan.data.dataset import NBodyDataset, get_edge_index
from nbody_gkan.device import get_device
from nbody_gkan.models import OGN, OrdinaryGraphKAN
from nbody_gkan.nbody import NBodySimulator, gravity
from nbody_gkan.training.trainer import Trainer, create_optimizer, create_scheduler


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train model and visualize learned dynamics"
    )

    # Model selection
    parser.add_argument(
        "--model",
        type=str,
        default="graph_kan",
        choices=["graph_kan", "baseline_gnn"],
        help="Model type",
    )

    # Data
    parser.add_argument(
        "--train_data", type=str, default="data/train.npz", help="Training data path"
    )
    parser.add_argument(
        "--val_data", type=str, default="data/val.npz", help="Validation data path"
    )

    # Model hyperparameters
    parser.add_argument("--msg_dim", type=int, default=100, help="Message dimension")
    parser.add_argument("--hidden", type=int, default=300, help="Hidden layer dimension")
    parser.add_argument(
        "--grid_size", type=int, default=5, help="KAN grid size (Graph-KAN only)"
    )
    parser.add_argument(
        "--spline_order", type=int, default=3, help="KAN spline order (Graph-KAN only)"
    )

    # Training
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--max_lr", type=float, default=5e-3, help="Max LR for OneCycleLR")
    parser.add_argument("--weight_decay", type=float, default=0.0, help="Weight decay")
    parser.add_argument(
        "--scheduler",
        type=str,
        default="onecycle",
        choices=["onecycle", "cosine", "step", "none"],
        help="LR scheduler",
    )
    parser.add_argument("--gradient_clip", type=float, default=1.0, help="Gradient clipping")

    # Checkpoint
    parser.add_argument(
        "--checkpoint_dir", type=str, default="checkpoints", help="Checkpoint directory"
    )
    parser.add_argument(
        "--load_checkpoint", type=str, default=None, help="Load from checkpoint"
    )

    # Rollout inference
    parser.add_argument(
        "--rollout_steps", type=int, default=100, help="Number of rollout steps"
    )
    parser.add_argument(
        "--rollout_dt", type=float, default=0.01, help="Rollout timestep"
    )

    # Visualization
    parser.add_argument(
        "--output_dir", type=str, default="results", help="Output directory for plots"
    )
    parser.add_argument(
        "--n_test_trajectories", type=int, default=3, help="Number of test trajectories"
    )
    parser.add_argument(
        "--live_visualization", action="store_true", help="Show live rollout visualization"
    )
    parser.add_argument(
        "--test_seed", type=int, default=42, help="Seed for selecting test trajectories"
    )

    # Device
    parser.add_argument(
        "--device", type=str, default=None, help="Device (cuda/mps/cpu, None=auto)"
    )

    return parser.parse_args()


def rollout_trajectory(
    model, initial_state, edge_index, n_steps, dt, device, masses, G=1.0, softening=1e-2,
    visualize=False, model_name="Model"
):
    """
    Perform multi-step rollout using the learned model.

    Parameters
    ----------
    model : nn.Module
        Trained model
    initial_state : torch.Tensor
        Initial node features [pos, vel, mass], shape (n_nodes, n_features)
    edge_index : torch.Tensor
        Edge indices
    n_steps : int
        Number of rollout steps
    dt : float
        Timestep
    device : torch.device
        Device
    masses : np.ndarray
        Particle masses
    G : float
        Gravitational constant
    softening : float
        Softening parameter
    visualize : bool, optional
        If True, show live visualization during rollout (2D only)
    model_name : str, optional
        Name for visualization title

    Returns
    -------
    dict
        Predicted and ground truth trajectories
    """
    model.eval()
    n_nodes = initial_state.shape[0]
    ndim = model.ndim

    # Storage for trajectories
    pred_positions = np.zeros((n_steps, n_nodes, ndim))
    pred_velocities = np.zeros((n_steps, n_nodes, ndim))
    true_positions = np.zeros((n_steps, n_nodes, ndim))
    true_velocities = np.zeros((n_steps, n_nodes, ndim))

    # Initial conditions
    state = initial_state.clone().to(device)
    edge_index = edge_index.to(device)  # Ensure edge_index is on correct device
    pos = state[:, :ndim].cpu().numpy()
    vel = state[:, ndim : 2 * ndim].cpu().numpy()

    pred_positions[0] = pos
    pred_velocities[0] = vel
    true_positions[0] = pos
    true_velocities[0] = vel

    # Set up live visualization if requested
    fig, axes, pred_dots, true_dots, pred_trails, true_trails = None, None, None, None, None, None
    if visualize and ndim == 2:
        plt.ion()
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        colors = plt.get_cmap("tab10")(np.linspace(0, 1, n_nodes))

        # Set up both panels
        for ax, title in zip(axes, [f"{model_name} - Predicted", "Ground Truth"]):
            ax.set_facecolor("#080818")
            ax.set_aspect("equal")
            for sp in ax.spines.values():
                sp.set_edgecolor("#1a1a3a")
            ax.tick_params(colors="#555577")

            spread = max(np.abs(pos).max() * 2.5, 2.0)
            ax.set_xlim(-spread, spread)
            ax.set_ylim(-spread, spread)
            ax.set_title(title, color="#aaaacc", fontsize=11)
            ax.grid(alpha=0.2, color="#333355")

        # Initialize plot elements
        pred_trails = [axes[0].plot([], [], color=colors[i], alpha=0.4, lw=1.2)[0] for i in range(n_nodes)]
        pred_dots = [axes[0].plot([], [], 'o', color=colors[i], ms=8,
                                  markeredgecolor='white', markeredgewidth=0.5)[0] for i in range(n_nodes)]

        true_trails = [axes[1].plot([], [], color=colors[i], alpha=0.4, lw=1.2)[0] for i in range(n_nodes)]
        true_dots = [axes[1].plot([], [], 'o', color=colors[i], ms=8,
                                  markeredgecolor='white', markeredgewidth=0.5)[0] for i in range(n_nodes)]

        fig.suptitle(f"Live Rollout - Step 0/{n_steps}", color="#aaaacc", fontsize=13, fontweight="bold")
        plt.tight_layout()
        fig.canvas.draw()
        fig.canvas.flush_events()

    with torch.no_grad():
        for step in tqdm(range(1, n_steps), desc="Rolling out", disable=visualize):
            # Predict acceleration using learned model
            data = Data(x=state, edge_index=edge_index)
            pred_acc = model.just_derivative(data, augment=False).cpu().numpy()

            # Ground truth acceleration
            true_acc = gravity(pos, masses, G, softening)

            # Update predicted state (Euler integration for simplicity)
            pred_vel_new = vel + pred_acc * dt
            pred_pos_new = pos + pred_vel_new * dt

            # Update true state
            true_vel_new = vel + true_acc * dt
            true_pos_new = pos + true_vel_new * dt

            # Store
            pred_positions[step] = pred_pos_new
            pred_velocities[step] = pred_vel_new
            true_positions[step] = true_pos_new
            true_velocities[step] = true_vel_new

            # Update visualization every few steps
            if visualize and ndim == 2 and step % 2 == 0:
                # Update predicted trajectories
                for i in range(n_nodes):
                    pred_trails[i].set_data(pred_positions[:step+1, i, 0], pred_positions[:step+1, i, 1])
                    pred_dots[i].set_data([pred_pos_new[i, 0]], [pred_pos_new[i, 1]])

                    true_trails[i].set_data(true_positions[:step+1, i, 0], true_positions[:step+1, i, 1])
                    true_dots[i].set_data([true_pos_new[i, 0]], [true_pos_new[i, 1]])

                fig.suptitle(f"Live Rollout - Step {step}/{n_steps}", color="#aaaacc", fontsize=13, fontweight="bold")
                fig.canvas.draw()
                fig.canvas.flush_events()

            # Update state for next step (use predicted state)
            pos = pred_pos_new
            vel = pred_vel_new
            state[:, :ndim] = torch.from_numpy(pos).float()
            state[:, ndim : 2 * ndim] = torch.from_numpy(vel).float()

    if visualize and fig is not None:
        plt.ioff()
        plt.show()

    return {
        "pred_positions": pred_positions,
        "pred_velocities": pred_velocities,
        "true_positions": true_positions,
        "true_velocities": true_velocities,
    }


def visualize_trajectory(trajectory_dict, title, save_path):
    """
    Visualize predicted vs true trajectories.

    Parameters
    ----------
    trajectory_dict : dict
        Output from rollout_trajectory
    title : str
        Plot title
    save_path : Path
        Where to save the plot
    """
    pred_pos = trajectory_dict["pred_positions"]
    true_pos = trajectory_dict["true_positions"]

    n_steps, n_nodes, ndim = pred_pos.shape

    if ndim != 2:
        print(f"Skipping visualization for dim={ndim} (only 2D supported)")
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    colors = plt.get_cmap("tab10")(np.linspace(0, 1, n_nodes))

    # Plot predicted trajectories
    ax = axes[0]
    for i in range(n_nodes):
        ax.plot(
            pred_pos[:, i, 0],
            pred_pos[:, i, 1],
            color=colors[i],
            alpha=0.7,
            label=f"Body {i}",
        )
        # Mark start and end
        ax.scatter(pred_pos[0, i, 0], pred_pos[0, i, 1], color=colors[i], s=100, marker="o", zorder=5)
        ax.scatter(pred_pos[-1, i, 0], pred_pos[-1, i, 1], color=colors[i], s=150, marker="*", zorder=5)

    ax.set_aspect("equal")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("Predicted Trajectories")
    ax.legend()
    ax.grid(alpha=0.3)

    # Plot ground truth trajectories
    ax = axes[1]
    for i in range(n_nodes):
        ax.plot(
            true_pos[:, i, 0],
            true_pos[:, i, 1],
            color=colors[i],
            alpha=0.7,
            label=f"Body {i}",
        )
        # Mark start and end
        ax.scatter(true_pos[0, i, 0], true_pos[0, i, 1], color=colors[i], s=100, marker="o", zorder=5)
        ax.scatter(true_pos[-1, i, 0], true_pos[-1, i, 1], color=colors[i], s=150, marker="*", zorder=5)

    ax.set_aspect("equal")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("Ground Truth Trajectories")
    ax.legend()
    ax.grid(alpha=0.3)

    plt.suptitle(title, fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"Saved trajectory plot: {save_path}")
    plt.close()


def compute_rollout_errors(trajectory_dict):
    """
    Compute position and velocity errors over rollout.

    Returns
    -------
    dict
        Position and velocity RMSE at each timestep
    """
    pred_pos = trajectory_dict["pred_positions"]
    true_pos = trajectory_dict["true_positions"]
    pred_vel = trajectory_dict["pred_velocities"]
    true_vel = trajectory_dict["true_velocities"]

    # RMSE at each timestep (averaged over all bodies)
    pos_errors = np.sqrt(np.mean((pred_pos - true_pos) ** 2, axis=(1, 2)))
    vel_errors = np.sqrt(np.mean((pred_vel - true_vel) ** 2, axis=(1, 2)))

    return {"position_rmse": pos_errors, "velocity_rmse": vel_errors}


def plot_rollout_errors(errors_list, labels, save_path):
    """Plot rollout errors over time."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for errors, label in zip(errors_list, labels):
        pos_rmse = errors["position_rmse"]
        vel_rmse = errors["velocity_rmse"]
        timesteps = np.arange(len(pos_rmse))

        axes[0].plot(timesteps, pos_rmse, label=label, linewidth=2)
        axes[1].plot(timesteps, vel_rmse, label=label, linewidth=2)

    axes[0].set_xlabel("Timestep")
    axes[0].set_ylabel("Position RMSE")
    axes[0].set_title("Position Error Over Rollout")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].set_xlabel("Timestep")
    axes[1].set_ylabel("Velocity RMSE")
    axes[1].set_title("Velocity Error Over Rollout")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"Saved error plot: {save_path}")
    plt.close()


def extract_kan_edge_functions(model, edge_index, n_samples=1000, device="cpu"):
    """
    Extract learned edge functions from Graph-KAN model.

    For each edge, we sample inputs and record the KAN layer outputs
    to visualize what function the network has learned.

    Parameters
    ----------
    model : OrdinaryGraphKAN
        Trained Graph-KAN model
    edge_index : torch.Tensor
        Edge indices
    n_samples : int
        Number of samples for function visualization
    device : str or torch.device
        Device

    Returns
    -------
    dict
        Sampled edge features and corresponding message outputs
    """
    model.eval()
    n_f = model.msg_layers[0].in_dim // 2  # Input features per node
    n_edges = edge_index.shape[1]

    # Sample random node features in a reasonable range
    # For N-body: positions ~[-2, 2], velocities ~[-1, 1], mass ~[0.5, 1.5]
    samples_i = torch.randn(n_samples, n_f, device=device)
    samples_j = torch.randn(n_samples, n_f, device=device)

    # Normalize to reasonable ranges (heuristic for N-body)
    # Assume features are [pos_x, pos_y, vel_x, vel_y, mass]
    if n_f == 5:  # 2D case
        samples_i[:, :2] = samples_i[:, :2] * 2.0  # positions
        samples_i[:, 2:4] = samples_i[:, 2:4] * 1.0  # velocities
        samples_i[:, 4:5] = torch.abs(samples_i[:, 4:5]) * 0.5 + 0.5  # mass [0.5, 1.5]

        samples_j[:, :2] = samples_j[:, :2] * 2.0
        samples_j[:, 2:4] = samples_j[:, 2:4] * 1.0
        samples_j[:, 4:5] = torch.abs(samples_j[:, 4:5]) * 0.5 + 0.5

    with torch.no_grad():
        # Concatenate to form edge inputs
        edge_inputs = torch.cat([samples_i, samples_j], dim=1)  # (n_samples, 2*n_f)

        # Pass through message function
        messages = model._forward_kan_layers(edge_inputs, model.msg_layers)

    return {
        "edge_inputs": edge_inputs.cpu().numpy(),
        "messages": messages.cpu().numpy(),
        "samples_i": samples_i.cpu().numpy(),
        "samples_j": samples_j.cpu().numpy(),
    }


def visualize_kan_edge_functions(kan_data, save_dir):
    """
    Visualize learned KAN edge functions.

    Creates plots showing:
    1. Message output vs. relative distance
    2. Message output vs. relative velocity
    3. 2D scatter of message magnitudes

    Parameters
    ----------
    kan_data : dict
        Output from extract_kan_edge_functions
    save_dir : Path
        Directory to save plots
    """
    samples_i = kan_data["samples_i"]
    samples_j = kan_data["samples_j"]
    messages = kan_data["messages"]

    # Compute relative features (for 2D case with [pos_x, pos_y, vel_x, vel_y, mass])
    if samples_i.shape[1] == 5:
        # Relative position
        rel_pos = samples_j[:, :2] - samples_i[:, :2]
        rel_dist = np.linalg.norm(rel_pos, axis=1)

        # Relative velocity
        rel_vel = samples_j[:, 2:4] - samples_i[:, 2:4]
        rel_vel_mag = np.linalg.norm(rel_vel, axis=1)

        # Mass product
        mass_product = samples_i[:, 4] * samples_j[:, 4]

        # Message magnitude
        msg_mag = np.linalg.norm(messages, axis=1)

        # Create plots
        fig, axes = plt.subplots(2, 2, figsize=(14, 12))

        # Plot 1: Message magnitude vs relative distance
        ax = axes[0, 0]
        scatter = ax.scatter(rel_dist, msg_mag, c=mass_product, alpha=0.5, s=10, cmap="viridis")
        ax.set_xlabel("Relative Distance |r_j - r_i|")
        ax.set_ylabel("Message Magnitude")
        ax.set_title("Learned Edge Function: Message vs Distance")
        ax.grid(alpha=0.3)
        plt.colorbar(scatter, ax=ax, label="Mass Product m_i * m_j")

        # Expected gravitational force magnitude: G * m_i * m_j / r^2
        # Sort by distance for smooth curve
        sort_idx = np.argsort(rel_dist)
        dist_sorted = rel_dist[sort_idx]
        # Theoretical (normalized)
        G = 1.0
        softening = 1e-2
        theoretical = G * mass_product[sort_idx] / (dist_sorted**2 + softening**2)
        # Plot theoretical (scaled to match message range for comparison)
        scale = np.median(msg_mag) / np.median(theoretical)
        ax.plot(dist_sorted, theoretical * scale, "r-", alpha=0.3, linewidth=2, label="Theoretical (scaled)")
        ax.legend()

        # Plot 2: Message magnitude vs mass product
        ax = axes[0, 1]
        scatter = ax.scatter(mass_product, msg_mag, c=rel_dist, alpha=0.5, s=10, cmap="plasma")
        ax.set_xlabel("Mass Product m_i * m_j")
        ax.set_ylabel("Message Magnitude")
        ax.set_title("Message vs Mass Product")
        ax.grid(alpha=0.3)
        plt.colorbar(scatter, ax=ax, label="Distance")

        # Plot 3: Message components (first 2 dimensions)
        if messages.shape[1] >= 2:
            ax = axes[1, 0]
            scatter = ax.scatter(messages[:, 0], messages[:, 1], c=rel_dist, alpha=0.5, s=10, cmap="coolwarm")
            ax.set_xlabel("Message Component 0")
            ax.set_ylabel("Message Component 1")
            ax.set_title("Message Space (first 2 components)")
            ax.grid(alpha=0.3)
            plt.colorbar(scatter, ax=ax, label="Distance")

        # Plot 4: Relative position components colored by message magnitude
        ax = axes[1, 1]
        scatter = ax.scatter(rel_pos[:, 0], rel_pos[:, 1], c=msg_mag, alpha=0.5, s=10, cmap="hot")
        ax.set_xlabel("Relative Position x")
        ax.set_ylabel("Relative Position y")
        ax.set_title("Message Magnitude in Position Space")
        ax.set_aspect("equal")
        ax.grid(alpha=0.3)
        plt.colorbar(scatter, ax=ax, label="Message Magnitude")

        plt.suptitle("Learned KAN Edge Functions", fontsize=14, fontweight="bold")
        plt.tight_layout()
        save_path = save_dir / "kan_edge_functions.png"
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved KAN edge function plot: {save_path}")
        plt.close()


def main():
    """Main function."""
    args = parse_args()

    # Device
    if args.device:
        device = torch.device(args.device)
    else:
        device = get_device()
    print(f"Using device: {device}")

    # Create output directory
    output_dir = Path(args.output_dir) / args.model
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

    print(f"Dataset: {n} bodies in {dim}D, {len(train_dataset)} training samples")

    # Create model
    if args.model == "graph_kan":
        print(f"\nCreating Graph-KAN model (4 layers, {args.hidden} hidden)")
        model = OrdinaryGraphKAN(
            n_f=n_features,
            msg_dim=args.msg_dim,
            ndim=dim,
            edge_index=edge_index,
            hidden=args.hidden,
            grid_size=args.grid_size,
            spline_order=args.spline_order,
        )
    else:
        print(f"\nCreating baseline GNN model (4 layers, {args.hidden} hidden)")
        model = OGN(
            n_f=n_features,
            msg_dim=args.msg_dim,
            ndim=dim,
            edge_index=edge_index,
            hidden=args.hidden,
        )

    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Load checkpoint or train
    if args.load_checkpoint:
        print(f"\nLoading checkpoint: {args.load_checkpoint}")
        checkpoint = torch.load(args.load_checkpoint, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(device)
    else:
        print(f"\nTraining for {args.epochs} epochs...")

        optimizer = create_optimizer(model, learning_rate=args.lr, weight_decay=args.weight_decay)
        scheduler = create_scheduler(
            optimizer,
            scheduler_type=args.scheduler,
            n_epochs=args.epochs,
            steps_per_epoch=len(train_loader),
            max_lr=args.max_lr,
        )

        checkpoint_dir = Path(args.checkpoint_dir) / args.model
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
            n_epochs=args.epochs,
            augment=True,
            augmentation_scale=3.0,
            gradient_clip=args.gradient_clip,
            save_every=max(1, args.epochs // 5),
            log_every=1,
        )

        model = trainer.model

    # Perform rollout inference on test trajectories
    print(f"\nPerforming rollout inference...")
    print(f"Rollout: {args.rollout_steps} steps with dt={args.rollout_dt}")
    if args.live_visualization:
        print(f"Live visualization: ENABLED")

    # Get a few test samples (with fixed seed for reproducibility)
    np.random.seed(args.test_seed)
    test_indices = np.random.choice(len(val_dataset), size=args.n_test_trajectories, replace=False)
    print(f"Test trajectory indices: {test_indices} (seed={args.test_seed})")

    all_errors = []
    for i, test_idx in enumerate(test_indices):
        print(f"\nTest trajectory {i + 1}/{args.n_test_trajectories}")

        sample = val_dataset[test_idx]
        initial_state = sample.x

        trajectory_dict = rollout_trajectory(
            model=model,
            initial_state=initial_state,
            edge_index=edge_index,
            n_steps=args.rollout_steps,
            dt=args.rollout_dt,
            device=device,
            masses=train_dataset.masses,
            G=train_dataset.G,
            softening=train_dataset.softening,
            visualize=args.live_visualization,
            model_name=args.model.upper(),
        )

        # Compute errors
        errors = compute_rollout_errors(trajectory_dict)
        all_errors.append(errors)

        # Visualize trajectory
        title = f"{args.model.upper()} - Test Trajectory {i + 1}"
        save_path = output_dir / f"trajectory_{i + 1}.png"
        visualize_trajectory(trajectory_dict, title, save_path)

        # Print final errors
        final_pos_error = errors["position_rmse"][-1]
        final_vel_error = errors["velocity_rmse"][-1]
        print(f"  Final position RMSE: {final_pos_error:.6f}")
        print(f"  Final velocity RMSE: {final_vel_error:.6f}")

    # Plot average errors
    print(f"\nPlotting average rollout errors...")
    save_path = output_dir / "rollout_errors.png"
    plot_rollout_errors(
        all_errors,
        labels=[f"Traj {i + 1}" for i in range(args.n_test_trajectories)],
        save_path=save_path,
    )

    # Extract and visualize KAN edge functions (Graph-KAN only)
    if args.model == "graph_kan":
        print(f"\nExtracting learned KAN edge functions...")
        kan_data = extract_kan_edge_functions(
            model=model, edge_index=edge_index, n_samples=5000, device=device
        )

        print(f"Visualizing KAN edge functions...")
        visualize_kan_edge_functions(kan_data, output_dir)

        # Save extracted data
        save_path = output_dir / "kan_edge_data.npz"
        np.savez(save_path, **kan_data)
        print(f"Saved KAN edge data: {save_path}")

    print(f"\n{'=' * 60}")
    print(f"Analysis complete!")
    print(f"{'=' * 60}")
    print(f"Output directory: {output_dir}")
    print(f"- Trajectory plots: trajectory_*.png")
    print(f"- Error plots: rollout_errors.png")
    if args.model == "graph_kan":
        print(f"- KAN edge functions: kan_edge_functions.png")
        print(f"- KAN edge data: kan_edge_data.npz")


if __name__ == "__main__":
    main()
