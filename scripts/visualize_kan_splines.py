"""
Visualize KAN B-splines per input feature.

This script provides detailed visualization of what each input feature
contributes to the learned functions. For N-body problems, this shows
how each feature (pos_x, pos_y, vel_x, vel_y, mass) affects the output.

Usage:
    python scripts/visualize_kan_splines.py --checkpoint checkpoints/graph_kan/best.pt
    python scripts/visualize_kan_splines.py --checkpoint PATH --output_dir spline_viz
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from nbody_gkan.data.dataset import NBodyDataset
from nbody_gkan.device import get_device
from nbody_gkan.models import OrdinaryGraphKAN


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Visualize KAN B-splines per input feature"
    )

    parser.add_argument(
        "--checkpoint", type=str, required=True, help="Graph-KAN checkpoint path"
    )
    parser.add_argument(
        "--train_data", type=str, default="data/train.npz", help="Training data path"
    )
    parser.add_argument(
        "--output_dir", type=str, default="spline_viz", help="Output directory"
    )
    parser.add_argument("--device", type=str, default=None, help="Device")

    # Model hyperparameters (must match checkpoint)
    parser.add_argument("--msg_dim", type=int, default=100, help="Message dimension")
    parser.add_argument("--hidden", type=int, default=300, help="Hidden dimension")
    parser.add_argument("--grid_size", type=int, default=5, help="KAN grid size")
    parser.add_argument("--spline_order", type=int, default=3, help="KAN spline order")

    # Visualization
    parser.add_argument(
        "--x_range",
        type=float,
        nargs=2,
        default=[-3, 3],
        help="Input range for plotting (default: -3 3)",
    )
    parser.add_argument(
        "--n_points", type=int, default=300, help="Number of points to plot (default: 300)"
    )

    return parser.parse_args()


def plot_layer_splines_per_feature(
    layer, layer_name, feature_names, save_dir, x_range=(-3, 3), n_points=300, device='cpu'
):
    """
    Plot B-spline functions for each input feature.

    For each input feature, show how it maps to each output dimension
    when all other inputs are held at zero.

    Parameters
    ----------
    layer : KANLayer
        The KAN layer to visualize
    layer_name : str
        Name for the layer (e.g., "Message Layer 1")
    feature_names : list of str
        Names of input features
    save_dir : Path
        Directory to save plots
    x_range : tuple
        (min, max) range for x-axis
    n_points : int
        Number of points to evaluate
    device : str or torch.device
        Device to use for computation
    """
    in_dim = layer.in_dim
    out_dim = layer.out_dim

    print(f"\n{layer_name}: {in_dim} inputs → {out_dim} outputs")

    # Create input range
    x_vals = np.linspace(x_range[0], x_range[1], n_points)

    # For each input feature
    for in_idx in range(in_dim):
        feature_name = feature_names[in_idx] if in_idx < len(feature_names) else f"Input {in_idx}"

        print(f"  Visualizing feature: {feature_name}")

        # Prepare input: zeros except for this feature
        x_tensor = torch.zeros(n_points, in_dim, device=device)
        x_tensor[:, in_idx] = torch.tensor(x_vals, dtype=torch.float32, device=device)

        # Evaluate layer
        with torch.no_grad():
            y, _, _, _ = layer(x_tensor)
            y = y.cpu().numpy()  # (n_points, out_dim)

        # Plot: one subplot for each output dimension
        n_cols = min(4, out_dim)
        n_rows = (out_dim + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3 * n_rows))

        if n_rows == 1 and n_cols == 1:
            axes = np.array([[axes]])
        elif n_rows == 1:
            axes = axes.reshape(1, -1)
        elif n_cols == 1:
            axes = axes.reshape(-1, 1)

        for out_idx in range(out_dim):
            row = out_idx // n_cols
            col = out_idx % n_cols
            ax = axes[row, col]

            # Plot the spline
            ax.plot(x_vals, y[:, out_idx], linewidth=2, color="C0")
            ax.set_xlabel(feature_name)
            ax.set_ylabel(f"Output {out_idx}")
            ax.set_title(f"f({feature_name} → out{out_idx})")
            ax.grid(alpha=0.3)
            ax.axhline(0, color="k", linewidth=0.5, alpha=0.3)
            ax.axvline(0, color="k", linewidth=0.5, alpha=0.3)

            # Add statistics
            y_mean = y[:, out_idx].mean()
            y_std = y[:, out_idx].std()
            y_range = y[:, out_idx].max() - y[:, out_idx].min()
            ax.text(
                0.05,
                0.95,
                f"μ={y_mean:.3f}\nσ={y_std:.3f}\nΔ={y_range:.3f}",
                transform=ax.transAxes,
                verticalalignment="top",
                fontsize=8,
                bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
            )

        # Remove empty subplots
        for idx in range(out_dim, n_rows * n_cols):
            row = idx // n_cols
            col = idx % n_cols
            fig.delaxes(axes[row, col])

        plt.suptitle(
            f"{layer_name} - Feature: {feature_name}", fontsize=14, fontweight="bold"
        )
        plt.tight_layout()

        # Save
        safe_name = layer_name.lower().replace(" ", "_")
        safe_feature = feature_name.lower().replace(" ", "_").replace(",", "")
        save_path = save_dir / f"{safe_name}_feature_{safe_feature}.png"
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"    Saved: {save_path}")
        plt.close()


def plot_all_features_summary(
    layer, layer_name, feature_names, save_dir, x_range=(-3, 3), n_points=300, device='cpu'
):
    """
    Create summary plot showing all input features for the first output dimension.

    This gives a quick overview of how each feature contributes.

    Parameters
    ----------
    device : str or torch.device
        Device to use for computation
    """
    in_dim = layer.in_dim
    out_dim = layer.out_dim

    x_vals = np.linspace(x_range[0], x_range[1], n_points)

    # Create subplots: one for each input feature
    n_cols = min(3, in_dim)
    n_rows = (in_dim + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))

    if n_rows == 1 and n_cols == 1:
        axes = np.array([[axes]])
    elif n_rows == 1:
        axes = axes.reshape(1, -1)
    elif n_cols == 1:
        axes = axes.reshape(-1, 1)

    for in_idx in range(in_dim):
        feature_name = feature_names[in_idx] if in_idx < len(feature_names) else f"Input {in_idx}"

        # Prepare input
        x_tensor = torch.zeros(n_points, in_dim, device=device)
        x_tensor[:, in_idx] = torch.tensor(x_vals, dtype=torch.float32, device=device)

        # Evaluate
        with torch.no_grad():
            y, _, _, _ = layer(x_tensor)
            y = y.cpu().numpy()

        row = in_idx // n_cols
        col = in_idx % n_cols
        ax = axes[row, col]

        # Plot first few output dimensions
        for out_idx in range(min(5, out_dim)):
            ax.plot(x_vals, y[:, out_idx], linewidth=2, alpha=0.7, label=f"Out {out_idx}")

        ax.set_xlabel(feature_name)
        ax.set_ylabel("Output")
        ax.set_title(f"{feature_name}")
        ax.grid(alpha=0.3)
        ax.axhline(0, color="k", linewidth=0.5, alpha=0.3)
        ax.axvline(0, color="k", linewidth=0.5, alpha=0.3)
        if out_dim <= 5:
            ax.legend(fontsize=8)

    # Remove empty subplots
    for idx in range(in_dim, n_rows * n_cols):
        row = idx // n_cols
        col = idx % n_cols
        fig.delaxes(axes[row, col])

    plt.suptitle(
        f"{layer_name} - All Features (first {min(5, out_dim)} outputs)",
        fontsize=14,
        fontweight="bold",
    )
    plt.tight_layout()

    safe_name = layer_name.lower().replace(" ", "_")
    save_path = save_dir / f"{safe_name}_all_features_summary.png"
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"  Saved summary: {save_path}")
    plt.close()


def main():
    """Main function."""
    args = parse_args()

    # Device
    device = torch.device(args.device) if args.device else get_device()
    print(f"Using device: {device}")

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load dataset (just to get dimensions)
    print(f"\nLoading dataset: {args.train_data}")
    dataset = NBodyDataset(args.train_data)
    n = dataset.n
    dim = dataset.dim
    n_features = 2 * dim + 1
    edge_index = dataset.edge_index

    # Feature names for N-body (2D case)
    if dim == 2:
        feature_names_node = ["pos_x", "pos_y", "vel_x", "vel_y", "mass"]
        feature_names_edge = [
            "pos_i_x", "pos_i_y", "vel_i_x", "vel_i_y", "mass_i",
            "pos_j_x", "pos_j_y", "vel_j_x", "vel_j_y", "mass_j",
        ]
    elif dim == 3:
        feature_names_node = ["pos_x", "pos_y", "pos_z", "vel_x", "vel_y", "vel_z", "mass"]
        feature_names_edge = [
            "pos_i_x", "pos_i_y", "pos_i_z", "vel_i_x", "vel_i_y", "vel_i_z", "mass_i",
            "pos_j_x", "pos_j_y", "pos_j_z", "vel_j_x", "vel_j_y", "vel_j_z", "mass_j",
        ]
    else:
        feature_names_node = [f"feature_{i}" for i in range(n_features)]
        feature_names_edge = [f"feature_{i}" for i in range(2 * n_features)]

    print(f"Dataset: {n} bodies in {dim}D, feature dimension = {n_features}")

    # Load checkpoint first to infer dimensions
    print(f"\nLoading checkpoint: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=device)

    # Try to infer dimensions from checkpoint
    # Architecture:
    # msg_layers: in -> hidden -> hidden -> hidden -> msg_dim
    # node_layers: (n_f + msg_dim) -> hidden -> hidden -> hidden -> ndim
    #
    # KANLayer.coef has shape (out_dim, in_dim, grid_size+spline_order)
    state_dict = checkpoint["model_state_dict"]

    # msg_layers.1.coef has shape (out_dim, in_dim, grid+order)
    # This is the hidden-to-hidden layer, so out_dim = in_dim = hidden
    msg_layer_1_coef_shape = state_dict["msg_layers.1.coef"].shape
    inferred_hidden = msg_layer_1_coef_shape[1]  # in_dim of layer 1 (= out_dim of layer 0)

    # msg_layers.3.coef is the last layer: hidden -> msg_dim
    msg_layer_3_coef_shape = state_dict["msg_layers.3.coef"].shape
    inferred_msg_dim = msg_layer_3_coef_shape[0]  # out_dim of last message layer

    print(f"\nInferred from checkpoint:")
    print(f"  Hidden dimension: {inferred_hidden}")
    print(f"  Message dimension: {inferred_msg_dim}")

    # Override with inferred values
    if args.hidden != inferred_hidden:
        print(f"  Note: Using inferred hidden={inferred_hidden} instead of specified {args.hidden}")
        args.hidden = inferred_hidden

    if args.msg_dim != inferred_msg_dim:
        print(f"  Note: Using inferred msg_dim={inferred_msg_dim} instead of specified {args.msg_dim}")
        args.msg_dim = inferred_msg_dim

    # Create model with inferred dimensions
    print(f"\nCreating Graph-KAN model...")
    model = OrdinaryGraphKAN(
        n_f=n_features,
        msg_dim=args.msg_dim,
        ndim=dim,
        edge_index=edge_index,
        hidden=args.hidden,
        grid_size=args.grid_size,
        spline_order=args.spline_order,
    )

    # Load state dict
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    model.to(device)

    print(f"Model loaded successfully!")
    print(f"Checkpoint epoch: {checkpoint.get('epoch', 'unknown')}")

    # Visualize message layers
    print(f"\n{'='*60}")
    print(f"Message Function Layers")
    print(f"{'='*60}")

    for i, layer in enumerate(model.msg_layers):
        layer_name = f"Message Layer {i + 1}"

        # Summary plot
        plot_all_features_summary(
            layer, layer_name, feature_names_edge, output_dir, args.x_range, args.n_points, device
        )

        # Detailed per-feature plots (only for first and last layer to avoid clutter)
        if i == 0 or i == len(model.msg_layers) - 1:
            plot_layer_splines_per_feature(
                layer, layer_name, feature_names_edge, output_dir, args.x_range, args.n_points, device
            )

    # Visualize node update layers
    print(f"\n{'='*60}")
    print(f"Node Update Function Layers")
    print(f"{'='*60}")

    for i, layer in enumerate(model.node_layers):
        layer_name = f"Node Layer {i + 1}"

        # For node layers, input is [node_features, aggregated_messages]
        # We'll use generic names since message features are learned
        if i == 0:
            # First node layer gets node features + aggregated message
            node_input_names = feature_names_node + [f"msg_{j}" for j in range(args.msg_dim)]
        else:
            # Subsequent layers get hidden representations
            node_input_names = [f"hidden_{j}" for j in range(layer.in_dim)]

        # Summary plot
        plot_all_features_summary(
            layer, layer_name, node_input_names, output_dir, args.x_range, args.n_points, device
        )

        # Detailed plots for first and last layer
        if i == 0 or i == len(model.node_layers) - 1:
            plot_layer_splines_per_feature(
                layer, layer_name, node_input_names, output_dir, args.x_range, args.n_points, device
            )

    print(f"\n{'='*60}")
    print(f"Visualization Complete!")
    print(f"{'='*60}")
    print(f"\nOutput directory: {output_dir}")
    print(f"\nGenerated plots:")
    print(f"  - *_all_features_summary.png: Overview of all features")
    print(f"  - *_feature_*.png: Detailed per-feature plots")
    print(f"\nKey insights to look for:")
    print(f"  1. Position features (pos_x, pos_y): Should show spatial relationships")
    print(f"  2. Velocity features: May show damping or momentum effects")
    print(f"  3. Mass features: Should scale forces (F ∝ m)")
    print(f"  4. Nonlinear patterns: Gravity is 1/r², look for inverse-like shapes")


if __name__ == "__main__":
    main()
