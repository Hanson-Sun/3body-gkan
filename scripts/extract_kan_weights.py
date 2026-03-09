"""
Extract and visualize KAN weights from a trained Graph-KAN model.

This script loads a trained Graph-KAN checkpoint and extracts the learned
spline coefficients, allowing you to:
1. Visualize the learned univariate functions
2. Export coefficients for symbolic regression
3. Analyze what the network has learned

Usage:
    python scripts/extract_kan_weights.py --checkpoint checkpoints/graph_kan/best.pt
    python scripts/extract_kan_weights.py --checkpoint PATH --output_dir kan_analysis
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from nbody_gkan.data.dataset import NBodyDataset, get_edge_index
from nbody_gkan.device import get_device
from nbody_gkan.models import OrdinaryGraphKAN


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Extract KAN weights from trained model")

    parser.add_argument(
        "--checkpoint", type=str, required=True, help="Path to model checkpoint"
    )
    parser.add_argument(
        "--train_data",
        type=str,
        default="data/train.npz",
        help="Training data (for getting model dimensions)",
    )
    parser.add_argument(
        "--output_dir", type=str, default="kan_weights", help="Output directory"
    )
    parser.add_argument("--device", type=str, default=None, help="Device (cuda/mps/cpu)")

    # Model hyperparameters (must match checkpoint)
    parser.add_argument("--msg_dim", type=int, default=100, help="Message dimension")
    parser.add_argument("--hidden", type=int, default=300, help="Hidden dimension")
    parser.add_argument("--grid_size", type=int, default=5, help="KAN grid size")
    parser.add_argument("--spline_order", type=int, default=3, help="KAN spline order")

    return parser.parse_args()


def plot_kan_layer_functions(layer, layer_name, save_dir, n_plots=10):
    """
    Plot univariate functions learned by a single KAN layer.

    Each KAN layer has in_dim × out_dim univariate functions.
    We plot a subset of these.

    Parameters
    ----------
    layer : KANLayer
        The KAN layer
    layer_name : str
        Name for the plots
    save_dir : Path
        Directory to save plots
    n_plots : int
        Maximum number of functions to plot
    """
    in_dim = layer.in_dim
    out_dim = layer.out_dim

    print(f"\n{layer_name}: {in_dim} → {out_dim}")

    # Sample input range
    x_range = np.linspace(-3, 3, 300)
    x_tensor = torch.tensor(x_range, dtype=torch.float32).unsqueeze(1)  # (300, 1)

    # Evaluate functions
    with torch.no_grad():
        # For each input dimension, fix others to 0 and vary one
        n_functions = min(in_dim * out_dim, n_plots)
        n_cols = min(4, n_functions)
        n_rows = (n_functions + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3 * n_rows))
        if n_rows == 1 and n_cols == 1:
            axes = np.array([[axes]])
        elif n_rows == 1:
            axes = axes.reshape(1, -1)
        elif n_cols == 1:
            axes = axes.reshape(-1, 1)

        plot_idx = 0
        for in_idx in range(in_dim):
            if plot_idx >= n_functions:
                break

            # Create input: zeros except for dimension in_idx
            x_full = torch.zeros(len(x_range), in_dim)
            x_full[:, in_idx] = x_tensor.squeeze()

            # Forward pass
            y, _, _, _ = layer(x_full)  # (300, out_dim)

            # Plot each output dimension (or just the first few)
            for out_idx in range(out_dim):
                if plot_idx >= n_functions:
                    break

                row = plot_idx // n_cols
                col = plot_idx % n_cols
                ax = axes[row, col]

                y_values = y[:, out_idx].numpy()
                ax.plot(x_range, y_values, linewidth=2)
                ax.set_xlabel(f"Input {in_idx}")
                ax.set_ylabel(f"Output {out_idx}")
                ax.set_title(f"f({in_idx} → {out_idx})")
                ax.grid(alpha=0.3)
                ax.axhline(0, color="k", linewidth=0.5, alpha=0.3)
                ax.axvline(0, color="k", linewidth=0.5, alpha=0.3)

                plot_idx += 1

        # Remove empty subplots
        for idx in range(plot_idx, n_rows * n_cols):
            row = idx // n_cols
            col = idx % n_cols
            fig.delaxes(axes[row, col])

        plt.suptitle(f"{layer_name} - Learned Functions", fontsize=14, fontweight="bold")
        plt.tight_layout()

        save_path = save_dir / f"{layer_name.lower().replace(' ', '_')}_functions.png"
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Saved: {save_path}")
        plt.close()


def extract_spline_coefficients(layer, layer_name):
    """
    Extract spline coefficients from a KAN layer.

    Returns
    -------
    dict
        Dictionary with coefficient arrays and metadata
    """
    # KAN layers store spline coefficients in layer.coef
    # Shape: (out_dim, in_dim, grid_size + spline_order)
    if hasattr(layer, "coef"):
        coef = layer.coef.detach().cpu().numpy()
    else:
        print(f"  Warning: {layer_name} has no 'coef' attribute")
        return None

    return {
        "coefficients": coef,
        "in_dim": layer.in_dim,
        "out_dim": layer.out_dim,
        "grid_size": layer.num if hasattr(layer, "num") else None,
        "spline_order": layer.k if hasattr(layer, "k") else None,
    }


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
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load dataset (just to get dimensions)
    print(f"\nLoading dataset: {args.train_data}")
    dataset = NBodyDataset(args.train_data)
    n = dataset.n
    dim = dataset.dim
    n_features = 2 * dim + 1
    edge_index = dataset.edge_index

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
    print(f"Best validation loss: {checkpoint.get('best_val_loss', 'unknown')}")

    # Extract and visualize message layers
    print(f"\n{'=' * 60}")
    print(f"Extracting Message Function Layers")
    print(f"{'=' * 60}")

    msg_coefficients = {}
    for i, layer in enumerate(model.msg_layers):
        layer_name = f"Message Layer {i + 1}"
        plot_kan_layer_functions(layer, layer_name, output_dir, n_plots=12)

        coef_data = extract_spline_coefficients(layer, layer_name)
        if coef_data:
            msg_coefficients[f"msg_layer_{i + 1}"] = coef_data

    # Extract and visualize node update layers
    print(f"\n{'=' * 60}")
    print(f"Extracting Node Update Function Layers")
    print(f"{'=' * 60}")

    node_coefficients = {}
    for i, layer in enumerate(model.node_layers):
        layer_name = f"Node Layer {i + 1}"
        plot_kan_layer_functions(layer, layer_name, output_dir, n_plots=12)

        coef_data = extract_spline_coefficients(layer, layer_name)
        if coef_data:
            node_coefficients[f"node_layer_{i + 1}"] = coef_data

    # Save coefficients to disk
    print(f"\n{'=' * 60}")
    print(f"Saving Coefficients")
    print(f"{'=' * 60}")

    save_path = output_dir / "kan_coefficients.npz"
    np.savez(
        save_path,
        **{
            f"{key}_coef": val["coefficients"]
            for key, val in {**msg_coefficients, **node_coefficients}.items()
        },
    )
    print(f"Saved: {save_path}")

    # Create summary
    print(f"\n{'=' * 60}")
    print(f"Summary")
    print(f"{'=' * 60}")
    print(f"Output directory: {output_dir}")
    print(f"\nMessage layers: {len(model.msg_layers)}")
    for i, layer in enumerate(model.msg_layers):
        print(f"  Layer {i + 1}: {layer.in_dim} → {layer.out_dim}")

    print(f"\nNode update layers: {len(model.node_layers)}")
    for i, layer in enumerate(model.node_layers):
        print(f"  Layer {i + 1}: {layer.in_dim} → {layer.out_dim}")

    print(f"\nGenerated files:")
    print(f"  - *_functions.png: Plots of learned univariate functions")
    print(f"  - kan_coefficients.npz: Raw spline coefficients")

    print(f"\n{'=' * 60}")
    print(f"Extraction Complete!")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
