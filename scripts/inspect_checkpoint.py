"""
Inspect a Graph-KAN checkpoint and print its architecture.

Usage:
    python scripts/inspect_checkpoint.py checkpoints/graph_kan/best.pt
"""

import argparse
import torch


def main():
    parser = argparse.ArgumentParser(description="Inspect Graph-KAN checkpoint")
    parser.add_argument("checkpoint", type=str, help="Path to checkpoint")
    args = parser.parse_args()

    print(f"Loading checkpoint: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location="cpu")

    print(f"\nCheckpoint keys: {list(ckpt.keys())}")

    if "epoch" in ckpt:
        print(f"Epoch: {ckpt['epoch']}")
    if "best_val_loss" in ckpt:
        print(f"Best validation loss: {ckpt['best_val_loss']}")

    print(f"\nAnalyzing architecture from state_dict...")
    state = ckpt["model_state_dict"]

    print(f"\nMessage layers:")
    for i in range(10):  # Try up to 10 layers
        key = f"msg_layers.{i}.coef"
        if key not in state:
            break
        shape = state[key].shape
        print(f"  Layer {i}: {shape[1]:3d} → {shape[0]:3d}  (shape: {shape})")

    print(f"\nNode layers:")
    for i in range(10):
        key = f"node_layers.{i}.coef"
        if key not in state:
            break
        shape = state[key].shape
        print(f"  Layer {i}: {shape[1]:3d} → {shape[0]:3d}  (shape: {shape})")

    # Try to infer hyperparameters
    print(f"\nInferred hyperparameters:")

    # Look at layer 2 (should be hidden -> hidden for standard arch)
    if "msg_layers.2.coef" in state:
        hidden_msg = state["msg_layers.2.coef"].shape[0]
        print(f"  Hidden dimension (message): ~{hidden_msg}")

    if "node_layers.2.coef" in state:
        hidden_node = state["node_layers.2.coef"].shape[0]
        print(f"  Hidden dimension (node): ~{hidden_node}")

    # Last message layer gives msg_dim
    for i in range(10, -1, -1):
        if f"msg_layers.{i}.coef" in state:
            msg_dim = state[f"msg_layers.{i}.coef"].shape[0]
            print(f"  Message dimension: {msg_dim}")
            break

    # Grid size
    if "msg_layers.0.grid" in state:
        grid_shape = state["msg_layers.0.grid"].shape
        grid_size = grid_shape[1] - 1  # grid has shape (in_dim, grid_size+1)
        print(f"  Grid size: {grid_size}")

    # Spline order (from coef shape)
    if "msg_layers.0.coef" in state:
        coef_shape = state["msg_layers.0.coef"].shape
        # coef shape is (out_dim, in_dim, grid_size + spline_order)
        # grid_size + spline_order = coef_shape[2]
        if "msg_layers.0.grid" in state:
            grid_size = state["msg_layers.0.grid"].shape[1] - 1
            spline_order = coef_shape[2] - grid_size
            print(f"  Spline order: {spline_order}")

    print(f"\n**WARNING**: This model has a non-standard architecture!")
    print(f"The visualization scripts expect: in → hidden → hidden → hidden → out")
    print(f"But this model has a different structure (see layer dimensions above)")
    print(f"\nTo visualize, you'll need to manually specify dimensions or retrain with standard architecture.")


if __name__ == "__main__":
    main()
