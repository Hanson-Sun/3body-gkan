"""Train small Graph-KAN and Baseline GNN models for comparison."""
from typing import Optional
import argparse

import torch
from torch_geometric.loader import DataLoader
from pathlib import Path
import matplotlib.pyplot as plt
from tqdm import tqdm

from nbody_gkan.data.dataset import NBodyDataset
from nbody_gkan.models import OrdinaryGraphKAN, OGN
from nbody_gkan.device import get_device
from nbody_gkan.models.model_loader import ModelLoader

from nbody_gkan.training import GNNTrainer, KANTrainer

def parse_args(args=None):
    parser = argparse.ArgumentParser(description="Train Graph-KAN and Baseline GNN models for comparison.")
    # Data
    parser.add_argument("--train_data", type=str, default="data/train.npz")
    parser.add_argument("--val_data", type=str, default="data/val.npz")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints/comparison")
    # GNN hyperparameters
    parser.add_argument("--hidden", type=int, default=200)
    parser.add_argument("--msg_dim", type=int, default=32)
    # GKAN hyperparameters
    parser.add_argument("--kan_hidden_layers", type=int, default=1)
    parser.add_argument("--kan_hidden", type=int, default=8)
    parser.add_argument("--kan_node_hidden_layers", type=int, default=1)
    parser.add_argument("--kan_node_hidden", type=int, default=8)
    parser.add_argument("--kan_msg_dim", type=int, default=16)
    parser.add_argument("--kan_grid_size", type=int, default=5)
    parser.add_argument("--kan_lamb_l1", type=float, default=1.0)
    parser.add_argument("--kan_lamb_entropy", type=float, default=2.0)

    # TODO: add these to yaml
    parser.add_argument("--kan_adam_warmup_epochs", type=int, default=10)
    parser.add_argument("--kan_grid_update_freq", type=int, default=10)
    parser.add_argument("--kan_grid_update_warmup", type=int, default=5)
    parser.add_argument("--kan_max_grid_updates", type=int, default=4)
    # Training hyperparameters
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--kan_batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--lamb", type=float, default=0)


    return parser.parse_args(args)



def train_gnn(model, train_loader, val_loader, n_epochs, lr, device, lamb):
    trainer = GNNTrainer(
        model, train_loader, val_loader,
        lr=lr, device=device,
        checkpoint_dir=None,
    )
    trainer.lamb = lamb
    trainer.train(
        n_epochs=n_epochs,
        augment=False,
        save_every=n_epochs,
    )
    return model, trainer.history


def train_kan(model, train_loader, val_loader, n_epochs, device, lamb,
              adam_warmup_epochs: int = 0,
              adam_lr: float = 1e-3,
              grid_update_freq: int = 10,
              grid_update_warmup: int = 5,
              max_grid_updates: int = 4):
    trainer = KANTrainer(
        model, train_loader, val_loader,
        lbfgs_lr=1.0,
        adam_lr=adam_lr,
        adam_warmup_epochs=adam_warmup_epochs,
        device=device,
        checkpoint_dir=None,
        grid_update_freq=grid_update_freq,
        grid_update_warmup=grid_update_warmup,
        max_grid_updates=max_grid_updates,
    )
    trainer.lamb = lamb
    trainer.train(n_epochs=n_epochs, augment=False, save_every=n_epochs)
    return model, trainer.history

def visualize_training_loss(
        history: dict[str, list[float]],
        save_path: str | Path | None = None,
        title: str = 'Training Loss'
) -> None:
    """
    Plot train and validation loss curves over epochs.

    Args:
        history:   dict with 'train' and 'val' loss lists
        save_path: if given, save figure here
        title:     plot title
    """
    epochs = range(len(history['train']))

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(epochs, history['train'], label='Train',      linewidth=2)
    ax.plot(epochs, history['val'],   label='Validation', linewidth=2, linestyle='--')

    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    else:
        plt.show()

    plt.close()

def main(yaml_params: Optional[dict] = None, checkpoint_dir: Optional[str] = None, data_dir: Optional[str] = None):
    args = parse_args([] if yaml_params is not None else None)

    if yaml_params is not None:
        args.train_data = str(Path(data_dir) / "train.npz")
        args.val_data = str(Path(data_dir) / "val.npz")
        args.checkpoint_dir = checkpoint_dir if checkpoint_dir is not None else args.checkpoint_dir
        args.hidden = yaml_params.get("gnn_hp", {}).get("hidden", args.hidden)
        args.msg_dim = yaml_params.get("gnn_hp", {}).get("msg_dim", args.msg_dim)
        args.kan_hidden_layers         = yaml_params.get("gkan_hp", {}).get("hidden_layers",       args.kan_hidden_layers)
        args.kan_hidden                = yaml_params.get("gkan_hp", {}).get("hidden",               args.kan_hidden)
        args.kan_node_hidden_layers    = yaml_params.get("gkan_hp", {}).get("node_hidden_layers",   args.kan_node_hidden_layers)
        args.kan_node_hidden           = yaml_params.get("gkan_hp", {}).get("node_hidden",          args.kan_node_hidden)
        args.kan_msg_dim               = yaml_params.get("gkan_hp", {}).get("msg_dim",              args.kan_msg_dim)
        args.kan_grid_size             = yaml_params.get("gkan_hp", {}).get("grid_size",            args.kan_grid_size)
        args.kan_lamb_l1               = yaml_params.get("gkan_hp", {}).get("lamb_l1",              args.kan_lamb_l1)
        args.kan_lamb_entropy          = yaml_params.get("gkan_hp", {}).get("lamb_entropy",         args.kan_lamb_entropy)
        args.kan_adam_warmup_epochs    = yaml_params.get("gkan_hp", {}).get("adam_warmup_epochs",   args.kan_adam_warmup_epochs)
        args.kan_grid_update_freq      = yaml_params.get("gkan_hp", {}).get("grid_update_freq",     args.kan_grid_update_freq)
        args.kan_grid_update_warmup    = yaml_params.get("gkan_hp", {}).get("grid_update_warmup",   args.kan_grid_update_warmup)
        args.kan_max_grid_updates      = yaml_params.get("gkan_hp", {}).get("max_grid_updates",     args.kan_max_grid_updates)
        args.epochs = yaml_params.get("training_hp", {}).get("epochs", args.epochs)
        args.batch_size = yaml_params.get("training_hp", {}).get("batch_size", args.batch_size)
        args.kan_batch_size = yaml_params.get("training_hp", {}).get("kan_batch_size", args.kan_batch_size)
        args.lr = float(yaml_params.get("training_hp", {}).get("lr", args.lr))
        args.lamb = yaml_params.get("training_hp", {}).get("lamb", args.lamb)

    device = get_device()
    print(f"Using device: {device}\n")

    # Load data
    print("Loading datasets...")
    train_dataset = NBodyDataset(args.train_data)
    val_dataset = NBodyDataset(args.val_data)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=8, pin_memory=True, persistent_workers=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=8, pin_memory=True, persistent_workers=True)

    kan_train_loader = DataLoader(train_dataset, batch_size=args.kan_batch_size, shuffle=True, num_workers=8, pin_memory=True, persistent_workers=True)
    kan_val_loader = DataLoader(val_dataset, batch_size=args.kan_batch_size, shuffle=False, num_workers=8, pin_memory=True, persistent_workers=True)

    n_features = 2 * train_dataset.dim + 1
    edge_index = train_dataset.edge_index

    print(f"Dataset: {train_dataset.n} bodies, {train_dataset.dim}D")
    print(f"Train: {len(train_dataset)} samples, Val: {len(val_dataset)} samples\n")

    # Create checkpoint directory
    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Create and train Graph-KAN
    print("="*60)
    print("Graph-KAN")
    print("="*60)
    kan_model = OrdinaryGraphKAN(
        n_f=n_features, msg_dim=args.kan_msg_dim, ndim=train_dataset.dim,
        edge_index=edge_index, hidden=args.kan_hidden, grid_size=args.kan_grid_size,
        spline_order=3, aggr="add", hidden_layers=args.kan_hidden_layers,
        lamb_l1=args.kan_lamb_l1, lamb_entropy=args.kan_lamb_entropy,
        node_hidden=args.kan_node_hidden, node_hidden_layers=args.kan_node_hidden_layers
    )
    kan_model.summary()
    print(" ")

    kan_model, kan_history = train_kan(
        kan_model, kan_train_loader, kan_val_loader,
        n_epochs=args.epochs,
        device=device,
        lamb=args.lamb,
        adam_warmup_epochs=args.kan_adam_warmup_epochs,
        adam_lr=args.lr,
        grid_update_freq=args.kan_grid_update_freq,
        grid_update_warmup=args.kan_grid_update_warmup,
        max_grid_updates=args.kan_max_grid_updates,
    )
    visualize_training_loss(kan_history,
                            title='Graph-KAN Training Loss',
                            save_path=f'{checkpoint_dir}/kan_loss.png')
    gkan_checkpoint_path = f"{checkpoint_dir}/graph_kan.pt"
    loader = ModelLoader(OrdinaryGraphKAN, gkan_checkpoint_path)
    loader.save(kan_model, gkan_checkpoint_path)
    print(f"Saved checkpoint: {gkan_checkpoint_path}\n")

    # Create and train Baseline GNN
    print("="*60)
    print("Baseline GNN")
    print("="*60)
    gnn_model = OGN(
        n_f=n_features, msg_dim=args.msg_dim, ndim=train_dataset.dim,
        edge_index=edge_index, hidden=args.hidden, aggr="add"
    )
    gnn_model.summary()
    print(" ")

    gnn_model, gnn_history = train_gnn(gnn_model, train_loader, val_loader, args.epochs, args.lr, device, args.lamb)
    visualize_training_loss(gnn_history,
                            title='GNN Training Loss',
                            save_path=f'{checkpoint_dir}/gnn_loss.png')

    gnn_checkpoint_path = f"{checkpoint_dir}/baseline_gnn.pt"
    loader = ModelLoader(OGN, gnn_checkpoint_path)
    loader.save(gnn_model, gnn_checkpoint_path)
    print(f"Saved checkpoint: {gnn_checkpoint_path}\n")
    print("="*60)
    print("Training complete!")
    print("="*60)


if __name__ == "__main__":
    main()
