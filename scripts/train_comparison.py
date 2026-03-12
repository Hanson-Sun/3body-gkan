"""Train small Graph-KAN and Baseline GNN models for comparison."""
from typing import Optional
import argparse

from tqdm import tqdm
import torch
from torch.optim import Adam
from torch_geometric.loader import DataLoader
from pathlib import Path

from nbody_gkan.data.dataset import NBodyDataset
from nbody_gkan.models import OrdinaryGraphKAN, OGN
from nbody_gkan.device import get_device
from nbody_gkan.models.model_loader import ModelLoader


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
    # Training hyperparameters
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--kan_batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--lamb", type=float, default=0)
    return parser.parse_args(args)


def train_model(model, train_loader, val_loader, n_epochs, lr, device, lamb):
    """Train a model and return the trained model and loss history."""
    model.to(device)
    optimizer = Adam(model.parameters(), lr=lr)

    history = {'train': [], 'val': []}

    print(f"Training {model.__class__.__name__}...")

    epoch_bar = tqdm(range(n_epochs), desc='Epochs')
    for _, epoch in enumerate(epoch_bar):
        model.train()
        train_loss = 0.0
        train_bar = tqdm(train_loader, desc=f'  Epoch {epoch:2d} train', leave=False)
        for i, batch in enumerate(train_bar):
            batch = batch.to(device)
            optimizer.zero_grad()
            loss = model.loss(
                batch,
                augment=False,
                lamb=lamb
            )
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * batch.num_graphs

            if i % 10 == 0:
                train_bar.set_postfix(loss=f'{loss.item():.4e}')

        train_loss /= len(train_loader.dataset)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            val_bar = tqdm(val_loader, desc=f'  Epoch {epoch:2d} val  ', leave=False)
            for batch in val_bar:
                batch = batch.to(device)
                loss = model.loss(batch, augment=False)
                val_loss += loss.item() * batch.num_graphs
                val_bar.set_postfix(loss=f'{loss.item():.4e}')

        val_loss /= len(val_loader.dataset)
        history['train'].append(train_loss)
        history['val'].append(val_loss)

        # update outer epoch bar with both losses
        epoch_bar.set_postfix(
            train=f'{train_loss:.6f}',
            val=f'{val_loss:.6f}',
        )
        tqdm.write(f"Epoch {epoch}: Train Loss = {train_loss:.6f}, Val Loss = {val_loss:.6f}")

    return model, history

def main(yaml_params: Optional[dict] = None, checkpoint_dir: Optional[str] = None, data_dir: Optional[str] = None):
    args = parse_args([] if yaml_params is not None else None)

    if yaml_params is not None:
        args.train_data = str(Path(data_dir) / "train.npz")
        args.val_data = str(Path(data_dir) / "val.npz")
        args.checkpoint_dir = checkpoint_dir if checkpoint_dir is not None else args.checkpoint_dir
        args.hidden = yaml_params.get("gnn_hp", {}).get("hidden", args.hidden)
        args.msg_dim = yaml_params.get("gnn_hp", {}).get("msg_dim", args.msg_dim)
        args.kan_hidden_layers = yaml_params.get("gkan_hp", {}).get("hidden_layers", args.kan_hidden_layers)
        args.kan_hidden = yaml_params.get("gkan_hp", {}).get("hidden", args.kan_hidden)
        args.kan_node_hidden_layers = yaml_params.get("gkan_hp", {}).get("node_hidden_layers", args.kan_node_hidden_layers)
        args.kan_node_hidden = yaml_params.get("gkan_hp", {}).get("node_hidden", args.kan_node_hidden)
        args.kan_msg_dim = yaml_params.get("gkan_hp", {}).get("msg_dim", args.kan_msg_dim)
        args.kan_grid_size = yaml_params.get("gkan_hp", {}).get("grid_size", args.kan_grid_size)
        args.kan_lamb_l1 = yaml_params.get("gkan_hp", {}).get("lamb_l1", args.kan_lamb_l1)
        args.kan_lamb_entropy = yaml_params.get("gkan_hp", {}).get("lamb_entropy", args.kan_lamb_entropy)
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

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

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

    kan_model, _history = train_model(kan_model, train_loader, val_loader, args.epochs, args.lr, device, args.lamb)


    # torch.save({
    #     'model_state': kan_model.state_dict(),
    #     'n_features': n_features,
    #     'dim': train_dataset.dim,
    #     'n_bodies': train_dataset.n,
    #     'edge_index': edge_index,
    #     'hidden': args.kan_hidden,
    #     'msg_dim': args.kan_msg_dim,
    #     'grid_size': args.kan_grid_size,
    #     'spline_order': 3,
    #     'hidden_layers': args.kan_hidden_layers,
    #     ''
    # }, checkpoint_dir / 'graph_kan.pt')
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

    gnn_model, _history = train_model(gnn_model, train_loader, val_loader, args.epochs, args.lr, device, args.lamb)

    # torch.save({
    #     'model_state': gnn_model.state_dict(),
    #     'n_features': n_features,
    #     'dim': train_dataset.dim,
    #     'n_bodies': train_dataset.n,
    #     'edge_index': edge_index,
    #     'hidden': args.hidden,
    #     'msg_dim': args.msg_dim,
    # }, checkpoint_dir / 'baseline_gnn.pt')
    gkan_checkpoint_path = f"{checkpoint_dir}/baseline_gnn.pt"
    loader = ModelLoader(OGN, gkan_checkpoint_path)
    loader.save(gnn_model, gkan_checkpoint_path)
    print(f"Saved checkpoint: {gkan_checkpoint_path}\n")

    print("="*60)
    print("Training complete!")
    print("="*60)


if __name__ == "__main__":
    main()
