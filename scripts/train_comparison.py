"""Train small Graph-KAN and Baseline GNN models for comparison."""

import torch
from torch.optim import Adam
from torch_geometric.loader import DataLoader
from pathlib import Path

from nbody_gkan.data.dataset import NBodyDataset
from nbody_gkan.models import OrdinaryGraphKAN, OGN
from nbody_gkan.device import get_device


# Config
TRAIN_DATA = "data/train.npz"
VAL_DATA = "data/val.npz"
CHECKPOINT_DIR = "checkpoints/comparison"
HIDDEN = 32
MSG_DIM = 16
GRID_SIZE = 5
EPOCHS = 5
BATCH_SIZE = 32
LR = 1e-3


def train_model(model, train_loader, val_loader, n_epochs, lr, device):
    """Train a model and return the trained model."""
    model.to(device)
    optimizer = Adam(model.parameters(), lr=lr)

    print(f"Training {model.__class__.__name__}...")
    print(f"  Batches per epoch: {len(train_loader)}")

    for epoch in range(n_epochs):
        model.train()
        train_loss = 0.0

        for batch_idx, batch in enumerate(train_loader):
            batch = batch.to(device)
            optimizer.zero_grad()
            loss = model.loss(batch, augment=False)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * batch.num_graphs

            # Progress within epoch
            if (batch_idx + 1) % 50 == 0 or (batch_idx + 1) == len(train_loader):
                avg_loss = train_loss / ((batch_idx + 1) * train_loader.batch_size)
                print(f"  Epoch {epoch:2d} [{batch_idx+1:3d}/{len(train_loader)}] Loss={avg_loss:.6f}", end='\r')

        train_loss /= len(train_loader.dataset)

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                loss = model.loss(batch, augment=False)
                val_loss += loss.item() * batch.num_graphs
        val_loss /= len(val_loader.dataset)

        print(f"  Epoch {epoch:2d}: Train={train_loss:.6f}, Val={val_loss:.6f}          ")

    return model


def main():
    device = get_device()
    print(f"Using device: {device}\n")

    # Load data
    print("Loading datasets...")
    train_dataset = NBodyDataset(TRAIN_DATA)
    val_dataset = NBodyDataset(VAL_DATA)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    n_features = 2 * train_dataset.dim + 1
    edge_index = train_dataset.edge_index

    print(f"Dataset: {train_dataset.n} bodies, {train_dataset.dim}D")
    print(f"Train: {len(train_dataset)} samples, Val: {len(val_dataset)} samples\n")

    # Create checkpoint directory
    checkpoint_dir = Path(CHECKPOINT_DIR)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Create and train Graph-KAN
    print("="*60)
    print("Graph-KAN")
    print("="*60)
    kan_model = OrdinaryGraphKAN(
        n_f=n_features, msg_dim=MSG_DIM, ndim=train_dataset.dim,
        edge_index=edge_index, hidden=HIDDEN, grid_size=GRID_SIZE,
        spline_order=3, aggr="add"
    )
    print(f"Parameters: {sum(p.numel() for p in kan_model.parameters()):,}\n")

    kan_model = train_model(kan_model, train_loader, val_loader, EPOCHS, LR, device)

    # Save checkpoint
    torch.save({
        'model_state': kan_model.state_dict(),
        'n_features': n_features,
        'dim': train_dataset.dim,
        'n_bodies': train_dataset.n,
        'edge_index': edge_index,
        'hidden': HIDDEN,
        'msg_dim': MSG_DIM,
        'grid_size': GRID_SIZE,
    }, checkpoint_dir / 'graph_kan.pt')
    print(f"Saved checkpoint: {checkpoint_dir / 'graph_kan.pt'}\n")

    # Create and train Baseline GNN
    print("="*60)
    print("Baseline GNN")
    print("="*60)
    gnn_model = OGN(
        n_f=n_features, msg_dim=MSG_DIM, ndim=train_dataset.dim,
        edge_index=edge_index, hidden=HIDDEN, aggr="add"
    )
    print(f"Parameters: {sum(p.numel() for p in gnn_model.parameters()):,}\n")

    gnn_model = train_model(gnn_model, train_loader, val_loader, EPOCHS, LR, device)

    # Save checkpoint
    torch.save({
        'model_state': gnn_model.state_dict(),
        'n_features': n_features,
        'dim': train_dataset.dim,
        'n_bodies': train_dataset.n,
        'edge_index': edge_index,
        'hidden': HIDDEN,
        'msg_dim': MSG_DIM,
    }, checkpoint_dir / 'baseline_gnn.pt')
    print(f"Saved checkpoint: {checkpoint_dir / 'baseline_gnn.pt'}\n")

    print("="*60)
    print("Training complete!")
    print("="*60)


if __name__ == "__main__":
    main()
