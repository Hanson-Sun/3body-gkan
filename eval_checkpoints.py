"""Evaluate all saved checkpoints on their train and val datasets."""
import sys
from pathlib import Path

import torch
from torch_geometric.loader import DataLoader

from nbody_gkan.data.dataset import NBodyDataset
from nbody_gkan.models import OrdinaryGraphKAN, OGN
from nbody_gkan.models.model_loader import ModelLoader

CHECKPOINT_DIR = Path("checkpoints")
DATA_DIR = Path("data")
BATCH_SIZE = 64

CLASS_MAP = {
    "graph_kan.pt": OrdinaryGraphKAN,
    "baseline_gnn.pt": OGN,
}


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    total_loss = 0.0
    n_samples = 0
    for batch in loader:
        batch = batch.to(device)
        loss = model.loss(batch, augment=False, square=False)
        bs = batch.num_graphs if hasattr(batch, "num_graphs") else 1
        total_loss += loss.item() * bs
        n_samples += bs
    return total_loss / n_samples


def main():
    device = torch.device("cpu")
    results = []

    for exp_dir in sorted(CHECKPOINT_DIR.iterdir()):
        if not exp_dir.is_dir():
            continue

        data_path = DATA_DIR / exp_dir.name
        if not data_path.exists():
            print(f"Skipping {exp_dir.name}: no matching data dir")
            continue

        for ckpt_name, model_class in CLASS_MAP.items():
            ckpt_path = exp_dir / ckpt_name
            if not ckpt_path.exists():
                continue

            print(f"Loading {exp_dir.name}/{ckpt_name} ...", end=" ", flush=True)
            loader = ModelLoader(model_class, ckpt_path)
            model, _ = loader.load()
            model.to(device)

            train_ds = NBodyDataset(str(data_path / "train.npz"))
            val_ds = NBodyDataset(str(data_path / "val.npz"))
            train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE)
            val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE)

            train_loss = evaluate(model, train_loader, device)
            val_loss = evaluate(model, val_loader, device)

            results.append((exp_dir.name, ckpt_name.replace(".pt", ""), train_loss, val_loss))
            print(f"train={train_loss:.6e}  val={val_loss:.6e}")

    if not results:
        print("No checkpoints found.")
        sys.exit(1)

    print("\n" + "=" * 72)
    print(f"{'Experiment':<25} {'Model':<15} {'Train Loss':>12} {'Val Loss':>12}")
    print("-" * 72)
    for name, model, tl, vl in results:
        print(f"{name:<25} {model:<15} {tl:>12.6e} {vl:>12.6e}")
    print("=" * 72)


if __name__ == "__main__":
    main()