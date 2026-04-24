"""Print parameter counts for all saved checkpoints."""
from pathlib import Path

import torch

from nbody_gkan.models import OrdinaryGraphKAN, OGN
from nbody_gkan.models.model_loader import ModelLoader

CHECKPOINT_DIR = Path("checkpoints")

CLASS_MAP = {
    "graph_kan.pt": OrdinaryGraphKAN,
    "baseline_gnn.pt": OGN,
}

results = []
for exp_dir in sorted(CHECKPOINT_DIR.iterdir()):
    if not exp_dir.is_dir():
        continue
    for ckpt_name, model_class in CLASS_MAP.items():
        ckpt_path = exp_dir / ckpt_name
        if not ckpt_path.exists():
            continue
        model, _ = ModelLoader(model_class, ckpt_path).load()
        n_params = sum(p.numel() for p in model.parameters())
        results.append((exp_dir.name, ckpt_name.replace(".pt", ""), n_params))

print(f"{'Experiment':<25} {'Model':<15} {'Params':>10}")
print("-" * 52)
for name, model, n in results:
    print(f"{name:<25} {model:<15} {n:>10,}")
