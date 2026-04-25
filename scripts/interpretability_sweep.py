"""Per-feature response curves for trained GKAN message and node KANs.

For each input dimension, sweeps it across its training-data range while
holding all other dimensions at their median. Plots output vs. swept input
so you can see what relationship each spline learned (linear, 1/r², etc.).
"""

import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from nbody_gkan.models import OrdinaryGraphKAN
from nbody_gkan.models.model_loader import ModelLoader
from nbody_gkan.models.edge_features import compute_edge_features

EXPERIMENTS = ["small", "small_rel_pos", "small_rel_pos_dsq", "small_rel_pos_idc"]
CHECKPOINT_DIR = Path("checkpoints")
DATA_DIR = Path("data")
OUTPUT_DIR = Path("interpretability")
N_SWEEP = 300
MIN_RANGE = 1.0


def get_feature_names(model):
    base = ["px", "py", "vx", "vy", "m"]
    n_f = model.n_f

    msg_names = [f"{v}_i" for v in base[:n_f]] + [f"{v}_j" for v in base[:n_f]]
    for feat in (model.edge_augmentations or []):
        if feat == "rel_pos":
            msg_names += ["dx", "dy"]
        elif feat == "dist_sq":
            msg_names += ["r²"]
        elif feat == "inv_dist_cu":
            msg_names += ["r⁻³"]

    node_names = base[:n_f] + [f"msg_{i}" for i in range(model.msg_dim)]
    msg_out = [f"msg_{i}" for i in range(model.msg_dim)]
    node_out = ["a_x", "a_y"]
    return msg_names, node_names, msg_out, node_out


def build_msg_kan_inputs(model, data_path):
    """Construct msg_kan inputs [x_i, x_j, augmentations] from training data."""
    data = np.load(data_path)
    positions = data["positions"]
    velocities = data["velocities"]
    masses = torch.from_numpy(data["masses"]).float()

    rng = np.random.default_rng(42)
    n_frames = positions.shape[0]
    sample_idx = rng.choice(n_frames, min(500, n_frames), replace=False)

    all_inputs = []
    for idx in sample_idx:
        pos = torch.from_numpy(positions[idx]).float()
        vel = torch.from_numpy(velocities[idx]).float()
        m = masses.unsqueeze(1)
        x = torch.cat([pos, vel, m], dim=1)

        src, dst = model.edge_index[0], model.edge_index[1]
        x_i, x_j = x[dst], x[src]

        inp = torch.cat([x_i, x_j], dim=1)
        if model.edge_augmentations:
            aug = compute_edge_features(
                x_i, x_j, model.ndim, model.edge_augmentations, model.softening
            )
            inp = torch.cat([inp, aug], dim=1)
        all_inputs.append(inp)

    return torch.cat(all_inputs, dim=0)


def build_node_kan_inputs(model, msg_inputs):
    """Run msg_kan on representative inputs, pair with node features."""
    with torch.no_grad():
        msgs = model.msg_kan(msg_inputs)
    x_part = msg_inputs[:, :model.n_f]
    return torch.cat([x_part, msgs], dim=1)


def compute_sweep_ranges(data_tensor):
    """Compute per-dim sweep ranges from data, enforcing a minimum width."""
    lo = data_tensor.quantile(0.02, dim=0)
    hi = data_tensor.quantile(0.98, dim=0)
    width = hi - lo
    too_narrow = width < MIN_RANGE
    mid = (lo + hi) / 2
    lo = torch.where(too_narrow, mid - MIN_RANGE / 2, lo)
    hi = torch.where(too_narrow, mid + MIN_RANGE / 2, hi)
    return list(zip(lo.tolist(), hi.tolist()))


def sweep_feature(kan, baseline, feat_idx, lo, hi, n_points=N_SWEEP):
    sweep_vals = torch.linspace(lo, hi, n_points)
    inputs = baseline.unsqueeze(0).expand(n_points, -1).clone()
    inputs[:, feat_idx] = sweep_vals
    with torch.no_grad():
        outputs = kan(inputs)
    return sweep_vals.numpy(), outputs.numpy()


def plot_responses(kan, baseline, feat_names, out_names, ranges, title, save_path):
    n_in = len(feat_names)
    n_out = len(out_names)

    fig, axes = plt.subplots(
        n_out, n_in,
        figsize=(2.8 * n_in, 2.8 * n_out),
        squeeze=False,
        sharey='row',
    )
    fig.suptitle(title, fontsize=13, fontweight='bold', y=1.02)

    colors = plt.cm.tab10.colors

    for col, (fname, (lo, hi)) in enumerate(zip(feat_names, ranges)):
        x_vals, y_vals = sweep_feature(kan, baseline, col, lo, hi)
        for row, oname in enumerate(out_names):
            ax = axes[row, col]
            ax.plot(x_vals, y_vals[:, row], linewidth=1.5, color=colors[row])
            ax.set_xlabel(fname, fontsize=9)
            if col == 0:
                ax.set_ylabel(oname, fontsize=10)
            ax.tick_params(labelsize=7)
            ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {save_path}")


def analyze_model(name):
    ckpt_path = CHECKPOINT_DIR / name / "graph_kan.pt"
    data_path = DATA_DIR / name / "train.npz"

    if not ckpt_path.exists():
        print(f"  Skipping {name}: no checkpoint")
        return

    print(f"\n{'='*60}")
    print(f"  {name}")
    print(f"{'='*60}")

    model, _ = ModelLoader(OrdinaryGraphKAN, str(ckpt_path)).load()
    msg_names, node_names, msg_out, node_out = get_feature_names(model)

    print(f"  msg_kan: {' | '.join(msg_names)} -> {msg_out}")
    print(f"  node_kan: {' | '.join(node_names)} -> {node_out}")

    msg_inputs = build_msg_kan_inputs(model, data_path)
    node_inputs = build_node_kan_inputs(model, msg_inputs)

    msg_baseline = msg_inputs.median(dim=0).values
    msg_ranges = compute_sweep_ranges(msg_inputs)

    node_baseline = node_inputs.median(dim=0).values
    node_ranges = compute_sweep_ranges(node_inputs)

    out_dir = OUTPUT_DIR / name
    out_dir.mkdir(parents=True, exist_ok=True)

    plot_responses(
        model.msg_kan, msg_baseline, msg_names, msg_out, msg_ranges,
        f"{name} — Message KAN responses",
        out_dir / "msg_kan_responses.png",
    )

    plot_responses(
        model.node_kan, node_baseline, node_names, node_out, node_ranges,
        f"{name} — Node Update KAN responses",
        out_dir / "node_kan_responses.png",
    )


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for name in EXPERIMENTS:
        analyze_model(name)
    print(f"\nAll plots saved to {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()