"""Smoke tests for GraphKAN train/visualization pipeline.

This test focuses on the GraphKAN path in:
- scripts/train_comparison.py
- scripts/visualize_comparison.py
"""

from pathlib import Path
import sys
import json
import shutil

import numpy as np
import torch
from torch_geometric.loader import DataLoader

from nbody_gkan.data.dataset import NBodyDataset
from nbody_gkan.models import OrdinaryGraphKAN


def _import_script_modules():
    """Import script modules by adding the scripts directory to sys.path."""
    scripts_dir = Path(__file__).resolve().parents[1] / "scripts"
    if str(scripts_dir) not in sys.path:
        sys.path.insert(0, str(scripts_dir))

    import train_comparison  # noqa: PLC0415
    import visualize_comparison  # noqa: PLC0415

    return train_comparison, visualize_comparison


def _write_tiny_dataset(path: Path) -> None:
    """Write a tiny synthetic N-body dataset for smoke testing."""
    rng = np.random.default_rng(0)
    n_traj, t_steps, n_bodies, dim = 2, 4, 3, 2

    positions = rng.normal(0.0, 0.5, size=(n_traj, t_steps, n_bodies, dim)).astype(np.float32)
    velocities = rng.normal(0.0, 0.1, size=(n_traj, t_steps, n_bodies, dim)).astype(np.float32)
    masses = np.ones(n_bodies, dtype=np.float32)

    np.savez(path, positions=positions, velocities=velocities, masses=masses)


def _live_print(capsys, msg: str) -> None:
    """Print immediately to terminal even when pytest capture is enabled."""
    with capsys.disabled():
        print(msg, flush=True)


def _count_symbolic_edges(suggestions: dict) -> tuple[int, int]:
    """Count suggested symbolic edges for message and node subnetworks."""
    msg_count = sum(len(edges) for edges in suggestions.get("msg_layers", {}).values())
    node_count = sum(len(edges) for edges in suggestions.get("node_layers", {}).values())
    return msg_count, node_count


def _best_symbolic_entry(payload: dict) -> dict | None:
    """Return the best symbolic entry by R^2 from serialized payload."""
    best = None
    for section in ("msg_layers", "node_layers"):
        for layer_idx, edges in payload.get(section, {}).items():
            for edge_key, info in edges.items():
                candidate = {
                    "section": section,
                    "layer": str(layer_idx),
                    "edge": str(edge_key),
                    "fn": str(info.get("fn", "")),
                    "expression": str(info.get("expression", "")),
                    "r2": float(info.get("r2", 0.0)),
                }
                if best is None or candidate["r2"] > best["r2"]:
                    best = candidate
    return best


def test_graphkan_train_and_visualize_smoke(tmp_path, capsys):
    """Test GraphKAN model creation, short training, and visualization outputs."""
    _live_print(capsys, "\n[smoke] Importing script modules...")
    train_comparison, visualize_comparison = _import_script_modules()

    _live_print(capsys, "[smoke] Creating tiny synthetic dataset...")
    data_path = tmp_path / "tiny_train.npz"
    _write_tiny_dataset(data_path)
    dataset = NBodyDataset(data_path)
    _live_print(capsys, f"[smoke] Dataset ready: n={dataset.n}, dim={dataset.dim}, samples={len(dataset)}")

    train_loader = DataLoader(dataset, batch_size=4, shuffle=True)
    val_loader = DataLoader(dataset, batch_size=4, shuffle=False)

    _live_print(capsys, "[smoke] Building GraphKAN model...")
    n_features = 2 * dataset.dim + 1
    msg_width = [2 * n_features, [4, 1], 4]
    node_width = [n_features + 4, [4, 1], dataset.dim]
    model = OrdinaryGraphKAN(
        n_f=n_features,
        msg_width=msg_width,
        node_width=node_width,
        edge_index=dataset.edge_index,
        msg_mult_arity=2,
        node_mult_arity=2,
        grid_size=3,
        spline_order=2,
        lamb_l1=0.1,
        lamb_entropy=0.1,
    )
    _live_print(capsys, f"[smoke] Model params: {sum(p.numel() for p in model.parameters()):,}")

    # 1) sample model creation and parameter sanity
    assert sum(p.numel() for p in model.parameters()) > 0
    assert any(p.requires_grad for p in model.parameters())
    assert all(not p.requires_grad for p in model.msg_kan.symbolic_fun.parameters())
    assert all(not p.requires_grad for p in model.node_kan.symbolic_fun.parameters())
    _live_print(capsys, "[smoke] Model creation checks passed")

    # 2) small training run through train_comparison GraphKAN path
    _live_print(capsys, "[smoke] Running short GraphKAN training...")
    model, history = train_comparison.train_kan(
        model,
        train_loader,
        val_loader,
        n_epochs=1,
        device="cpu",
        lamb=0.0,
        adam_warmup_epochs=1,
        adam_lr=1e-3,
        grid_update_freq=0,
        grid_update_warmup=0,
        max_grid_updates=0,
    )

    assert len(history["train"]) == 1
    assert len(history["val"]) == 1
    assert np.isfinite(history["train"][0])
    assert np.isfinite(history["val"][0])
    _live_print(capsys, f"[smoke] Training done: train={history['train'][0]:.4e}, val={history['val'][0]:.4e}")

    # Exercise rollout utility from visualize_comparison (GraphKAN path)
    _live_print(capsys, "[smoke] Running rollout utility...")
    sample = dataset[0]
    x_nodes = sample.x
    masses = x_nodes[:, -1]
    pos0 = x_nodes[:, :dataset.dim]
    vel0 = x_nodes[:, dataset.dim:2 * dataset.dim]

    pos_hist, vel_hist = visualize_comparison.rollout(
        model,
        pos0,
        vel0,
        masses,
        dt=0.01,
        n_steps=2,
        edge_index=model.edge_index,
        feature_spec=dataset.feature_spec,
    )

    assert pos_hist.shape == (3, dataset.n, dataset.dim)
    assert vel_hist.shape == (3, dataset.n, dataset.dim)
    _live_print(capsys, f"[smoke] Rollout shapes: pos={pos_hist.shape}, vel={vel_hist.shape}")
    _live_print(
        capsys,
        "[smoke] Rollout sample final body0: "
        f"pos={np.round(pos_hist[-1, 0], 4).tolist()}, "
        f"vel={np.round(vel_hist[-1, 0], 4).tolist()}",
    )

    # 3) GraphKAN-focused visualization checks
    _live_print(capsys, "[smoke] Generating spline visualizations...")
    project_root = Path(__file__).resolve().parents[1]
    out_dir = project_root / "test_outputs" / "graphkan_pipeline_smoke"
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    splines_dir = out_dir / "splines"
    _live_print(capsys, f"[smoke] Visualization output dir: {out_dir}")
    from torch_scatter import scatter_add
    src, dst = model.edge_index
    msg_sample = torch.cat([x_nodes[src], x_nodes[dst]], dim=1)
    with torch.no_grad():
        msg_out = model.msg_kan(msg_sample)
        aggr_msg = scatter_add(msg_out, src, dim=0, dim_size=x_nodes.size(0))
        node_sample = torch.cat([x_nodes, aggr_msg], dim=1)

    msg_sample_batch = msg_sample.repeat(2, 1)
    node_sample_batch = node_sample.repeat(2, 1)

    visualize_comparison.visualize_kan_splines(
        model,
        save_dir=splines_dir,
        msg_sample=msg_sample_batch,
        node_sample=node_sample_batch,
    )
    spline_pngs = list(splines_dir.glob("*.png"))
    assert spline_pngs, "Expected at least one spline visualization image"
    preview_names = ", ".join(sorted(p.name for p in spline_pngs)[:3])
    if len(spline_pngs) > 3:
        preview_names += ", ..."
    _live_print(capsys, f"[smoke] Spline images: {len(spline_pngs)} ({preview_names})")

    msg_plot = out_dir / "kan_msg_network.png"
    node_plot = out_dir / "kan_node_network.png"

    visualize_comparison.visualize_kan_network(
        model,
        msg_sample_batch,
        network="msg",
        save_path=msg_plot,
    )
    visualize_comparison.visualize_kan_network(
        model,
        node_sample_batch,
        network="node",
        save_path=node_plot,
    )

    assert msg_plot.exists() and msg_plot.stat().st_size > 0
    assert node_plot.exists() and node_plot.stat().st_size > 0
    _live_print(
        capsys,
        "[smoke] Network plot images created "
        f"(msg={msg_plot.stat().st_size / 1024:.1f}KB, "
        f"node={node_plot.stat().st_size / 1024:.1f}KB)",
    )
    _live_print(capsys, f"[smoke] Network plots saved: {msg_plot}, {node_plot}")

    # 4) quick symbolic regression verifier
    _live_print(capsys, "[smoke] Running quick symbolic regression verifier...")
    suggestions = visualize_comparison.visualize_symbolic_expressions(
        model,
        x_nodes=x_nodes,
        output_dir=out_dir,
        lib=["x", "sin"],
        threshold=0.0,
        max_batches=1,
    )

    assert "msg_layers" in suggestions and "node_layers" in suggestions
    assert isinstance(suggestions["msg_layers"], dict)
    assert isinstance(suggestions["node_layers"], dict)
    msg_count, node_count = _count_symbolic_edges(suggestions)
    _live_print(capsys, f"[smoke] Symbolic edge counts: msg={msg_count}, node={node_count}")

    symbolic_json = out_dir / "symbolic_regression.json"
    assert symbolic_json.exists() and symbolic_json.stat().st_size > 0
    _live_print(capsys, f"[smoke] Symbolic JSON saved: {symbolic_json}")

    payload = json.loads(symbolic_json.read_text())
    assert "metadata" in payload
    assert "msg_layers" in payload and "node_layers" in payload
    meta = payload.get("metadata", {})
    _live_print(
        capsys,
        "[smoke] Symbolic metadata: "
        f"threshold={meta.get('threshold')}, "
        f"library={meta.get('library')}",
    )

    best_entry = _best_symbolic_entry(payload)
    if best_entry is not None:
        _live_print(
            capsys,
            "[smoke] Example symbolic regression: "
            f"{best_entry['section']} layer={best_entry['layer']} edge={best_entry['edge']} "
            f"fn={best_entry['fn']} R^2={best_entry['r2']:.4f} "
            f"expr={best_entry['expression']}",
        )
    else:
        _live_print(capsys, "[smoke] Example symbolic regression: none found")

    _live_print(capsys, "[smoke] Symbolic regression output verified")

    _live_print(capsys, "[smoke] GraphKAN pipeline smoke test complete")
