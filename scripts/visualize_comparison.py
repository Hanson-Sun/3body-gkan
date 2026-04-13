"""Visualize trained models: rollout comparison and spline analysis."""

import argparse
import json
from pathlib import Path
from typing import Optional
from typing import Any

import numpy as np
import torch
import matplotlib
import shutil
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader as PyGLoader

from nbody_gkan.data.dataset import (
    FORCE_FN_MAP as DATASET_FORCE_FN_MAP,
    build_node_features_torch,
    node_feature_dim,
    normalize_feature_spec,
)
from nbody_gkan.models import OrdinaryGraphKAN, OGN
from nbody_gkan.nbody import NBodySimulator
from nbody_gkan.models.model_loader import ModelLoader


DEFAULT_WARMUP_GRAPHS = 256


def _extract_graphkan_subnet_info(model: 'OrdinaryGraphKAN', network: str) -> dict[str, Any]:
    """Extract GraphKAN subnet metadata with backward-compatible fallbacks."""
    if network not in {'msg', 'node'}:
        raise ValueError(f"network must be 'msg' or 'node', got {network!r}")

    if network == 'msg':
        layers = model.msg_layers
        subnet = model.msg_kan
        width = list(getattr(model, 'msg_width', []))
        mult_arity = getattr(model, 'msg_mult_arity', 2)
        mult_nodes = getattr(model, 'msg_mult_nodes', 0)
    else:
        layers = model.node_layers
        subnet = model.node_kan
        width = list(getattr(model, 'node_width', []))
        mult_arity = getattr(model, 'node_mult_arity', 2)
        mult_nodes = getattr(model, 'node_mult_nodes', 0)

    if not width:
        width = [int(layers[0].in_dim)] + [int(l.out_dim) for l in layers]

    return {
        'layers': layers,
        'subnet': subnet,
        'width': width,
        'mult_arity': mult_arity,
        'mult_nodes': mult_nodes,
    }


def parse_args(args=None):
    parser = argparse.ArgumentParser(description="Visualize trained models: rollout comparison and spline analysis.")
    parser.add_argument("--output_dir",        type=str,   default="visualizations")
    parser.add_argument("--checkpoint_dir",    type=str,   default="checkpoints/comparison")
    parser.add_argument("--data_file",         type=str,   default="data/train.npz")
    parser.add_argument(
        "--include_velocity",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Legacy alias for toggling velocity in node features. "
            "Ignored when --input_features is provided or checkpoint has feature spec."
        ),
    )
    parser.add_argument(
        "--input_features",
        type=str,
        default=None,
        help=(
            "Optional JSON feature spec override, e.g. "
            "'{\"include\": [\"pos\", \"mass\"]}'."
        ),
    )
    parser.add_argument("--rollout_time",      type=float, default=5.0)
    parser.add_argument("--dt",                type=float, default=0.01)
    parser.add_argument("--save_video",        action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--plot_trajectories", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--plot_splines",      action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--prune_kan",         action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--prune_edge_threshold", type=float, default=3e-2)
    parser.add_argument("--prune_node_threshold", type=float, default=None)
    parser.add_argument("--warmup_graphs", type=int, default=DEFAULT_WARMUP_GRAPHS)
    parser.add_argument(
        "--symbolic",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Run symbolic regression stage (enable/disable)",
    )
    return parser.parse_args(args)


def _threshold_or_none(value: float | None) -> float | None:
    if value is None:
        return None
    value = float(value)
    return value if value > 0 else None


def _decode_npz_scalar(value) -> str:
    """Decode scalar or byte payloads loaded from NPZ metadata."""
    if isinstance(value, np.ndarray):
        if value.shape == ():
            value = value.item()
        elif value.size == 1:
            value = value.reshape(()).item()
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


def _load_force_config(data) -> tuple[str, Any, dict]:
    """Load force function name/callable/kwargs from dataset metadata."""
    if "force_name" in data.files:
        force_name = _decode_npz_scalar(data["force_name"])
    else:
        force_name = "gravity"

    force_kwargs: dict[str, Any] = {}
    if "force_kwargs" in data.files:
        raw_kwargs = _decode_npz_scalar(data["force_kwargs"])
        if raw_kwargs and raw_kwargs != "None":
            try:
                parsed = json.loads(raw_kwargs)
                if isinstance(parsed, dict):
                    force_kwargs = parsed
            except json.JSONDecodeError:
                force_kwargs = {}

    if not force_kwargs and force_name in {"gravity", "linear_gravity", "cubic_gravity", "hooke_pairwise"}:
        force_kwargs = {"G": 1.0, "softening": 1e-2}

    force_fn = DATASET_FORCE_FN_MAP.get(force_name)
    if force_fn is None:
        valid = ", ".join(sorted(DATASET_FORCE_FN_MAP))
        raise ValueError(f"Unknown force function {force_name!r} in {data}; available: {valid}")

    return force_name, force_fn, force_kwargs


def _sample_indices(total: int, max_items: int) -> np.ndarray:
    if total <= 0:
        raise ValueError("total must be > 0")
    if max_items <= 0:
        raise ValueError("max_items must be > 0")
    if total <= max_items:
        return np.arange(total, dtype=np.int64)
    return np.linspace(0, total - 1, num=max_items, dtype=np.int64)


def _flatten_frames(positions: np.ndarray, velocities: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if positions.shape != velocities.shape:
        raise ValueError(
            f"positions and velocities must have matching shapes; got {positions.shape} and {velocities.shape}."
        )

    if positions.ndim == 4:
        n_traj, n_steps, n_nodes, dim = positions.shape
        return (
            positions.reshape(n_traj * n_steps, n_nodes, dim),
            velocities.reshape(n_traj * n_steps, n_nodes, dim),
        )

    if positions.ndim == 3:
        return positions, velocities

    raise ValueError(f"Unsupported positions shape {positions.shape}; expected 4D or 3D arrays.")


def _coerce_feature_spec_arg(value):
    if value is None:
        return None
    if isinstance(value, (dict, list, tuple)):
        return value
    if isinstance(value, str):
        stripped = value.strip()
        if stripped == "":
            return None
        try:
            return json.loads(stripped)
        except json.JSONDecodeError as exc:
            raise ValueError(
                "input_features must be JSON dict/list, e.g. "
                "'{\"include\":[\"pos\",\"mass\"]}'"
            ) from exc
    raise ValueError(f"input_features must be dict/list/JSON-string, got {type(value).__name__}.")


def _infer_legacy_feature_spec_from_n_f(n_f: int, dim: int) -> dict[str, list[str]]:
    with_velocity = 2 * dim + 1
    without_velocity = dim + 1

    if n_f == with_velocity:
        return {"include": ["pos", "vel", "mass"], "augment": []}
    if n_f == without_velocity:
        return {"include": ["pos", "mass"], "augment": []}

    raise ValueError(
        f"Cannot infer feature spec from checkpoint: n_f={n_f}, dim={dim}. "
        f"Expected n_f={with_velocity} ([pos, vel, mass]) or n_f={without_velocity} ([pos, mass]). "
        "For newer checkpoints, store input_feature_spec during training or pass --input_features."
    )


def _tile_edge_index(edge_index: torch.Tensor, n_nodes: int, n_graphs: int) -> torch.Tensor:
    edge_index = edge_index.to(dtype=torch.long)
    if n_graphs == 1:
        return edge_index

    n_edges = edge_index.shape[1]
    offsets = (torch.arange(n_graphs, dtype=torch.long) * n_nodes).repeat_interleave(n_edges)
    return edge_index.repeat(1, n_graphs) + offsets.unsqueeze(0)


def _build_representative_samples(
    positions: np.ndarray,
    velocities: np.ndarray,
    masses: torch.Tensor,
    edge_index: torch.Tensor,
    feature_spec: dict[str, list[str]],
    max_graphs: int = DEFAULT_WARMUP_GRAPHS,
) -> dict[str, Any]:
    flat_pos, flat_vel = _flatten_frames(positions, velocities)
    total_graphs = int(flat_pos.shape[0])
    graph_idx = _sample_indices(total_graphs, max_graphs)

    pos_sel = torch.from_numpy(flat_pos[graph_idx]).float()
    vel_sel = torch.from_numpy(flat_vel[graph_idx]).float()

    n_graphs = int(pos_sel.shape[0])
    n_nodes = int(pos_sel.shape[1])
    if masses.numel() != n_nodes:
        raise ValueError(
            f"Mass vector length ({masses.numel()}) must match node count ({n_nodes})."
        )

    masses_graph = masses.float().view(1, n_nodes).expand(n_graphs, -1)
    x_graphs = build_node_features_torch(
        pos=pos_sel,
        vel=vel_sel,
        masses=masses_graph,
        feature_spec=feature_spec,
    )
    x_nodes = x_graphs.reshape(n_graphs * n_nodes, x_graphs.shape[-1])

    batched_edge_index = _tile_edge_index(edge_index=edge_index, n_nodes=n_nodes, n_graphs=n_graphs)
    src, dst = batched_edge_index
    msg_inputs = torch.cat([x_nodes[src], x_nodes[dst]], dim=1)

    return {
        "x_nodes": x_nodes,
        "edge_index": batched_edge_index,
        "msg_inputs": msg_inputs,
        "n_graphs": n_graphs,
        "n_node_samples": int(x_nodes.shape[0]),
        "n_message_samples": int(msg_inputs.shape[0]),
    }


def rollout(model, pos0, vel0, masses, dt, n_steps, edge_index, feature_spec: dict[str, list[str]]):
    """Rollout dynamics using leapfrog integration."""
    positions = [pos0.cpu().numpy()]
    velocities = [vel0.cpu().numpy()]

    pos = pos0.clone()
    vel = vel0.clone()

    with torch.no_grad():
        for _ in range(n_steps):
            x = build_node_features_torch(pos=pos, vel=vel, masses=masses, feature_spec=feature_spec)
            graph = Data(x=x, edge_index=edge_index)

            # Leapfrog step
            acc = model.just_derivative(graph, augment=False)
            vel_half = vel + 0.5 * dt * acc
            pos = pos + dt * vel_half

            # Recompute acceleration at new position
            x = build_node_features_torch(
                pos=pos,
                vel=vel_half,
                masses=masses,
                feature_spec=feature_spec,
            )
            graph = Data(x=x, edge_index=edge_index)
            acc_new = model.just_derivative(graph, augment=False)
            vel = vel_half + 0.5 * dt * acc_new

            positions.append(pos.cpu().numpy())
            velocities.append(vel.cpu().numpy())

    return np.array(positions), np.array(velocities)


def animate_rollout(positions_dict, dt, video_name='rollout_comparison.mp4'):
    """Create animated comparison with video export."""
    print(f"\nCreating animation...")

    n_models = len(positions_dict)
    fig, axes = plt.subplots(1, n_models, figsize=(6 * n_models, 6))
    if n_models == 1:
        axes = [axes]

    first_pos = list(positions_dict.values())[0]
    n_steps, n_bodies, _ = first_pos.shape

    # Compute shared square extent across all models
    all_pos = np.concatenate(list(positions_dict.values()), axis=0)
    margin = 0.5
    x_min, x_max = all_pos[:, :, 0].min() - margin, all_pos[:, :, 0].max() + margin
    y_min, y_max = all_pos[:, :, 1].min() - margin, all_pos[:, :, 1].max() + margin

    # Force square extent so axes aren't squished
    max_range = max(x_max - x_min, y_max - y_min)
    x_mid, y_mid = (x_min + x_max) / 2, (y_min + y_max) / 2
    x_min, x_max = x_mid - max_range / 2, x_mid + max_range / 2
    y_min, y_max = y_mid - max_range / 2, y_mid + max_range / 2

    scatters, trails, time_texts = [], [], []
    trail_data = {name: [[] for _ in range(n_bodies)] for name in positions_dict}
    trail_length = 50

    for ax, name in zip(axes, positions_dict):
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_title(name)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)

        scatters.append(ax.scatter([], [], s=100, alpha=0.8))
        trails.append([ax.plot([], [], '-', alpha=0.4, linewidth=1.5)[0] for _ in range(n_bodies)])
        time_texts.append(ax.text(0.02, 0.98, '', transform=ax.transAxes,
                                  verticalalignment='top', fontsize=10))

    plt.tight_layout()

    def init():
        for scatter in scatters:
            scatter.set_offsets(np.empty((0, 2)))
        for model_trails in trails:
            for trail in model_trails:
                trail.set_data([], [])
        for time_text in time_texts:
            time_text.set_text('')
        return scatters + [t for mt in trails for t in mt] + time_texts

    def update(frame):
        for i, (name, pos) in enumerate(positions_dict.items()):
            scatters[i].set_offsets(pos[frame])
            for body_idx in range(n_bodies):
                trail_data[name][body_idx].append(pos[frame, body_idx].copy())
                if len(trail_data[name][body_idx]) > trail_length:
                    trail_data[name][body_idx].pop(0)
                if len(trail_data[name][body_idx]) > 1:
                    trail_array = np.array(trail_data[name][body_idx])
                    trails[i][body_idx].set_data(trail_array[:, 0], trail_array[:, 1])
            time_texts[i].set_text(f'Time: {frame * dt:.2f}s')
        return scatters + [t for mt in trails for t in mt] + time_texts

    anim = FuncAnimation(fig, update, init_func=init, frames=n_steps, interval=20, blit=True)

    try:
        writer = FFMpegWriter(fps=30, bitrate=2000)
        anim.save(video_name, writer=writer, dpi=100)
        print(f"Saved video: {video_name}")
    except Exception as e:
        print(f"Could not save MP4 (ffmpeg not available): {e}")
        print("Saving as GIF instead...")
        gif_name = str(Path(video_name).with_suffix('.gif'))
        anim.save(gif_name, writer='pillow', fps=30)
        print(f"Saved {gif_name}")

    plt.close()


def visualize_kan_splines(
    model,
    save_dir=None,
    msg_sample: Optional[torch.Tensor] = None,
    node_sample: Optional[torch.Tensor] = None,
):
    """Render smooth per-edge activation curves for message and node subnets."""
    save_dir = Path(save_dir).resolve() if save_dir else Path("visualizations/splines").resolve()
    save_dir.mkdir(parents=True, exist_ok=True)

    def _plot_subnet(subnet, sample, tag):
        tag_dir = save_dir / tag
        tag_dir.mkdir(parents=True, exist_ok=True)

        was_training = subnet.training
        subnet.eval()
        try:
            # Optional: seed subnet state from representative data.
            if sample is not None:
                with torch.no_grad():
                    subnet(sample[:4000])

            for layer_idx, layer in enumerate(subnet.act_fun):
                in_dim = int(layer.in_dim)
                out_dim = int(layer.out_dim)
                k = int(layer.k)
                device = layer.grid.device

                for in_idx in range(in_dim):
                    x_min = float(layer.grid[in_idx, k].item())
                    x_max = float(layer.grid[in_idx, -k - 1].item())
                    if not np.isfinite(x_min) or not np.isfinite(x_max) or x_max <= x_min:
                        continue

                    x_eval = torch.linspace(x_min, x_max, steps=800, device=device)
                    x_full = torch.zeros(x_eval.shape[0], in_dim, device=device)
                    x_full[:, in_idx] = x_eval

                    with torch.no_grad():
                        _, _, spline_postacts, _ = layer(x_full)

                    x_np = x_eval.detach().cpu().numpy()
                    for out_idx in range(out_dim):
                        y_np = spline_postacts[:, out_idx, in_idx].detach().cpu().numpy()
                        fig, ax = plt.subplots(figsize=(2.0, 2.0))
                        ax.plot(x_np, y_np, color="black", linewidth=2.5)
                        ax.set_xticks([])
                        ax.set_yticks([])
                        for spine in ax.spines.values():
                            spine.set_linewidth(2.0)
                            spine.set_color("black")
                        fig.tight_layout(pad=0.05)
                        fig.savefig(
                            tag_dir / f"sp_{layer_idx}_{in_idx}_{out_idx}.png",
                            bbox_inches="tight",
                            dpi=400,
                        )
                        plt.close(fig)
        finally:
            subnet.train(was_training)

    _plot_subnet(model.msg_kan, msg_sample, "message")
    _plot_subnet(model.node_kan, node_sample, "node")


def visualize_kan_network(model: 'OrdinaryGraphKAN',
                          data_sample: torch.Tensor,
                          network: str = 'msg',
                          save_path: str | Path | None = None) -> None:
    """
    Visualize msg or node KAN sub-network using pykan's native model.plot().

    Args:
        model:       Loaded OrdinaryGraphKAN instance
        data_sample: A representative input tensor.
                     For msg network:  shape (N, 2*n_f)
                     For node network: shape (N, n_f + msg_dim)
        network:     'msg' or 'node'
        save_path:   If given, save the figure here (e.g. 'kan_msg.png')
    """
    import gc

    # ── Select sub-network ────────────────────────────────────────
    info = _extract_graphkan_subnet_info(model, network)
    layers    = info['layers']
    src_kan   = info['subnet']
    width     = info['width']
    mult_arity = info['mult_arity']
    mult_nodes = info['mult_nodes']

    grid_size = model.grid_size
    k         = model.spline_order
    n_f       = model.n_f
    ndim      = model.ndim
    msg_dim   = model.msg_dim

    # Clamp sample size; use evenly spaced rows to preserve representativeness.
    max_plot_samples = 4000
    if data_sample.shape[0] > max_plot_samples:
        idx = torch.linspace(0, data_sample.shape[0] - 1, steps=max_plot_samples).long()
        data_sample = data_sample[idx]

    print(f"\nPreparing trained '{network}' KAN for plotting (no cloning):")
    print(f"  width={width}, mult_nodes={mult_nodes}, mult_arity={mult_arity}, grid={grid_size}, k={k}, n_samples={len(data_sample)}")

    # ── Populate activations in the trained subnet ───────────────
    src_kan.eval()
    with torch.no_grad():
        src_kan(data_sample)

    def _dense_plot_cache(subnet, points: int = 800) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        """Build dense x/y traces for each spline edge used by pykan.plot()."""
        dense_acts: list[torch.Tensor] = []
        dense_spline_postacts: list[torch.Tensor] = []

        for layer in subnet.act_fun:
            in_dim = int(layer.in_dim)
            out_dim = int(layer.out_dim)
            spline_k = int(layer.k)
            device = layer.grid.device

            acts_layer = torch.zeros(points, in_dim, device=device)
            spline_layer = torch.zeros(points, out_dim, in_dim, device=device)

            for in_idx in range(in_dim):
                x_min = float(layer.grid[in_idx, spline_k].item())
                x_max = float(layer.grid[in_idx, -spline_k - 1].item())
                if not np.isfinite(x_min) or not np.isfinite(x_max) or x_max <= x_min:
                    x_min, x_max = -1.0, 1.0

                x_eval = torch.linspace(x_min, x_max, steps=points, device=device)
                x_full = torch.zeros(points, in_dim, device=device)
                x_full[:, in_idx] = x_eval

                with torch.no_grad():
                    _, _, spline_postacts, _ = layer(x_full)

                acts_layer[:, in_idx] = x_eval
                spline_layer[:, :, in_idx] = spline_postacts[:, :, in_idx]

            dense_acts.append(acts_layer)
            dense_spline_postacts.append(spline_layer)

        return dense_acts, dense_spline_postacts

    # ── Variable names ─────────────────────────────────────────
    base_vars = ['x', 'y', 'vx', 'vy', 'm'][:n_f]
    if network == 'msg':
        in_vars  = [f'{v}_i' for v in base_vars] + [f'{v}_j' for v in base_vars]
        out_vars = [f'msg{i}' for i in range(msg_dim)]
    else:
        in_vars  = base_vars + [f'msg{i}' for i in range(msg_dim)]
        out_vars = ['ax', 'ay'] if ndim == 2 else [f'a{i}' for i in range(ndim)]

    arch_str = '×'.join(str(w) for w in width)
    title    = (f"Graph-KAN — {'Message' if network == 'msg' else 'Node Update'} "
                f"Network  [{arch_str}]")

    print(f"  Plotting '{title}' via pykan.plot()...")

    # pykan.plot writes to folder/figures.png; direct it to a temp asset folder and rename.
    folder = None
    asset_dir = None
    if save_path:
        save_path = Path(save_path).resolve()
        save_path.parent.mkdir(parents=True, exist_ok=True)
        asset_dir = save_path.parent / f"{save_path.stem}_assets"
        asset_dir.mkdir(parents=True, exist_ok=True)
        folder = str(asset_dir)

    original_acts = getattr(src_kan, "acts", None)
    original_spline_postacts = getattr(src_kan, "spline_postacts", None)
    try:
        dense_acts, dense_spline_postacts = _dense_plot_cache(src_kan, points=800)
        if isinstance(original_acts, list) and len(original_acts) >= len(dense_acts) + 1:
            plot_acts = list(original_acts)
            for layer_idx in range(len(dense_acts)):
                plot_acts[layer_idx] = dense_acts[layer_idx]
        else:
            # Fallback to expected depth+1 shape if pykan cache layout differs.
            plot_acts = list(dense_acts)
            plot_acts.append(dense_acts[-1][:, :int(width[-1][0] if isinstance(width[-1], list) else width[-1])])

        src_kan.acts = plot_acts
        src_kan.spline_postacts = dense_spline_postacts

        src_kan.plot(
            in_vars=in_vars,
            out_vars=out_vars,
            title=title,
            beta=120,            # match pykan doc recommendation for network overview
            scale=2,
            varscale=0.2,
            metric="forward_n",  # use forward scales to skip attribute() for high-arity mult
            folder=folder,
        )
    finally:
        src_kan.acts = original_acts
        if original_spline_postacts is not None:
            src_kan.spline_postacts = original_spline_postacts

    # Optional inset-style scaling: enlarge the smallest-width panels (e.g., the activations grid)
    fig = plt.gcf()
    axes = fig.get_axes()
    if axes:
        widths = sorted(set(ax.get_position().width for ax in axes))
        if widths:
            min_w = widths[0]
            inset_scale = 3
            for ax in axes:
                pos = ax.get_position()
                # print(pos.width)
                if abs(pos.width - min_w) < 1e-6:
                    cx = pos.x0 + pos.width / 2
                    cy = pos.y0 + pos.height / 2
                    ax.set_position([
                        cx - pos.width * inset_scale / 2,
                        cy - pos.height * inset_scale / 2,
                        pos.width * inset_scale,
                        pos.height * inset_scale,
                    ])

    if save_path:
        if fig.get_axes():
            # Save the actual network visualization figure shown by pykan.plot().
            fig.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f"  Saved network overview: {save_path}")
        else:
            print("  Warning: pykan.plot returned no visible axes to save.")

        if asset_dir:
            generated_images = sorted(
                [p for p in asset_dir.rglob("*")
                 if p.is_file() and p.suffix.lower() in {".png", ".pdf", ".svg", ".jpg", ".jpeg"}]
            )
            if generated_images:
                print(f"  Auxiliary pykan assets retained at: {asset_dir}")
            else:
                shutil.rmtree(asset_dir, ignore_errors=True)
    else:
        plt.show()

    plt.close(fig)

    gc.collect()
    torch.cuda.empty_cache()
    
def visualize_symbolic_expressions(
    kan_model,
    x_nodes: torch.Tensor,
    output_dir: str | Path,
    lib: list[str] | None = None,
    threshold: float = 0.85,
    max_batches: int = 10,
    edge_index: Optional[torch.Tensor] = None,
) -> dict:
    """
    Run symbolic regression on a trained GraphKAN model, print results,
    and save to JSON.

    Parameters
    ----------
    kan_model : OrdinaryGraphKAN
        Trained model
    x_nodes : torch.Tensor
        Representative node features, shape (n_nodes, n_f)
    output_dir : str or Path
        Directory to save symbolic_regression.json
    lib : list of str, optional
        Candidate symbolic functions. Defaults to physics-relevant library.
    threshold : float
        Minimum R² to report a match
    max_batches : int
        Number of batches to run forward pass over
    edge_index : torch.Tensor, optional
        Graph connectivity to pair with ``x_nodes``. Defaults to
        ``kan_model.edge_index``.

    Returns
    -------
    dict
        Raw suggestions from suggest_symbolic()
    """
    import json

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if lib is None:
        lib = list(getattr(kan_model, 'DEFAULT_SYMBOLIC_LIBRARY', (
            'x', 'x^2', 'x^3', '1/x', '1/x^2', '1/x^3', 'sqrt', 'log', 'abs', 'exp'
        )))

    base_edge_index = edge_index if edge_index is not None else kan_model.edge_index
    base_edge_index = base_edge_index.to(dtype=torch.long)

    n_nodes_single = int(base_edge_index.max().item()) + 1
    requested_batches = max(1, int(max_batches))
    symbolic_batches = 1

    # x_nodes typically contains many tiled graphs from warmup; keep symbolic
    # fitting bounded by selecting up to max_batches individual graphs.
    if n_nodes_single > 0 and int(x_nodes.shape[0]) % n_nodes_single == 0:
        total_graphs = int(x_nodes.shape[0]) // n_nodes_single
        graph_count = max(1, min(total_graphs, requested_batches))
        graph_indices = _sample_indices(total_graphs, graph_count)

        sym_graphs = []
        for g_idx in graph_indices:
            start = int(g_idx) * n_nodes_single
            stop = start + n_nodes_single
            sym_graphs.append(Data(x=x_nodes[start:stop], edge_index=base_edge_index))

        sym_loader = PyGLoader(sym_graphs, batch_size=1, shuffle=False)
        symbolic_batches = graph_count
        print(
            "  Symbolic sample set: "
            f"graphs={graph_count}/{total_graphs}, "
            f"nodes={graph_count * n_nodes_single}"
        )
    else:
        sym_graph = Data(x=x_nodes, edge_index=base_edge_index)
        sym_loader = PyGLoader([sym_graph], batch_size=1, shuffle=False)
        print(
            "  Symbolic sample set: using fallback single graph with "
            f"{int(x_nodes.shape[0])} nodes"
        )

    suggestions = kan_model.suggest_symbolic(
        sym_loader,
        device=next(kan_model.parameters()).device,
        lib=lib,
        max_batches=symbolic_batches,
        threshold=threshold,
        fit_affine_after_select=True,
    )

    kan_model.print_symbolic_suggestions(
        suggestions,
        threshold=threshold,
        max_edges_per_layer=25,
    )

    msg_edges = sum(len(edges) for edges in suggestions.get('msg_layers', {}).values())
    node_edges = sum(len(edges) for edges in suggestions.get('node_layers', {}).values())
    print(f"  Symbolic edge coverage: msg={msg_edges}, node={node_edges}")

    if hasattr(kan_model, 'serialize_symbolic_suggestions'):
        serializable = kan_model.serialize_symbolic_suggestions(
            suggestions,
            threshold=threshold,
            lib=lib,
        )
    else:
        # Fallback for older checkpoints/models lacking the helper method.
        serializable = {
            'metadata': {
                'threshold': float(threshold),
                'library': list(lib),
            },
            'msg_layers': {},
            'node_layers': {},
        }

    save_path = output_dir / 'symbolic_regression.json'
    with open(save_path, 'w') as f:
        json.dump(serializable, f, indent=2)
    print(f"  Saved: {save_path}")

    return suggestions


def main(
    yaml_params: Optional[dict] = None,
    output_dir: Optional[str] = None,
    checkpoint_dir: Optional[str] = None,
    data_file: Optional[str] = None,
):
    args = parse_args([] if yaml_params is not None else None)

    if yaml_params is not None:
        args.output_dir        = output_dir     or args.output_dir
        args.checkpoint_dir    = checkpoint_dir or args.checkpoint_dir
        args.data_file         = data_file      or args.data_file
        args.input_features    = yaml_params.get("input_features",    args.input_features)
        args.include_velocity  = yaml_params.get("include_velocity", args.include_velocity)
        args.save_video        = yaml_params.get("save_video",        args.save_video)
        args.plot_trajectories = yaml_params.get("plot_trajectories", args.plot_trajectories)
        args.plot_splines      = yaml_params.get("plot_splines",      args.plot_splines)
        args.prune_kan         = yaml_params.get("prune_kan",         args.prune_kan)
        args.prune_edge_threshold = yaml_params.get(
            "prune_edge_threshold", args.prune_edge_threshold
        )
        args.prune_node_threshold = yaml_params.get(
            "prune_node_threshold", args.prune_node_threshold
        )
        args.warmup_graphs = yaml_params.get("warmup_graphs", args.warmup_graphs)
        args.symbolic = yaml_params.get("symbolic", args.symbolic)

    args.input_features = _coerce_feature_spec_arg(args.input_features)

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Model Comparison Visualization")
    print("=" * 60)

    # Load models
    print("\nLoading trained models...")
    kan_model, _ = ModelLoader(OrdinaryGraphKAN, f'{args.checkpoint_dir}/graph_kan.pt').load()
    gnn_ckpt_path = Path(args.checkpoint_dir) / 'baseline_gnn.pt'
    gnn_model = None
    if gnn_ckpt_path.exists():
        gnn_model, _ = ModelLoader(OGN, str(gnn_ckpt_path)).load()
    else:
        print(f"Baseline checkpoint not found at {gnn_ckpt_path}; skipping baseline visualization.")
    print(f"Graph-KAN msg_width={getattr(kan_model, 'msg_width', 'N/A')}, node_width={getattr(kan_model, 'node_width', 'N/A')}")
    print(f"Graph-KAN msg_mult_nodes={getattr(kan_model, 'msg_mult_nodes', 0)}, node_mult_nodes={getattr(kan_model, 'node_mult_nodes', 0)}")

    # Load initial conditions
    print("\nLoading initial conditions...")
    data       = np.load(args.data_file)
    idx        = 1
    positions  = data['positions']
    velocities = data['velocities']
    if positions.ndim == 4:
        pos0 = torch.from_numpy(positions[idx, 0]).float()
        vel0 = torch.from_numpy(velocities[idx, 0]).float()
    elif positions.ndim == 3:
        if 'traj_offsets' in data.files:
            offsets = np.asarray(data['traj_offsets'], dtype=np.int64)
            n_traj = max(1, offsets.size - 1)
            traj_idx = min(idx, n_traj - 1)
            frame_idx = int(offsets[traj_idx])
        else:
            frame_idx = min(idx, positions.shape[0] - 1)
        pos0 = torch.from_numpy(positions[frame_idx]).float()
        vel0 = torch.from_numpy(velocities[frame_idx]).float()
    else:
        raise ValueError(
            f"Unsupported positions shape {positions.shape}; expected 4D or 3D arrays."
        )
    masses     = torch.from_numpy(data['masses']).float()
    edge_index = kan_model.edge_index
    force_name, force_fn, force_kwargs = _load_force_config(data)
    print(f"Using rollout force function from data: {force_name} with kwargs={force_kwargs}")

    ckpt_feature_spec = getattr(kan_model, "input_feature_spec", None)
    if args.input_features is not None:
        raw_feature_spec = args.input_features
    elif ckpt_feature_spec is not None:
        raw_feature_spec = ckpt_feature_spec
    else:
        raw_feature_spec = _infer_legacy_feature_spec_from_n_f(
            n_f=int(getattr(kan_model, "n_f", -1)),
            dim=int(pos0.shape[1]),
        )

    feature_spec = normalize_feature_spec(raw_feature_spec)

    if args.include_velocity is not None:
        expected_include_velocity = bool(args.include_velocity)
        spec_include_velocity = "vel" in feature_spec["include"]
        if expected_include_velocity != spec_include_velocity:
            raise ValueError(
                "include_velocity override conflicts with resolved feature spec: "
                f"override={expected_include_velocity}, spec_include={feature_spec['include']}"
            )

    expected_n_f = node_feature_dim(int(pos0.shape[1]), feature_spec)
    if int(getattr(kan_model, "n_f", -1)) != expected_n_f:
        raise ValueError(
            "Resolved feature spec does not match checkpoint input width: "
            f"spec_n_f={expected_n_f}, checkpoint_n_f={kan_model.n_f}, "
            f"include={feature_spec['include']}, augment={feature_spec['augment']}"
        )

    print(
        "Node input layout: "
        f"include={feature_spec['include']}, augment={feature_spec['augment']}"
    )

    rep_samples = _build_representative_samples(
        positions=positions,
        velocities=velocities,
        masses=masses,
        edge_index=edge_index,
        feature_spec=feature_spec,
        max_graphs=max(1, int(args.warmup_graphs)),
    )
    print(
        "Warmup samples: "
        f"graphs={rep_samples['n_graphs']}, "
        f"nodes={rep_samples['n_node_samples']}, "
        f"messages={rep_samples['n_message_samples']}"
    )

    # Single warmup forward: seeds pykan caches reused across prune/plots/symbolic steps.
    with torch.no_grad():
        _ = kan_model(rep_samples['x_nodes'], rep_samples['edge_index'])

    if args.prune_kan:
        print("Applying Graph-KAN pruning...")
        prune_summary = kan_model.prune_subnets(
            edge_threshold=_threshold_or_none(args.prune_edge_threshold),
            node_threshold=_threshold_or_none(args.prune_node_threshold),
        )
        print(
            "Graph-KAN pruned widths: "
            f"msg={prune_summary['msg_width']}, node={prune_summary['node_width']}"
        )

    print("Models loaded successfully")

    # Ground truth rollout
    print("Generating ground truth...")
    sim       = NBodySimulator(masses.numpy(), force_fn=force_fn, **force_kwargs)
    gt_result = sim.simulate(pos0.numpy(), vel0.numpy(), t_end=args.rollout_time, dt=args.dt, save_every=5)
    gt_pos    = gt_result['positions']

    # Model rollouts
    n_steps = int(args.rollout_time / args.dt)
    print(f"Rolling out Graph-KAN ({n_steps} steps)...")
    kan_pos, _ = rollout(
        kan_model,
        pos0,
        vel0,
        masses,
        args.dt,
        n_steps,
        edge_index,
        feature_spec=feature_spec,
    )

    gnn_pos = None
    if gnn_model is not None:
        print(f"Rolling out Baseline GNN ({n_steps} steps)...")
        gnn_pos, _ = rollout(
            gnn_model,
            pos0,
            vel0,
            masses,
            args.dt,
            n_steps,
            edge_index,
            feature_spec=feature_spec,
        )

    # Animate (sampled to match ground truth cadence)
    if (args.save_video):
        print("\n" + "=" * 60)
        print("Creating Animated Comparison")
        print("=" * 60)
        animate_rollout(
            ({
                'Ground Truth': gt_pos,
                'Graph-KAN':    kan_pos[::5],
                'Baseline GNN': gnn_pos[::5],
            } if gnn_pos is not None else {
                'Ground Truth': gt_pos,
                'Graph-KAN': kan_pos[::5],
            }),
            dt=args.dt * 5,
            video_name=f'{args.output_dir}/rollout_comparison.mp4',
        )

    # Spline visualizations
    if args.plot_splines:
        print("\n" + "=" * 60)
        print("Graph-KAN Spline Analysis")
        print("=" * 60)
        msg_sample_batch = getattr(kan_model.msg_kan, 'cache_data', None)
        node_sample_batch = getattr(kan_model.node_kan, 'cache_data', None)
        if msg_sample_batch is None or node_sample_batch is None:
            raise RuntimeError("Expected cached subnet inputs after warmup forward.")

        visualize_kan_splines(
            kan_model,
            save_dir=f"{args.output_dir}/splines",
            msg_sample=msg_sample_batch,
            node_sample=node_sample_batch,
        )

        print("\n" + "=" * 60)
        print("Graph-KAN Native Network Visualization")
        print("=" * 60)
        visualize_kan_network(kan_model, msg_sample_batch,
                                network='msg',
                                save_path=f'{args.output_dir}/kan_msg_network.png')
        visualize_kan_network(kan_model, node_sample_batch,
                                network='node',
                                save_path=f'{args.output_dir}/kan_node_network.png')

        # ── Symbolic regression — only once, after network viz ────
        if args.symbolic:
            print("\n" + "=" * 60)
            print("Symbolic Regression Analysis")
            print("=" * 60)
            visualize_symbolic_expressions(
                kan_model,
                x_nodes=rep_samples['x_nodes'],
                output_dir=args.output_dir,
                threshold=0.85,
                max_batches=1,
                edge_index=edge_index,
            )
        else:
            print("Skipping symbolic regression (disabled by --no-symbolic).")

    print("\n" + "=" * 60)
    print("Done! Generated files:")
    if args.save_video:
        print("  - rollout_comparison.mp4 (or .gif)")
    if args.plot_splines:
        print("  - splines/*.png")
        print("  - kan_msg_network.png")
        print("  - kan_node_network.png")
        if args.symbolic:
            print("  - symbolic_regression.json")
    print("=" * 60)


if __name__ == "__main__":
    main()
