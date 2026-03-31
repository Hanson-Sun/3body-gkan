"""Visualize trained models: rollout comparison and spline analysis."""

import argparse
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

from nbody_gkan.models import OrdinaryGraphKAN, OGN
from nbody_gkan.nbody import NBodySimulator
from nbody_gkan.models.model_loader import ModelLoader


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
    parser.add_argument("--rollout_time",      type=float, default=5.0)
    parser.add_argument("--dt",                type=float, default=0.01)
    parser.add_argument("--save_video",        action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--plot_trajectories", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--plot_splines",      action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args(args)


# def load_model(checkpoint_path, model_class):
#     """Load model from checkpoint."""
#     ckpt = torch.load(checkpoint_path, map_location='cpu')

#     if model_class == OrdinaryGraphKAN:
#         model = model_class(
#             n_f=ckpt['n_features'],
#             msg_dim=ckpt['msg_dim'],
#             ndim=ckpt['dim'],
#             edge_index=ckpt['edge_index'],
#             hidden=ckpt['hidden'],
#             grid_size=ckpt['grid_size'],
#             spline_order=ckpt.get('spline_order', 3),
#             aggr="add",
#             hidden_layers=ckpt.get('hidden_layers', 0),
#         )
#     else:  # OGN
#         model = model_class(
#             n_f=ckpt['n_features'],
#             msg_dim=ckpt['msg_dim'],
#             ndim=ckpt['dim'],
#             edge_index=ckpt['edge_index'],
#             hidden=ckpt['hidden'],
#             aggr="add",
#         )

#     model.load_state_dict(ckpt['model_state'])
#     model.eval()
#     return model, ckpt

def rollout(model, pos0, vel0, masses, dt, n_steps, edge_index):
    """Rollout dynamics using leapfrog integration."""
    positions = [pos0.cpu().numpy()]
    velocities = [vel0.cpu().numpy()]

    pos = pos0.clone()
    vel = vel0.clone()

    with torch.no_grad():
        for _ in range(n_steps):
            mass_expanded = masses.unsqueeze(1)
            x = torch.cat([pos, vel, mass_expanded], dim=1)
            graph = Data(x=x, edge_index=edge_index)

            # Leapfrog step
            acc = model.just_derivative(graph, augment=False)
            vel_half = vel + 0.5 * dt * acc
            pos = pos + dt * vel_half

            # Recompute acceleration at new position
            x = torch.cat([pos, vel_half, mass_expanded], dim=1)
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


def visualize_kan_splines(model, save_dir=None):
    """Use pykan's native plot() to visualize learned activations."""
    import tempfile
    from kan import KAN

    save_dir = Path(save_dir).resolve() if save_dir else Path("visualizations/splines").resolve()
    save_dir.mkdir(parents=True, exist_ok=True)

    # Build small representative samples to populate activations
    n_f = model.n_f
    msg_dim = model.msg_dim
    msg_sample = torch.randn(64, 2 * n_f)
    node_sample = torch.randn(64, n_f + msg_dim)

    def _plot_subnet(subnet, width, sample, tag):
        safe_mult_arity = subnet.mult_arity
        tag_dir = save_dir / tag
        tag_dir.mkdir(parents=True, exist_ok=True)

        with tempfile.TemporaryDirectory() as tmpdir:
            # Clone weights into a fresh KAN so plot() runs without side-effects
            clone = KAN(width=width, grid=subnet.grid, k=subnet.k,
                       mult_arity=safe_mult_arity,
                       seed=0, auto_save=False, ckpt_path=tmpdir,
                       symbolic_enabled=False, save_act=True)

            # Copy parameters
            for src_layer, dst_layer in zip(subnet.act_fun, clone.act_fun):
                dst_layer.coef.data.copy_(src_layer.coef.data)
                dst_layer.grid.data.copy_(src_layer.grid.data)
                dst_layer.scale_sp.data.copy_(src_layer.scale_sp.data)
                if hasattr(src_layer, 'scale_base') and hasattr(dst_layer, 'scale_base'):
                    dst_layer.scale_base.data.copy_(src_layer.scale_base.data)
                if hasattr(src_layer, 'mask') and hasattr(dst_layer, 'mask'):
                    dst_layer.mask.data.copy_(src_layer.mask.data)

            for l in range(len(clone.node_scale)):
                clone.node_scale[l].data.copy_(subnet.node_scale[l].data)
                clone.node_bias[l].data.copy_(subnet.node_bias[l].data)
                clone.subnode_scale[l].data.copy_(subnet.subnode_scale[l].data)
                clone.subnode_bias[l].data.copy_(subnet.subnode_bias[l].data)

            # Populate activations, then plot
            clone.eval()
            with torch.no_grad():
                clone(sample)

            out_file = tag_dir / "kan_plot"
            clone.plot(
                folder=str(tag_dir),
                metric="forward_n",  # avoid backward attribution path on mult_arity-heavy nets
                title=f"Graph-KAN — {tag.title()} Network",
                in_vars=None,
                out_vars=None,
            )

            # pykan.plot saves figures into the provided folder; rename the default file if present
            fig_path = tag_dir / "figures.png"
            alt_figs = sorted(tag_dir.glob("figures*.png"))
            if fig_path.exists():
                fig_path.rename(out_file.with_suffix(".png"))
            elif alt_figs:
                alt_figs[0].rename(out_file.with_suffix(".png"))

    _plot_subnet(model.msg_kan, model.msg_width, msg_sample, "message")
    _plot_subnet(model.node_kan, model.node_width, node_sample, "node")


def visualize_kan_network(model: 'OrdinaryGraphKAN',
                          data_sample: torch.Tensor,
                          network: str = 'msg',
                          save_path: str | Path | None = None) -> None:
    """
    Visualize msg or node KAN sub-network using pykan's native model.plot().

    Args:
        model:       Loaded OrdinaryGraphKAN instance
        data_sample: A representative input tensor — keep small (~50 samples).
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

    # Clamp sample size — pykan stores all activations, keep it small
    data_sample = data_sample[:50]

    print(f"\nPreparing trained '{network}' KAN for plotting (no cloning):")
    print(f"  width={width}, mult_nodes={mult_nodes}, mult_arity={mult_arity}, grid={grid_size}, k={k}, n_samples={len(data_sample)}")

    # ── Populate activations in the trained subnet ───────────────
    src_kan.eval()
    with torch.no_grad():
        src_kan(data_sample)

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

    # src_kan = src_kan.prune()
    src_kan.plot(
        in_vars=in_vars,
        out_vars=out_vars,
        title=title,
        beta=100,            # match pykan doc recommendation for network overview
        scale=2,
        varscale=0.2,
        metric="forward_n",  # use forward scales to skip attribute() for high-arity mult
        folder=folder,
    )

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

    Returns
    -------
    dict
        Raw suggestions from suggest_symbolic()
    """
    import json

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if lib is None:
        lib = ['x', 'x^2', 'x^3', '1/x', '1/x^2',
               'sqrt(x)', 'log(x)', 'abs(x)', 'sin(x)', 'cos(x)', 'exp(x)']

    sym_graph  = Data(x=x_nodes, edge_index=kan_model.edge_index)
    sym_loader = PyGLoader([sym_graph] * max_batches, batch_size=1)

    suggestions = kan_model.suggest_symbolic(
        sym_loader,
        device=next(kan_model.parameters()).device,
        lib=lib,
        threshold=threshold,
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
        args.save_video        = yaml_params.get("save_video",        args.save_video)
        args.plot_trajectories = yaml_params.get("plot_trajectories", args.plot_trajectories)
        args.plot_splines      = yaml_params.get("plot_splines",      args.plot_splines)

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
    print("Models loaded successfully")

    # Load initial conditions
    print("\nLoading initial conditions...")
    data       = np.load(args.data_file)
    idx        = 1
    pos0       = torch.from_numpy(data['positions'][idx, 0]).float()
    vel0       = torch.from_numpy(data['velocities'][idx, 0]).float()
    masses     = torch.from_numpy(data['masses']).float()
    edge_index = kan_model.edge_index

    # Ground truth rollout
    print("Generating ground truth...")
    sim       = NBodySimulator(masses.numpy())
    gt_result = sim.simulate(pos0.numpy(), vel0.numpy(), t_end=args.rollout_time, dt=args.dt, save_every=5)
    gt_pos    = gt_result['positions']

    # Model rollouts
    n_steps = int(args.rollout_time / args.dt)
    print(f"Rolling out Graph-KAN ({n_steps} steps)...")
    kan_pos, _ = rollout(kan_model, pos0, vel0, masses, args.dt, n_steps, edge_index)

    gnn_pos = None
    if gnn_model is not None:
        print(f"Rolling out Baseline GNN ({n_steps} steps)...")
        gnn_pos, _ = rollout(gnn_model, pos0, vel0, masses, args.dt, n_steps, edge_index)

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
        visualize_kan_splines(kan_model, save_dir=f"{args.output_dir}/splines")

        # Build representative inputs for activation population
        with torch.no_grad():
            mass_expanded = masses.unsqueeze(1)
            x_nodes  = torch.cat([pos0, vel0, mass_expanded], dim=1)
            src, dst = kan_model.edge_index[0], kan_model.edge_index[1]
            msg_sample = torch.cat([x_nodes[src], x_nodes[dst]], dim=1)

            msg_sample_batch = (msg_sample.repeat(50, 1)
                                + torch.randn(msg_sample.shape[0] * 50,
                                                msg_sample.shape[1]) * 0.5)
            node_sample_batch = torch.randn(200, kan_model.n_f + kan_model.msg_dim) * 0.5

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
        print("\n" + "=" * 60)
        print("Symbolic Regression Analysis")
        print("=" * 60)
        visualize_symbolic_expressions(
            kan_model,
            x_nodes=x_nodes,
            output_dir=args.output_dir,
            threshold=0.85,
        )

    print("\n" + "=" * 60)
    print("Done! Generated files:")
    if args.save_video:
        print("  - rollout_comparison.mp4 (or .gif)")
    if args.plot_splines:
        print("  - splines/*.png")
        print("  - kan_msg_network.png")
        print("  - kan_node_network.png")
        print("  - symbolic_regression.json")
    print("=" * 60)


if __name__ == "__main__":
    main()
