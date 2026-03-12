"""Visualize trained models: rollout comparison and spline analysis."""

import argparse
import tempfile
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch_geometric.data import Data
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
from kan.spline import coef2curve

from nbody_gkan.models import OrdinaryGraphKAN, OGN
from nbody_gkan.nbody import NBodySimulator
from nbody_gkan.models.model_loader import ModelLoader


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


def visualize_kan_splines(model, var_names=('x', 'y', 'vx', 'vy', 'm'), save_dir=None):
    """Visualize learned spline activations in KAN layers."""
    print("\nVisualizing Graph-KAN splines...")

    save_dir = Path(save_dir)
    layer_groups = [("msg", model.msg_layers), ("node", model.node_layers)]

    for tag, layers in layer_groups:
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        axes = axes.flatten()

        for layer_idx, layer in enumerate(layers):
            ax = axes[layer_idx]
            x_eval = torch.linspace(-3, 3, 200)

            in_dim  = min(layer.in_dim,  5)
            out_dim = min(layer.out_dim, 3)

            for i in range(in_dim):
                for j in range(out_dim):
                    x_full = torch.zeros(200, layer.in_dim)
                    x_full[:, i] = x_eval

                    with torch.no_grad():
                        spline_out = coef2curve(x_full, layer.grid, layer.coef, layer.k)
                        y = spline_out[:, i, j] * layer.scale_sp[i, j].item()

                    label = f'{var_names[i]}→out{j}' if i < len(var_names) else f'in{i}→out{j}'
                    ax.plot(x_eval.numpy(), y.numpy(), alpha=0.6, label=label)

            layer_label = "Message" if tag == "msg" else "Node"
            ax.set_xlabel('Input value')
            ax.set_ylabel('Spline output')
            ax.set_title(f'{layer_label} Layer {layer_idx}')
            ax.grid(True, alpha=0.3)
            if layer_idx == 0:
                ax.legend(fontsize=8)

        plt.tight_layout()
        fname = f"kan_splines_{tag}.png"
        plt.savefig(save_dir / fname, dpi=150)
        print(f"Saved {fname}")
        plt.close()


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
    import tempfile
    from kan import KAN

    # ── Select sub-network ────────────────────────────────────────
    layers    = model.msg_layers if network == 'msg' else model.node_layers
    grid_size = model.grid_size
    k         = model.spline_order
    n_f       = model.n_f
    ndim      = model.ndim
    msg_dim   = model.msg_dim

    # Clamp sample size — pykan stores all activations, keep it small
    data_sample = data_sample[:50]

    # Infer width from stacked KANLayers — works for any depth
    width = [int(layers[0].in_dim)] + [int(l.out_dim) for l in layers]

    print(f"\nBuilding standalone KAN for '{network}' network:")
    print(f"  width={width}, grid={grid_size}, k={k}, n_samples={len(data_sample)}")

    with tempfile.TemporaryDirectory() as tmpdir:
        kan = KAN(width=width, grid=grid_size, k=k,
                  seed=0, auto_save=False, ckpt_path=tmpdir)

        # ── Transplant weights ─────────────────────────────────────
        for src_layer, dst_layer in zip(layers, kan.act_fun):
            dst_layer.coef.data.copy_(src_layer.coef.data)
            dst_layer.grid.data.copy_(src_layer.grid.data)
            dst_layer.scale_sp.data.copy_(src_layer.scale_sp.data)
            if hasattr(src_layer, 'scale_base') and hasattr(dst_layer, 'scale_base'):
                dst_layer.scale_base.data.copy_(src_layer.scale_base.data)

        # ── Forward pass to populate activations ──────────────────
        kan.eval()
        with torch.no_grad():
            kan(data_sample)

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

        print(f"  Plotting '{title}'...")
        kan.plot(in_vars=in_vars, out_vars=out_vars,
                 title=title, beta=3, scale=2, varscale=0.2)

        fig = plt.gcf()
        w, h = fig.get_size_inches()
        fig.set_size_inches(w * 8, h * 4)

        INSET_SCALE = 10
        for ax in fig.get_axes():
            pos = ax.get_position()
            if pos.width < 0.005:
                cx = pos.x0 + pos.width  / 2
                cy = pos.y0 + pos.height / 2
                ax.set_position([
                    cx - pos.width  * INSET_SCALE / 2,
                    cy - pos.height * INSET_SCALE / 2,
                    pos.width  * INSET_SCALE,
                    pos.height * INSET_SCALE,
                ])

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"  Saved: {save_path}")
        else:
            plt.show()

        plt.close()

        # ── Explicit cleanup — pykan holds large activation buffers ─
        for attr in ('acts', 'acts_scale', 'spline_preacts',
                     'spline_postacts', 'spline_postacts_symb'):
            try:
                delattr(kan, attr)
            except AttributeError:
                pass

    gc.collect()
    torch.cuda.empty_cache()

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
    kan_model, kan_ckpt = ModelLoader(OrdinaryGraphKAN, f'{args.checkpoint_dir}/graph_kan.pt').load()
    gnn_model, _        = ModelLoader(OGN             , f'{args.checkpoint_dir}/baseline_gnn.pt').load()
    print("Models loaded successfully")

    # Load initial conditions
    print("\nLoading initial conditions...")
    data       = np.load(args.data_file)
    idx        = 2
    pos0       = torch.from_numpy(data['positions'][idx, 0]).float()
    vel0       = torch.from_numpy(data['velocities'][idx, 0]).float()
    masses     = torch.from_numpy(data['masses']).float()
    edge_index = kan_ckpt['edge_index']

    # Ground truth rollout
    print("Generating ground truth...")
    sim       = NBodySimulator(masses.numpy())
    gt_result = sim.simulate(pos0.numpy(), vel0.numpy(), t_end=args.rollout_time, dt=args.dt, save_every=5)
    gt_pos    = gt_result['positions']

    # Model rollouts
    n_steps = int(args.rollout_time / args.dt)
    print(f"Rolling out Graph-KAN ({n_steps} steps)...")
    kan_pos, _ = rollout(kan_model, pos0, vel0, masses, args.dt, n_steps, edge_index)

    print(f"Rolling out Baseline GNN ({n_steps} steps)...")
    gnn_pos, _ = rollout(gnn_model, pos0, vel0, masses, args.dt, n_steps, edge_index)

    # Animate (sampled to match ground truth cadence)
    print("\n" + "=" * 60)
    print("Creating Animated Comparison")
    print("=" * 60)
    animate_rollout(
        {
            'Ground Truth': gt_pos,
            'Graph-KAN':    kan_pos[::5],
            'Baseline GNN': gnn_pos[::5],
        },
        dt=args.dt * 5,
        video_name=f'{args.output_dir}/rollout_comparison.mp4',
    )

    # Spline visualizations
    if args.plot_splines:
        print("\n" + "=" * 60)
        print("Graph-KAN Spline Analysis")
        print("=" * 60)
        visualize_kan_splines(kan_model, save_dir=args.output_dir)

        # Build representative inputs for activation population
        with torch.no_grad():
            mass_expanded = masses.unsqueeze(1)
            x_nodes   = torch.cat([pos0, vel0, mass_expanded], dim=1)       # (n_bodies, n_f)
            src, dst  = kan_model.edge_index[0], kan_model.edge_index[1]    # ← from model, not external
            msg_sample = torch.cat([x_nodes[src], x_nodes[dst]], dim=1)     # (n_edges, 2*n_f)

            msg_sample_batch = (msg_sample.repeat(50, 1)
                                + torch.randn(msg_sample.shape[0] * 50,
                                              msg_sample.shape[1]) * 0.5)

            node_sample_batch = torch.randn(200, kan_model.n_f + kan_model.msg_dim) * 0.5  # ← from model

        print("\n" + "=" * 60)
        print("Graph-KAN Native Network Visualization")
        print("=" * 60)
        visualize_kan_network(kan_model, msg_sample_batch,
                              network='msg',
                              save_path=f'{args.output_dir}/kan_msg_network.png')
        visualize_kan_network(kan_model, node_sample_batch,
                              network='node',
                              save_path=f'{args.output_dir}/kan_node_network.png')

    print("\n" + "=" * 60)
    print("Done! Generated files:")
    print("  - rollout_comparison.mp4 (or .gif)")
    print("  - kan_splines_msg.png")
    print("  - kan_splines_node.png")
    print("=" * 60)


if __name__ == "__main__":
    main()
