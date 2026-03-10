"""Visualize trained models: rollout comparison and spline analysis."""

import numpy as np
import torch
from torch_geometric.data import Data
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
from kan.spline import coef2curve

from nbody_gkan.models import OrdinaryGraphKAN, OGN
from nbody_gkan.nbody import NBodySimulator


# Config
CHECKPOINT_DIR = "checkpoints/comparison"
DATA_FILE = "data/train.npz"
ROLLOUT_TIME = 5.0
DT = 0.01
SAVE_VIDEO = True


def load_model(checkpoint_path, model_class):
    """Load model from checkpoint."""
    ckpt = torch.load(checkpoint_path, map_location='cpu')

    if model_class == OrdinaryGraphKAN:
        model = model_class(
            n_f=ckpt['n_features'],
            msg_dim=ckpt['msg_dim'],
            ndim=ckpt['dim'],
            edge_index=ckpt['edge_index'],
            hidden=ckpt['hidden'],
            grid_size=ckpt['grid_size'],
            spline_order=3,
            aggr="add"
        )
    else:  # OGN
        model = model_class(
            n_f=ckpt['n_features'],
            msg_dim=ckpt['msg_dim'],
            ndim=ckpt['dim'],
            edge_index=ckpt['edge_index'],
            hidden=ckpt['hidden'],
            aggr="add"
        )

    model.load_state_dict(ckpt['model_state'])
    model.eval()

    return model, ckpt


def rollout(model, pos0, vel0, masses, dt, n_steps, edge_index):
    """Rollout dynamics using leapfrog integration."""
    n_bodies, dim = pos0.shape
    positions = [pos0.cpu().numpy()]
    velocities = [vel0.cpu().numpy()]

    pos = pos0.clone()
    vel = vel0.clone()

    with torch.no_grad():
        for _ in range(n_steps):
            # Build features
            mass_expanded = masses.unsqueeze(1)
            x = torch.cat([pos, vel, mass_expanded], dim=1)
            graph = Data(x=x, edge_index=edge_index)

            # Leapfrog step
            acc = model.just_derivative(graph, augment=False)
            vel_half = vel + 0.5 * dt * acc
            pos = pos + dt * vel_half

            # Recompute with new position
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
    fig, axes = plt.subplots(1, n_models, figsize=(5*n_models, 5))
    if n_models == 1:
        axes = [axes]

    # Get dimensions
    first_pos = list(positions_dict.values())[0]
    n_steps, n_bodies, _ = first_pos.shape

    # Compute global axis limits
    all_pos = np.concatenate(list(positions_dict.values()), axis=0)
    margin = 0.5
    x_min, x_max = all_pos[:, :, 0].min() - margin, all_pos[:, :, 0].max() + margin
    y_min, y_max = all_pos[:, :, 1].min() - margin, all_pos[:, :, 1].max() + margin

    # Initialize plots
    scatters = []
    trails = []
    trail_data = {name: [[] for _ in range(n_bodies)] for name in positions_dict.keys()}
    time_texts = []
    trail_length = 50

    for ax, name in zip(axes, positions_dict.keys()):
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_title(name)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)

        scatter = ax.scatter([], [], s=100, alpha=0.8)
        scatters.append(scatter)

        model_trails = [ax.plot([], [], '-', alpha=0.4, linewidth=1.5)[0] for _ in range(n_bodies)]
        trails.append(model_trails)

        time_text = ax.text(0.02, 0.98, '', transform=ax.transAxes,
                           verticalalignment='top', fontsize=10)
        time_texts.append(time_text)

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

    anim = FuncAnimation(fig, update, init_func=init, frames=n_steps,
                        interval=20, blit=True)

    try:
        writer = FFMpegWriter(fps=30, bitrate=2000)
        anim.save(video_name, writer=writer, dpi=100)
        print(f"Saved video: {video_name}")
    except Exception as e:
        print(f"Could not save MP4 (ffmpeg not available): {e}")
        print("Saving as GIF instead...")
        anim.save(video_name.replace('.mp4', '.gif'), writer='pillow', fps=30)
        print(f"Saved {video_name.replace('.mp4', '.gif')}")

    plt.close()


def visualize_kan_splines(model, var_names=['x', 'y', 'vx', 'vy', 'm']):
    """Visualize learned spline activations in KAN layers."""
    print("\nVisualizing Graph-KAN splines...")

    # Message layers
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()

    for layer_idx, layer in enumerate(model.msg_layers):
        ax = axes[layer_idx]
        x_eval = torch.linspace(-3, 3, 200)

        in_dim = min(layer.in_dim, 5)
        out_dim = min(layer.out_dim, 3)

        for i in range(in_dim):
            for j in range(out_dim):
                x_full = torch.zeros(200, layer.in_dim)
                x_full[:, i] = x_eval

                with torch.no_grad():
                    spline_out = coef2curve(x_full, layer.grid, layer.coef, layer.k)
                    y = spline_out[:, i, j] * layer.scale_sp[i, j].item()

                label = f'in{i}→out{j}'
                if i < len(var_names):
                    label = f'{var_names[i]}→out{j}'
                ax.plot(x_eval.numpy(), y.numpy(), alpha=0.6, label=label)

        ax.set_xlabel('Input value')
        ax.set_ylabel('Spline output')
        ax.set_title(f'Message Layer {layer_idx}')
        ax.grid(True, alpha=0.3)
        if layer_idx == 0:
            ax.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig('kan_splines_msg.png', dpi=150)
    print("Saved kan_splines_msg.png")
    plt.close()

    # Node layers
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()

    for layer_idx, layer in enumerate(model.node_layers):
        ax = axes[layer_idx]
        x_eval = torch.linspace(-3, 3, 200)

        in_dim = min(layer.in_dim, 5)
        out_dim = min(layer.out_dim, 3)

        for i in range(in_dim):
            for j in range(out_dim):
                x_full = torch.zeros(200, layer.in_dim)
                x_full[:, i] = x_eval

                with torch.no_grad():
                    spline_out = coef2curve(x_full, layer.grid, layer.coef, layer.k)
                    y = spline_out[:, i, j] * layer.scale_sp[i, j].item()

                ax.plot(x_eval.numpy(), y.numpy(), alpha=0.6, label=f'in{i}→out{j}')

        ax.set_xlabel('Input value')
        ax.set_ylabel('Spline output')
        ax.set_title(f'Node Layer {layer_idx}')
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('kan_splines_node.png', dpi=150)
    print("Saved kan_splines_node.png")
    plt.close()


def main():
    print("="*60)
    print("Model Comparison Visualization")
    print("="*60)

    # Load models
    print("\nLoading trained models...")
    kan_model, kan_ckpt = load_model(f'{CHECKPOINT_DIR}/graph_kan.pt', OrdinaryGraphKAN)
    gnn_model, gnn_ckpt = load_model(f'{CHECKPOINT_DIR}/baseline_gnn.pt', OGN)
    print("Models loaded successfully")

    # Load initial conditions
    print("\nLoading initial conditions...")
    data = np.load(DATA_FILE)
    pos0 = torch.from_numpy(data['positions'][0, 0]).float()
    vel0 = torch.from_numpy(data['velocities'][0, 0]).float()
    masses = torch.from_numpy(data['masses']).float()
    edge_index = kan_ckpt['edge_index']

    # Ground truth
    print("Generating ground truth...")
    sim = NBodySimulator(masses.numpy())
    gt_result = sim.simulate(pos0.numpy(), vel0.numpy(), t_end=ROLLOUT_TIME, dt=DT, save_every=5)
    gt_pos = gt_result['positions']

    # Model rollouts
    n_steps = int(ROLLOUT_TIME / DT)
    print(f"Rolling out Graph-KAN ({n_steps} steps)...")
    kan_pos, _ = rollout(kan_model, pos0, vel0, masses, DT, n_steps, edge_index)

    print(f"Rolling out Baseline GNN ({n_steps} steps)...")
    gnn_pos, _ = rollout(gnn_model, pos0, vel0, masses, DT, n_steps, edge_index)

    # Sample for animation
    kan_pos_sampled = kan_pos[::5]
    gnn_pos_sampled = gnn_pos[::5]

    # Animate
    print("\n" + "="*60)
    print("Creating Animated Comparison")
    print("="*60)
    animate_rollout({
        'Ground Truth': gt_pos,
        'Graph-KAN': kan_pos_sampled,
        'Baseline GNN': gnn_pos_sampled
    }, dt=DT*5, video_name='rollout_comparison.mp4')

    # Spline visualization
    print("\n" + "="*60)
    print("Graph-KAN Spline Analysis")
    print("="*60)
    visualize_kan_splines(kan_model, var_names=['x', 'y', 'vx', 'vy', 'm'])

    print("\n" + "="*60)
    print("Done! Generated files:")
    print("  - rollout_comparison.mp4 (or .gif)")
    print("  - kan_splines_msg.png")
    print("  - kan_splines_node.png")
    print("="*60)


if __name__ == "__main__":
    main()
