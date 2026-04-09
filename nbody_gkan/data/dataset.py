"""
PyTorch Geometric dataset for N-body dynamics.

Generates graph data with:
- Node features: [pos_x, pos_y, vel_x, vel_y, mass] for 2D
- Edge index: fully connected graph
- Targets: accelerations from a configured force law
"""

import json
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data

from .. import nbody_force_fns


FORCE_FN_MAP = {
    "gravity": nbody_force_fns.gravity,
    "linear_gravity": nbody_force_fns.linear_gravity,
    "cubic_gravity": nbody_force_fns.cubic_gravity,
    "linear_spring": nbody_force_fns.linear_spring,
    "hooke_pairwise": nbody_force_fns.hooke_pairwise,
}


def get_edge_index(n: int) -> torch.Tensor:
    """
    Generate edge indices for a fully connected graph.

    Creates bidirectional edges between all pairs of nodes (excluding self-loops).

    Parameters
    ----------
    n : int
        Number of nodes

    Returns
    -------
    torch.Tensor
        Edge indices, shape (2, n_edges) where n_edges = n*(n-1)
    """
    # Adjacency matrix: 1 everywhere except diagonal
    adj = (np.ones((n, n)) - np.eye(n)).astype(int)
    edge_index = torch.from_numpy(np.array(np.where(adj)))
    return edge_index.long()


class NBodyDataset(Dataset):
    """
    PyTorch Geometric dataset for N-body dynamics.

    Loads trajectory data and yields graph data samples where:
    - Node features x: [pos, vel, mass] concatenated
    - Targets y: accelerations
    - edge_index: fully connected graph

    Parameters
    ----------
    data_path : str or Path
        Path to .npz file with trajectory data
        Expected keys: 'positions', 'velocities', 'masses'
        Optional metadata keys: 'force_name', 'force_kwargs'
    G : float, optional (default=1.0)
        Default gravity coefficient used when force metadata is absent
    softening : float, optional (default=1e-2)
        Default gravity softening used when force metadata is absent
    force_fn : str, optional
        Override force function name. If omitted, uses NPZ metadata when
        available, else falls back to 'gravity'.
    force_kwargs : dict, optional
        Override kwargs passed to the selected force function.
    """

    def __init__(
            self,
            data_path: str | Path,
            G: float = 1.0,
            softening: float = 1e-2,
            force_fn: Optional[str] = None,
            force_kwargs: Optional[dict] = None,
    ):
        self.data_path = Path(data_path)

        # Load data
        data = np.load(self.data_path)
        self.positions = data["positions"]  # (n_traj, T, n, dim)
        self.velocities = data["velocities"]  # (n_traj, T, n, dim)
        self.masses = data["masses"]  # (n,)

        self.G = G
        self.softening = softening

        # Force selection priority: explicit args -> file metadata -> gravity fallback.
        if force_fn is not None:
            self.force_name = str(force_fn)
        elif "force_name" in data.files:
            self.force_name = self._decode_scalar(data["force_name"])
        else:
            self.force_name = "gravity"

        if force_kwargs is not None:
            parsed_kwargs = dict(force_kwargs)
        elif "force_kwargs" in data.files:
            parsed_kwargs = self._decode_force_kwargs(data["force_kwargs"])
        else:
            parsed_kwargs = {}

        # Backward-compatible default for older files with no metadata.
        if not parsed_kwargs and self.force_name in {"gravity", "linear_gravity", "cubic_gravity", "hooke_pairwise"}:
            parsed_kwargs = {"G": self.G, "softening": self.softening}

        if self.force_name not in FORCE_FN_MAP:
            valid = ", ".join(sorted(FORCE_FN_MAP))
            raise ValueError(
                f"Unknown force function {self.force_name!r} in {self.data_path}. "
                f"Available: {valid}"
            )
        self.force_fn = FORCE_FN_MAP[self.force_name]
        self.force_kwargs = parsed_kwargs

        # Flatten trajectories: treat each timestep as a separate sample
        self.n_traj, self.T, self.n, self.dim = self.positions.shape

        # Reshape to (n_samples, n, dim)
        self.positions = self.positions.reshape(-1, self.n, self.dim)
        self.velocities = self.velocities.reshape(-1, self.n, self.dim)

        self.n_samples = self.positions.shape[0]

        # Precompute edge index (same for all samples)
        self.edge_index = get_edge_index(self.n)

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, idx: int) -> Data:
        """
        Get a single graph data sample.

        Returns
        -------
        torch_geometric.data.Data
            Graph with:
            - x: node features [pos, vel, mass] shape (n, 2*dim+1)
            - y: accelerations shape (n, dim)
            - edge_index: shape (2, n_edges)
        """
        pos = self.positions[idx]  # (n, dim)
        vel = self.velocities[idx]  # (n, dim)

        # Compute force-consistent targets on-the-fly.
        acc = self.force_fn(pos, self.masses, **self.force_kwargs)

        # Node features: [pos, vel, mass]
        # Expand mass to (n, 1) and concatenate
        masses_expanded = self.masses[:, np.newaxis]  # (n, 1)
        x = np.concatenate([pos, vel, masses_expanded], axis=1)  # (n, 2*dim+1)

        # Convert to tensors
        x = torch.from_numpy(x).float()
        y = torch.from_numpy(acc).float()

        return Data(x=x, y=y, edge_index=self.edge_index)

    @staticmethod
    def collate_fn(batch):
        """Custom collate function for batching graph data."""
        # PyG's Batch.from_data_list handles batching automatically
        from torch_geometric.data import Batch

        return Batch.from_data_list(batch)

    @staticmethod
    def _decode_scalar(value) -> str:
        """Decode NPZ scalar/string payloads into a Python string."""
        if isinstance(value, np.ndarray):
            if value.shape == ():
                value = value.item()
            elif value.size == 1:
                value = value.reshape(()).item()
        if isinstance(value, bytes):
            return value.decode("utf-8")
        return str(value)

    @classmethod
    def _decode_force_kwargs(cls, value) -> dict:
        """Decode serialized force kwargs metadata from NPZ."""
        raw = cls._decode_scalar(value)
        if not raw or raw == "None":
            return {}
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            return {}
        if isinstance(parsed, dict):
            return parsed
        return {}


def create_dataset_from_simulator(
        sim,
        n_trajectories: int,
        t_end: float,
        dt: float,
        save_every: int = 1,
        output_path: Optional[str | Path] = None,
        seed: int = 0,
        **ic_kwargs,
) -> Path:
    """
    Generate N-body trajectory data using the simulator and save to disk.

    Parameters
    ----------
    sim : NBodySimulator
        Simulator instance
    n_trajectories : int
        Number of trajectories to generate
    t_end : float
        End time for each trajectory
    dt : float
        Simulation timestep
    save_every : int, optional (default=1)
        Save every k-th step
    output_path : str or Path, optional
        Output file path (.npz). If None, uses "data/nbody_data.npz"
    seed : int
        Random seed for reproducibility
    **ic_kwargs
        Additional keyword arguments for initial conditions

    Returns
    -------
    Path
        Path to the saved data file
    """
    if output_path is None:
        output_path = Path("data/nbody_data.npz")
    else:
        output_path = Path(output_path)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Generate initial conditions
    pos_batch, vel_batch = sim.batch_initial_conditions(
        n_trajectories=n_trajectories, seed=seed, **ic_kwargs
    )

    # Simulate
    results = sim.simulate_batch(pos_batch, vel_batch, t_end, dt, save_every)

    # Stack results
    positions = np.stack([r["positions"] for r in results])
    velocities = np.stack([r["velocities"] for r in results])
    times = results[0]["times"]
    masses = sim.masses

    force_name = getattr(sim.force_fn, "__name__", "gravity")
    force_kwargs = json.dumps(sim.force_kwargs)

    # Save
    np.savez_compressed(
        output_path,
        positions=positions,
        velocities=velocities,
        times=times,
        masses=masses,
        force_name=force_name,
        force_kwargs=force_kwargs,
    )

    print(f"Saved {n_trajectories} trajectories to {output_path}")
    print(
        "Data shape: "
        f"positions {positions.shape}, "
        f"velocities {velocities.shape}"
    )

    return output_path
