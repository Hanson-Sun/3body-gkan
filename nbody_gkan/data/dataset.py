"""
PyTorch Geometric dataset for N-body dynamics.

Generates graph data with:
- Node features: [pos, vel, mass] (default) or [pos, mass] when velocity is disabled
- Edge index: fully connected graph
- Targets: accelerations from a configured force law
"""

import json
from pathlib import Path
from typing import Any, Optional

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
    "nice_function": nbody_force_fns.nice_function,
}

DEFAULT_FEATURE_SPEC = {
    "include": ["pos", "vel", "mass"],
    "augment": [],
}

_VALID_BASE_FEATURES = {"pos", "vel", "mass"}


def normalize_feature_spec(
    feature_spec: Any = None,
    include_velocity: Optional[bool] = None,
) -> dict[str, list[str]]:
    """
    Normalize feature selection config.

    Parameters
    ----------
    feature_spec : dict | list[str] | None
        Supported forms:
        - None: defaults to include=[pos, vel, mass], augment=[]
        - list[str]: interpreted as include list
        - dict: {"include": [...]} (optional "augment" must be empty if present)
    include_velocity : bool, optional
        Backward-compatible alias used only when feature_spec is None.

    Returns
    -------
    dict[str, list[str]]
        Normalized spec with keys: include, augment.
    """
    if feature_spec is None:
        include = ["pos", "vel", "mass"] if include_velocity is None or include_velocity else ["pos", "mass"]
        augment = []
    elif isinstance(feature_spec, (list, tuple)):
        include = list(feature_spec)
        augment = []
    elif isinstance(feature_spec, dict):
        include = feature_spec.get("include", DEFAULT_FEATURE_SPEC["include"])
        augment = feature_spec.get("augment", feature_spec.get("augmentations", []))
    else:
        raise ValueError("feature_spec must be None, a list, or a dict with 'include'.")

    include = [str(v).strip() for v in include]
    if not include:
        raise ValueError("feature_spec.include must not be empty.")
    if augment:
        raise ValueError("feature_spec.augment is not supported.")

    unknown_include = set(include) - _VALID_BASE_FEATURES
    if unknown_include:
        raise ValueError(f"Unknown base feature(s): {sorted(unknown_include)}")

    return {
        "include": include,
        "augment": [],
    }


def node_feature_dim(dim: int, feature_spec: Any) -> int:
    """Return resulting node feature width for a given spatial dimension and feature spec."""
    if dim <= 0:
        raise ValueError(f"dim must be > 0, got {dim}.")

    spec = normalize_feature_spec(feature_spec)
    return sum(dim if name in {"pos", "vel"} else 1 for name in spec["include"])


def build_node_features_np(
    pos: np.ndarray,
    vel: np.ndarray,
    masses: np.ndarray,
    feature_spec: Any,
) -> np.ndarray:
    """Build node features from raw position/velocity/mass arrays using feature_spec."""
    spec = normalize_feature_spec(feature_spec)

    parts = {
        "pos": pos,
        "vel": vel,
        "mass": np.asarray(masses)[..., np.newaxis],
    }
    return np.concatenate([parts[name] for name in spec["include"]], axis=-1)


def build_node_features_torch(
    pos: torch.Tensor,
    vel: torch.Tensor,
    masses: torch.Tensor,
    feature_spec: Any,
) -> torch.Tensor:
    """Torch equivalent of build_node_features_np with matching feature ordering."""
    spec = normalize_feature_spec(feature_spec)

    parts = {
        "pos": pos,
        "vel": vel,
        "mass": masses.to(dtype=pos.dtype, device=pos.device).unsqueeze(-1),
    }
    return torch.cat([parts[name] for name in spec["include"]], dim=-1)


def _trajectory_min_pairwise_distance(positions: np.ndarray) -> float:
    """Return the minimum pairwise distance over all timesteps of one trajectory."""
    # positions shape: (T, n, dim)
    t_steps, n_bodies, _ = positions.shape
    if n_bodies < 2 or t_steps == 0:
        return float("inf")

    dmin = float("inf")
    for t in range(t_steps):
        pos_t = positions[t]
        for i in range(n_bodies):
            for j in range(i + 1, n_bodies):
                d = float(np.linalg.norm(pos_t[j] - pos_t[i]))
                if d < dmin:
                    dmin = d
    return dmin


def _safe_frame_mask(positions: np.ndarray, min_separation: float) -> np.ndarray:
    """Return a boolean mask of frames where all pairwise distances are >= threshold."""
    t_steps, n_bodies, _ = positions.shape
    if n_bodies < 2 or t_steps == 0:
        return np.ones(t_steps, dtype=bool)

    mask = np.ones(t_steps, dtype=bool)
    for t in range(t_steps):
        pos_t = positions[t]
        for i in range(n_bodies):
            for j in range(i + 1, n_bodies):
                d = float(np.linalg.norm(pos_t[j] - pos_t[i]))
                if d < min_separation:
                    mask[t] = False
                    break
            if not mask[t]:
                break
    return mask


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
    - Node features x: configurable subset/augmentation of [pos, vel, mass]
    - Targets y: accelerations
    - edge_index: fully connected graph

    Parameters
    ----------
    data_path : str or Path
        Path to .npz file with trajectory data
        Expected keys: 'positions', 'velocities', 'masses'
        Optional metadata keys: 'force_name', 'force_kwargs', 'traj_offsets'
    G : float, optional (default=1.0)
        Default gravity coefficient used when force metadata is absent
    softening : float, optional (default=1e-2)
        Default gravity softening used when force metadata is absent
    force_fn : str, optional
        Override force function name. If omitted, uses NPZ metadata when
        available, else falls back to 'gravity'.
    force_kwargs : dict, optional
        Override kwargs passed to the selected force function.
    include_velocity : bool, optional
        Backward-compatible alias for toggling velocity features only.
        Used only when feature_spec is not provided.
    feature_spec : dict | list[str], optional
        Input feature specification.
                - list form: ["pos", "mass"] means include only those base features
                - dict form: {"include": [...]} (augment must be empty if present)
    """

    def __init__(
            self,
            data_path: str | Path,
            G: float = 1.0,
            softening: float = 1e-2,
            force_fn: Optional[str] = None,
            force_kwargs: Optional[dict] = None,
            include_velocity: Optional[bool] = None,
            feature_spec: Any = None,
    ):
        self.data_path = Path(data_path)
        self.feature_spec = normalize_feature_spec(
            feature_spec=feature_spec,
            include_velocity=include_velocity,
        )
        self.include_velocity = "vel" in self.feature_spec["include"]

        # Load data
        data = np.load(self.data_path)
        self.positions = data["positions"]
        self.velocities = data["velocities"]
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

        # Support both trajectory-major (n_traj, T, n, dim) and pre-flattened
        # frame-major (n_samples, n, dim) datasets.
        if self.positions.ndim == 4:
            self.n_traj, self.T, self.n, self.dim = self.positions.shape
            self.positions = self.positions.reshape(-1, self.n, self.dim)
            self.velocities = self.velocities.reshape(-1, self.n, self.dim)
            self.traj_offsets = None
        elif self.positions.ndim == 3:
            if self.velocities.shape != self.positions.shape:
                raise ValueError(
                    f"positions and velocities must have matching shapes; got "
                    f"{self.positions.shape} and {self.velocities.shape}."
                )
            self.T, self.n, self.dim = self.positions.shape
            if "traj_offsets" in data.files:
                offsets = np.asarray(data["traj_offsets"], dtype=np.int64)
                if offsets.ndim != 1 or offsets.size < 2:
                    raise ValueError(
                        f"traj_offsets must be a 1D array with at least 2 entries; got {offsets.shape}."
                    )
                if offsets[0] != 0 or offsets[-1] != self.positions.shape[0]:
                    raise ValueError(
                        "traj_offsets must start at 0 and end at n_samples for frame-major datasets."
                    )
                if np.any(offsets[1:] < offsets[:-1]):
                    raise ValueError("traj_offsets must be non-decreasing.")
                self.traj_offsets = offsets
                self.n_traj = offsets.size - 1
            else:
                self.traj_offsets = None
                self.n_traj = 1
        else:
            raise ValueError(
                f"positions must have shape (n_traj, T, n, dim) or (n_samples, n, dim); "
                f"got {self.positions.shape}."
            )

        self.n_samples = self.positions.shape[0]
        self.n_node_features = node_feature_dim(self.dim, self.feature_spec)

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
            - x: configured node features based on feature_spec
            - y: accelerations shape (n, dim)
            - edge_index: shape (2, n_edges)
        """
        pos = self.positions[idx]  # (n, dim)
        vel = self.velocities[idx]  # (n, dim)

        # Compute force-consistent targets on-the-fly.
        acc = self.force_fn(pos, self.masses, **self.force_kwargs)

        x = build_node_features_np(
            pos=pos,
            vel=vel,
            masses=self.masses,
            feature_spec=self.feature_spec,
        )

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

    # Optional trajectory-level filtering using minimum pairwise separation.
    min_separation = float(ic_kwargs.pop("min_separation", 0.0))
    max_retries = ic_kwargs.pop("max_retries", None)
    if min_separation < 0:
        raise ValueError("min_separation must be >= 0.")
    if max_retries is None:
        max_retries = 3
    else:
        max_retries = int(max_retries)
    if max_retries < 1:
        raise ValueError("max_retries must be >= 1.")

    n_frame_filtered = 0
    if min_separation <= 0:
        # Generate initial conditions
        pos_batch, vel_batch = sim.batch_initial_conditions(
            n_trajectories=n_trajectories, seed=seed, **ic_kwargs
        )

        # Simulate
        results = sim.simulate_batch(pos_batch, vel_batch, t_end, dt, save_every)
    else:
        # Per trajectory: retry a small number of times for a fully safe rollout,
        # then fall back to dropping unsafe frames from that trajectory.
        results = []
        total_simulations = 0

        for traj_idx in range(n_trajectories):
            accepted = False
            candidate = None

            for attempt_idx in range(max_retries):
                total_simulations += 1
                pos_batch, vel_batch = sim.batch_initial_conditions(
                    n_trajectories=1,
                    seed=seed + traj_idx * max_retries + attempt_idx,
                    **ic_kwargs,
                )
                candidate = sim.simulate(pos_batch[0], vel_batch[0], t_end, dt, save_every)
                if _trajectory_min_pairwise_distance(candidate["positions"]) >= min_separation:
                    results.append(candidate)
                    accepted = True
                    break

            if accepted:
                continue

            if candidate is None:
                raise RuntimeError("Trajectory sampling failed unexpectedly.")

            safe_mask = _safe_frame_mask(candidate["positions"], min_separation)
            if not np.any(safe_mask):
                raise ValueError(
                    f"No safe frames remain for trajectory {traj_idx} after {max_retries} retries "
                    f"with min_separation={min_separation}. Relax min_separation or increase max_retries."
                )

            results.append(
                {
                    "positions": candidate["positions"][safe_mask],
                    "velocities": candidate["velocities"][safe_mask],
                    "times": candidate["times"][safe_mask],
                }
            )
            n_frame_filtered += 1

        print(
            f"Trajectory filter produced {len(results)} trajectories in {total_simulations} simulations "
            f"with {n_frame_filtered} frame-filtered fallback trajectories "
            f"(min_separation={min_separation}, max_retries={max_retries})."
        )

    # Stack when all trajectories have equal frame counts, otherwise preserve all
    # data by concatenating frames across trajectories.
    lengths = [r["positions"].shape[0] for r in results]
    traj_offsets = None
    used_frame_filtering = min_separation > 0 and n_frame_filtered > 0
    if len(set(lengths)) == 1 and not used_frame_filtering:
        positions = np.stack([r["positions"] for r in results])
        velocities = np.stack([r["velocities"] for r in results])
        times = results[0]["times"]
    else:
        positions = np.concatenate([r["positions"] for r in results], axis=0)
        velocities = np.concatenate([r["velocities"] for r in results], axis=0)
        times = np.concatenate([r["times"] for r in results], axis=0)
        lengths_arr = np.asarray(lengths, dtype=np.int64)
        traj_offsets = np.concatenate(
            [np.array([0], dtype=np.int64), np.cumsum(lengths_arr, dtype=np.int64)]
        )
    masses = sim.masses

    force_name = getattr(sim.force_fn, "__name__", "gravity")
    force_kwargs = json.dumps(sim.force_kwargs)

    # Save
    payload = {
        "positions": positions,
        "velocities": velocities,
        "times": times,
        "masses": masses,
        "force_name": force_name,
        "force_kwargs": force_kwargs,
    }
    if traj_offsets is not None:
        payload["traj_offsets"] = traj_offsets

    np.savez_compressed(output_path, **payload)

    print(f"Saved {n_trajectories} trajectories to {output_path}")
    print(
        "Data shape: "
        f"positions {positions.shape}, "
        f"velocities {velocities.shape}"
    )

    return output_path
