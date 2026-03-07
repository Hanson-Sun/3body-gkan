"""
nbody.py — Fast N-Body Simulator
=================================
A self-contained class for simulating N-body particle systems with:
  - Configurable force functions
  - Vectorised RK4 integrator (NumPy, optional Numba JIT)
  - Random initial condition generation (with presets)
  - Efficient binary save/load (NPZ + optional HDF5)
  - Static trajectory visualizer
  - Live visualizer (toggled via simulate(..., live=True))
"""

from __future__ import annotations

import time
import warnings
from pathlib import Path
from typing import Callable, Literal, Optional

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

# ── Optional accelerators ────────────────────────────────────────────────────
try:
    from numba import njit, prange

    _NUMBA = True
except ImportError:
    _NUMBA = False
    warnings.warn("numba not found — running in pure NumPy mode (slower).", stacklevel=2)

try:
    import h5py

    _H5PY = True
except ImportError:
    _H5PY = False

try:
    from tqdm import tqdm as _tqdm

    _TQDM = True
except ImportError:
    _TQDM = False


# ── Built-in force functions ─────────────────────────────────────────────────

def gravity(pos: np.ndarray, masses: np.ndarray,
            G: float = 1.0, softening: float = 1e-2) -> np.ndarray:
    """
    Newtonian gravity — fully vectorised, no Python loops.
    Uses (N,1,D)-(1,N,D) broadcasting to compute all pairwise differences at once.
    Returns acceleration array (F/m) of shape (N, D).
    """
    # dr[i,j] = pos[j] - pos[i],  shape (N, N, D)
    dr = pos[np.newaxis, :, :] - pos[:, np.newaxis, :]  # (N, N, D)
    # squared distances + softening, shape (N, N)
    dist2 = (dr * dr).sum(axis=-1) + softening * softening
    # inv_dist3[i,j] = 1 / |r_j - r_i|^3, zero on diagonal
    inv_dist3 = dist2 ** (-1.5)
    np.fill_diagonal(inv_dist3, 0.0)
    # acc[i] = G * sum_j m_j * (r_j - r_i) / |r_j - r_i|^3
    # einsum: (N,N) * (N,N,D) -> sum over j -> (N,D)
    acc = G * np.einsum('ij,ijd->id', masses[np.newaxis, :] * inv_dist3, dr)
    return acc


def spring_mesh(pos: np.ndarray, masses: np.ndarray,
                edges: np.ndarray, rest_lengths: np.ndarray,
                k: float = 1.0, damping: float = 0.0,
                vel: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Spring-mass mesh — vectorised over all edges at once.
    edges: (E, 2) int array of particle index pairs
    rest_lengths: (E,) float array
    """
    i_idx, j_idx = edges[:, 0], edges[:, 1]
    dr = pos[j_idx] - pos[i_idx]  # (E, D)
    dist = np.linalg.norm(dr, axis=-1, keepdims=True)  # (E, 1)
    dist = np.maximum(dist, 1e-12)
    unit = dr / dist  # (E, D)
    f = k * (dist - rest_lengths[:, np.newaxis]) * unit  # (E, D)

    if damping > 0.0 and vel is not None:
        f += damping * (vel[j_idx] - vel[i_idx])

    acc = np.zeros_like(pos)
    np.add.at(acc, i_idx, f / masses[i_idx, np.newaxis])
    np.add.at(acc, j_idx, -f / masses[j_idx, np.newaxis])
    return acc


def coulomb(pos: np.ndarray, masses: np.ndarray,
            charges: np.ndarray, k_e: float = 1.0,
            softening: float = 1e-3) -> np.ndarray:
    """Coulomb force — fully vectorised with broadcasting."""
    dr = pos[np.newaxis, :, :] - pos[:, np.newaxis, :]  # (N, N, D)
    dist2 = (dr * dr).sum(axis=-1) + softening * softening  # (N, N)
    inv_d3 = dist2 ** (-1.5)
    np.fill_diagonal(inv_d3, 0.0)
    # charge product matrix: q_i * q_j
    qmat = charges[:, np.newaxis] * charges[np.newaxis, :]  # (N, N)
    # F on i from j is repulsive (same sign → positive qmat → away from j)
    # dr[i,j] = pos[j]-pos[i], so we negate to get force direction away
    acc = -k_e * np.einsum('ij,ijd->id', qmat * inv_d3, dr)
    return acc / masses[:, np.newaxis]


# ── If Numba available, JIT-compile the gravity inner loop ───────────────────
if _NUMBA:
    @njit(parallel=True, fastmath=True, cache=True)
    def _gravity_numba(pos, masses, G, softening):
        N = pos.shape[0]
        D = pos.shape[1]
        acc = np.zeros((N, D))
        soft2 = softening * softening
        for i in prange(N):
            for j in range(N):
                if i == j:
                    continue
                dr = pos[j] - pos[i]
                dist2 = soft2
                for d in range(D):
                    dist2 += dr[d] * dr[d]
                inv_dist3 = dist2 ** (-1.5)
                fac = G * masses[j] * inv_dist3
                for d in range(D):
                    acc[i, d] += fac * dr[d]
        return acc


    def gravity_fast(pos, masses, G=1.0, softening=1e-2):
        """JIT-compiled gravity (use in place of `gravity` for speed)."""
        return _gravity_numba(pos, masses, G, softening)
else:
    gravity_fast = gravity  # fallback


# ═══════════════════════════════════════════════════════════════════════════════
# NBodySimulator
# ═══════════════════════════════════════════════════════════════════════════════

class NBodySimulator:
    """
    Fast N-body particle simulator with RK4 integration.

    Quick start
    -----------
    >>> sim = NBodySimulator(n=3, dim=2, masses=[1, 1, 1])
    >>> pos0, vel0 = sim.random_initial_conditions(seed=0)
    >>> tracks = sim.simulate(pos0, vel0, t_end=10.0, dt=0.01)
    >>> NBodySimulator.plot_tracks(tracks)
    >>> sim.save(tracks, "run_001.npz")
    """

    # ── Construction ──────────────────────────────────────────────────────────

    def __init__(
            self,
            n: int = 3,
            dim: int = 2,
            masses: Optional[np.ndarray | list] = None,
            force_fn: Optional[Callable] = None,
            force_kwargs: Optional[dict] = None,
    ):
        """
        Parameters
        ----------
        n           : number of particles
        dim         : spatial dimension (2 or 3)
        masses      : (n,) array of particle masses; defaults to all ones
        force_fn    : callable(pos, masses, **force_kwargs) → acc (N, D)
                      Defaults to gravity_fast (Numba-accelerated if available)
        force_kwargs: extra keyword arguments forwarded to force_fn every step
        """
        self.n = n
        self.dim = dim
        self.masses = np.ones(n) if masses is None else np.asarray(masses, dtype=float)
        assert self.masses.shape == (n,), f"masses must be length {n}"

        self.force_fn = gravity_fast if force_fn is None else force_fn
        self.force_kwargs = force_kwargs or {}

    # ── Initial conditions ────────────────────────────────────────────────────

    def random_initial_conditions(
            self,
            seed: Optional[int] = None,
            pos_scale: float = 1.0,
            vel_scale: float = 0.5,
            zero_momentum: bool = True,
            preset: Literal["random", "circular", "figure8", "solar"] = "random",
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Generate initial positions and velocities.

        Parameters
        ----------
        seed          : RNG seed for reproducibility
        pos_scale     : scale factor for initial positions
        vel_scale     : scale factor for initial velocities
        zero_momentum : subtract centre-of-mass velocity (conserves momentum)
        preset        : "random" | "circular" | "figure8" | "solar"
                        "circular"  — equal-mass bodies on a ring with circular velocities
                        "figure8"   — classic 3-body figure-8 (n=3, dim=2 only)
                        "solar"     — sun-planet system (n bodies, dim=2)

        Returns
        -------
        pos0 : (n, dim) float64
        vel0 : (n, dim) float64
        """
        rng = np.random.default_rng(seed)

        if preset == "figure8":
            if self.n != 3 or self.dim != 2:
                raise ValueError("figure8 preset requires n=3, dim=2")
            # Chenciner & Montgomery (2000) — normalised
            pos0 = np.array([
                [0.97000436, -0.24308753],
                [-0.97000436, 0.24308753],
                [0.0, 0.0]
            ])
            v0 = np.array([0.93240737, 0.86473146])
            vel0 = np.array([v0 / 2, v0 / 2, -v0])
            return pos0 * pos_scale, vel0 * vel_scale

        if preset == "circular":
            if self.dim != 2:
                raise ValueError("circular preset requires dim=2")
            angles = np.linspace(0, 2 * np.pi, self.n, endpoint=False)
            pos0 = pos_scale * np.column_stack([np.cos(angles), np.sin(angles)])
            # Circular velocity magnitude for equal mass ring
            v_circ = np.sqrt(self.masses.sum() / pos_scale) * vel_scale
            vel0 = v_circ * np.column_stack([-np.sin(angles), np.cos(angles)])
            return pos0, vel0

        if preset == "solar":
            if self.dim != 2:
                raise ValueError("solar preset requires dim=2")
            # First body is the "sun" (large mass assumed)
            pos0 = np.zeros((self.n, 2))
            vel0 = np.zeros((self.n, 2))
            for i in range(1, self.n):
                r = pos_scale * (0.5 + rng.uniform(0, 1))
                angle = rng.uniform(0, 2 * np.pi)
                pos0[i] = r * np.array([np.cos(angle), np.sin(angle)])
                v = np.sqrt(self.masses[0] / r) * vel_scale
                vel0[i] = v * np.array([-np.sin(angle), np.cos(angle)])
            return pos0, vel0

        # default: "random" — rejection-sample until all pairs are well-separated
        # min_sep scales as pos_scale / N^(1/D) so packing stays feasible for any N
        min_sep = pos_scale / (max(self.n, 1) ** (1.0 / self.dim)) * 0.4
        for _ in range(10_000):
            pos0 = rng.uniform(-pos_scale, pos_scale, (self.n, self.dim))
            if self.n < 2:
                break
            # Vectorised pairwise check
            diff = pos0[:, np.newaxis, :] - pos0[np.newaxis, :, :]  # (N,N,D)
            dist2 = (diff * diff).sum(axis=-1)  # (N,N)
            np.fill_diagonal(dist2, np.inf)
            if np.sqrt(dist2.min()) >= min_sep:
                break
        else:
            raise RuntimeError(
                f"Could not place {self.n} bodies with min_sep={min_sep:.2f} "
                f"inside pos_scale={pos_scale}. Increase pos_scale."
            )
        vel0 = rng.uniform(-vel_scale, vel_scale, (self.n, self.dim))
        if zero_momentum:
            total_p = (self.masses[:, None] * vel0).sum(axis=0)
            vel0 -= total_p / self.masses.sum()
        return pos0, vel0

    def batch_initial_conditions(
            self,
            n_trajectories: int,
            seed: Optional[int] = None,
            **kwargs
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Generate a batch of random initial conditions.

        Returns
        -------
        pos_batch : (n_trajectories, n, dim)
        vel_batch : (n_trajectories, n, dim)
        """
        rng = np.random.default_rng(seed)
        seeds = rng.integers(0, 2 ** 31, size=n_trajectories)
        pos_list, vel_list = [], []
        for s in seeds:
            p, v = self.random_initial_conditions(seed=int(s), **kwargs)
            pos_list.append(p)
            vel_list.append(v)
        return np.stack(pos_list), np.stack(vel_list)

    # ── Integration ───────────────────────────────────────────────────────────

    def _derivatives(self, pos: np.ndarray, vel: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Returns (dpos/dt, dvel/dt) = (vel, acc)."""
        return vel, self.force_fn(pos, self.masses, **self.force_kwargs)

    def _rk4_step(
            self, pos: np.ndarray, vel: np.ndarray, dt: float
    ) -> tuple[np.ndarray, np.ndarray]:
        """Single RK4 step — minimal allocations, no redundant copies."""
        h = dt
        h2 = 0.5 * h
        h6 = h / 6.0

        a1 = self.force_fn(pos, self.masses, **self.force_kwargs)  # k1v
        # k1p = vel  (no copy needed, used read-only below)

        p2 = pos + h2 * vel;
        v2 = vel + h2 * a1
        a2 = self.force_fn(p2, self.masses, **self.force_kwargs)

        p3 = pos + h2 * v2;
        v3 = vel + h2 * a2
        a3 = self.force_fn(p3, self.masses, **self.force_kwargs)

        p4 = pos + h * v3;
        v4 = vel + h * a3
        a4 = self.force_fn(p4, self.masses, **self.force_kwargs)

        new_pos = pos + h6 * (vel + 2.0 * v2 + 2.0 * v3 + v4)
        new_vel = vel + h6 * (a1 + 2.0 * a2 + 2.0 * a3 + a4)
        return new_pos, new_vel

    def simulate(
            self,
            pos0: np.ndarray,
            vel0: np.ndarray,
            t_end: float,
            dt: float = 0.005,
            save_every: int = 1,
            visualize: bool = False,
            trail: int = 80,
            steps_per_frame: int = 5,
            energy_check: bool = False,
            verbose: bool = False,
            adaptive_dt: bool = True,
            close_encounter_radius: Optional[float] = None,
    ) -> dict:
        """
        Simulate the system using RK4.

        Parameters
        ----------
        pos0                  : (n, dim) initial positions
        vel0                  : (n, dim) initial velocities
        t_end                 : end time
        dt                    : base timestep
        save_every            : store every k-th step (reduces memory)
        visualize             : if True, opens a live window and renders frame-by-frame
        trail                 : number of past positions shown as a fading trail
        steps_per_frame       : how many RK4 steps to run before redrawing
        energy_check          : compute and store kinetic + potential energies
        verbose               : print progress bar (requires tqdm)
        adaptive_dt           : if True, automatically halve dt when any pair of bodies
                                comes within close_encounter_radius of each other,
                                preventing energy blowup near collisions
        close_encounter_radius: distance threshold that triggers dt halving.
                                Defaults to 10x the softening parameter (or 0.1).

        Returns
        -------
        dict with keys:
          'positions'  : (T, n, dim)
          'velocities' : (T, n, dim)
          'times'      : (T,)
          'masses'     : (n,)
          'dt'         : float
          'energies'   : (T, 3) or None
        """
        pos0 = np.asarray(pos0, dtype=np.float64)
        vel0 = np.asarray(vel0, dtype=np.float64)
        assert pos0.shape == (self.n, self.dim)
        assert vel0.shape == (self.n, self.dim)

        softening = self.force_kwargs.get("softening", 1e-2)
        if close_encounter_radius is None:
            close_encounter_radius = max(softening * 10.0, 0.05)

        n_steps = int(t_end / dt)
        n_saved = n_steps // save_every + 1

        pos_track = np.empty((n_saved, self.n, self.dim))
        vel_track = np.empty((n_saved, self.n, self.dim))
        time_track = np.empty(n_saved)
        energies = np.empty((n_saved, 3)) if energy_check else None

        pos = pos0.copy()
        vel = vel0.copy()
        pos_track[0] = pos
        vel_track[0] = vel
        time_track[0] = 0.0
        if energy_check:
            energies[0] = self._compute_energy(pos, vel)

        save_idx = 1
        t = 0.0

        # ── Set up live plot ──────────────────────────────────────────────────
        if visualize:
            assert self.dim == 2, "visualize only supports dim=2"
            plt.ion()
            fig, ax = plt.subplots(figsize=(6, 6), facecolor="#080818")
            ax.set_facecolor("#080818")
            ax.set_aspect("equal")
            for sp in ax.spines.values():
                sp.set_edgecolor("#1a1a3a")
            ax.tick_params(colors="#555577")

            spread = max(np.abs(pos0).max() * 2.5, 2.0)
            ax.set_xlim(-spread, spread)
            ax.set_ylim(-spread, spread)

            COLORS = plt.get_cmap("tab10")(np.linspace(0, 1, self.n))
            history = [[] for _ in range(self.n)]

            trail_lines = [ax.plot([], [], color=COLORS[i], alpha=0.4, lw=1.2)[0]
                           for i in range(self.n)]
            glow_dots = [ax.plot([], [], 'o', color=COLORS[i], ms=14, alpha=0.15)[0]
                         for i in range(self.n)]
            dots = [ax.plot([], [], 'o', color=COLORS[i], ms=5,
                            markeredgecolor='white', markeredgewidth=0.4)[0]
                    for i in range(self.n)]
            time_label = ax.text(0.02, 0.97, '', transform=ax.transAxes,
                                 color='#8888aa', fontsize=9, va='top', family='monospace')
            ax.set_title(f"{self.n}-Body Simulation", color="#aaaacc", fontsize=11)
            fig.tight_layout()

        # ── Main integration loop ─────────────────────────────────────────────
        iterator = range(1, n_steps + 1)
        if verbose and _TQDM:
            iterator = _tqdm(iterator, total=n_steps, desc="Simulating", unit="step")

        current_dt = dt
        for step in iterator:

            # ── Adaptive dt: check proximity BEFORE stepping ──────────────────
            if adaptive_dt:
                # Vectorised pairwise distances — no Python loop
                diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]  # (N,N,D)
                dist2 = (diff * diff).sum(axis=-1)  # (N,N)
                np.fill_diagonal(dist2, np.inf)
                min_dist = np.sqrt(dist2.min())
                if min_dist < close_encounter_radius:
                    ratio = min_dist / close_encounter_radius
                    n_sub = int(np.clip(32.0 / (ratio + 0.05), 8, 512))
                    sub_dt = current_dt / n_sub
                    for _ in range(n_sub):
                        pos, vel = self._rk4_step(pos, vel, sub_dt)
                        t += sub_dt
                else:
                    pos, vel = self._rk4_step(pos, vel, current_dt)
                    t += current_dt
            else:
                pos, vel = self._rk4_step(pos, vel, current_dt)
                t += current_dt

            if step % save_every == 0 and save_idx < n_saved:
                pos_track[save_idx] = pos
                vel_track[save_idx] = vel
                time_track[save_idx] = t
                if energy_check:
                    energies[save_idx] = self._compute_energy(pos, vel)
                save_idx += 1

            # ── Redraw every `steps_per_frame` steps ──────────────────────
            if visualize and step % steps_per_frame == 0:
                for i in range(self.n):
                    history[i].append(pos[i].copy())
                    if len(history[i]) > trail:
                        history[i].pop(0)
                    tr = np.array(history[i])
                    trail_lines[i].set_data(tr[:, 0], tr[:, 1])
                    glow_dots[i].set_data([pos[i, 0]], [pos[i, 1]])
                    dots[i].set_data([pos[i, 0]], [pos[i, 1]])
                time_label.set_text(f"t = {t:.2f}")
                fig.canvas.draw()
                fig.canvas.flush_events()

        if visualize:
            plt.ioff()
            plt.show()

        actual = save_idx
        result = {
            "positions": pos_track[:actual],
            "velocities": vel_track[:actual],
            "times": time_track[:actual],
            "masses": self.masses.copy(),
            "dt": dt,
            "energies": energies[:actual] if energy_check else None,
        }
        if energy_check:
            e = energies[:actual]
            drift = abs((e[-1, 2] - e[0, 2]) / (abs(e[0, 2]) + 1e-12))
            if drift > 0.05:
                warnings.warn(
                    f"Energy drift = {drift:.1%} — try a smaller dt (currently {dt}) "
                    f"or reduce vel_scale in initial conditions.",
                    stacklevel=2
                )
        return result

    def simulate_batch(
            self,
            pos_batch: np.ndarray,
            vel_batch: np.ndarray,
            t_end: float,
            dt: float = 0.01,
            save_every: int = 1,
            verbose: bool = True,
    ) -> list[dict]:
        """
        Simulate multiple initial conditions sequentially.
        Returns a list of trajectory dicts.
        """
        n_traj = pos_batch.shape[0]
        results = []
        iterator = range(n_traj)
        if verbose and _TQDM:
            iterator = _tqdm(iterator, total=n_traj, desc="Batch simulate")
        for i in iterator:
            r = self.simulate(pos_batch[i], vel_batch[i], t_end, dt, save_every)
            results.append(r)
        return results

    # ── Energy diagnostics ────────────────────────────────────────────────────

    def _compute_energy(self, pos: np.ndarray, vel: np.ndarray) -> np.ndarray:
        """Returns [KE, PE, total]."""
        ke = 0.5 * np.sum(self.masses[:, None] * vel ** 2)
        pe = 0.0
        G = self.force_kwargs.get("G", 1.0)
        soft = self.force_kwargs.get("softening", 1e-3)
        for i in range(self.n):
            for j in range(i + 1, self.n):
                dr = pos[j] - pos[i]
                dist = np.sqrt(np.dot(dr, dr) + soft ** 2)
                pe -= G * self.masses[i] * self.masses[j] / dist
        return np.array([ke, pe, ke + pe])

    # ── Save / Load ───────────────────────────────────────────────────────────

    @staticmethod
    def save(
            track: dict,
            path: str | Path,
            fmt: Literal["npz", "npy", "h5"] = "npz",
            compress: bool = True,
    ) -> None:
        """
        Save a trajectory to disk.

        Formats
        -------
        'npz' : NumPy compressed archive (default, no extra deps)
        'npy' : raw uncompressed NumPy tensors (fastest read)
        'h5'  : HDF5 via h5py (requires h5py; best for huge datasets)
        """
        path = Path(path)
        if fmt == "npz":
            fn = np.savez_compressed if compress else np.savez
            fn(
                path.with_suffix(".npz"),
                positions=track["positions"],
                velocities=track["velocities"],
                times=track["times"],
                masses=track["masses"],
                dt=np.array([track["dt"]]),
                **({"energies": track["energies"]}
                   if track.get("energies") is not None else {}),
            )
        elif fmt == "npy":
            d = path.with_suffix("")
            d.mkdir(parents=True, exist_ok=True)
            np.save(d / "positions.npy", track["positions"])
            np.save(d / "velocities.npy", track["velocities"])
            np.save(d / "times.npy", track["times"])
            np.save(d / "masses.npy", track["masses"])
        elif fmt == "h5":
            if not _H5PY:
                raise ImportError("h5py is required for HDF5 format. pip install h5py")
            with h5py.File(path.with_suffix(".h5"), "w") as f:
                for k, v in track.items():
                    if v is not None:
                        f.create_dataset(k, data=v, compression="gzip" if compress else None)
        else:
            raise ValueError(f"Unknown format '{fmt}'. Choose 'npz', 'npy', or 'h5'.")

    @staticmethod
    def load(path: str | Path) -> dict:
        """
        Load a trajectory saved by NBodySimulator.save().
        Auto-detects format from file extension (.npz, .h5, or directory).
        """
        path = Path(path)
        if path.suffix == ".npz" or path.with_suffix(".npz").exists():
            p = path if path.suffix == ".npz" else path.with_suffix(".npz")
            data = np.load(p)
            return {
                "positions": data["positions"],
                "velocities": data["velocities"],
                "times": data["times"],
                "masses": data["masses"],
                "dt": float(data["dt"][0]),
                "energies": data["energies"] if "energies" in data else None,
            }
        elif path.suffix == ".h5" or path.with_suffix(".h5").exists():
            if not _H5PY:
                raise ImportError("h5py required. pip install h5py")
            p = path if path.suffix == ".h5" else path.with_suffix(".h5")
            with h5py.File(p, "r") as f:
                return {k: f[k][()] for k in f.keys()}
        elif path.is_dir():
            return {
                "positions": np.load(path / "positions.npy"),
                "velocities": np.load(path / "velocities.npy"),
                "times": np.load(path / "times.npy"),
                "masses": np.load(path / "masses.npy"),
                "dt": None,
                "energies": None,
            }
        else:
            raise FileNotFoundError(f"Cannot find trajectory at {path}")

    @staticmethod
    def save_batch(
            tracks: list[dict],
            path: str | Path,
            fmt: Literal["npz", "h5"] = "npz",
    ) -> None:
        """
        Save a batch of trajectories as a single stacked file.
        All tracks must have the same shape.

        Saves stacked arrays: positions (B, T, N, D), velocities (B, T, N, D), etc.
        """
        path = Path(path)
        pos = np.stack([t["positions"] for t in tracks])
        vel = np.stack([t["velocities"] for t in tracks])
        times = np.stack([t["times"] for t in tracks])
        masses = tracks[0]["masses"]
        if fmt == "npz":
            np.savez_compressed(
                path.with_suffix(".npz"),
                positions=pos, velocities=vel, times=times, masses=masses
            )
        elif fmt == "h5":
            if not _H5PY:
                raise ImportError("h5py required.")
            with h5py.File(path.with_suffix(".h5"), "w") as f:
                f.create_dataset("positions", data=pos, compression="gzip")
                f.create_dataset("velocities", data=vel, compression="gzip")
                f.create_dataset("times", data=times, compression="gzip")
                f.create_dataset("masses", data=masses)

    # ── Static Visualizer ─────────────────────────────────────────────────────

    @staticmethod
    def plot_tracks(
            track: dict,
            ax: Optional[plt.Axes] = None,
            color_by: Literal["body", "time", "speed"] = "body",
            show_start: bool = True,
            show_end: bool = True,
            trail_alpha: float = 0.7,
            figsize: tuple = (8, 8),
            title: str = "N-Body Trajectories",
            save_path: Optional[str | Path] = None,
            show: bool = True,
            cmap: str = "tab10",
    ) -> plt.Figure:
        """
        Plot static 2D/3D trajectory tracks.

        Parameters
        ----------
        track      : dict returned by simulate()
        ax         : existing Axes; created if None
        color_by   : 'body' — each body its own colour
                     'time' — trail fades from dark (start) to bright (end)
                     'speed'— trail coloured by instantaneous speed
        show_start : mark initial positions with a circle
        show_end   : mark final positions with a star
        """
        pos = track["positions"]  # (T, N, D)
        vel = track.get("velocities")
        T, N, D = pos.shape

        if D not in (2, 3):
            raise ValueError("plot_tracks only supports dim=2 or dim=3")

        fig = None
        if ax is None:
            fig = plt.figure(figsize=figsize)
            if D == 3:
                ax = fig.add_subplot(111, projection="3d")
            else:
                ax = fig.add_subplot(111)
        else:
            fig = ax.get_figure()

        colors = plt.get_cmap(cmap)(np.linspace(0, 1, N))

        for i in range(N):
            xi = pos[:, i, :]  # (T, D)

            if color_by == "body":
                if D == 2:
                    ax.plot(xi[:, 0], xi[:, 1], color=colors[i],
                            alpha=trail_alpha, lw=1.2, label=f"Body {i}")
                else:
                    ax.plot(xi[:, 0], xi[:, 1], xi[:, 2], color=colors[i],
                            alpha=trail_alpha, lw=1.2, label=f"Body {i}")

            elif color_by == "time":
                # Gradient trail using LineCollection (2D only)
                if D != 2:
                    ax.plot(xi[:, 0], xi[:, 1], xi[:, 2], color=colors[i],
                            alpha=trail_alpha, lw=1.2)
                else:
                    segs = np.stack([xi[:-1], xi[1:]], axis=1)
                    lc = LineCollection(segs, cmap="plasma",
                                        norm=plt.Normalize(0, T), alpha=trail_alpha)
                    lc.set_array(np.arange(T - 1))
                    ax.add_collection(lc)

            elif color_by == "speed":
                if vel is not None and D == 2:
                    speed = np.linalg.norm(vel[:, i, :], axis=-1)
                    segs = np.stack([xi[:-1], xi[1:]], axis=1)
                    lc = LineCollection(segs, cmap="viridis",
                                        norm=plt.Normalize(speed.min(), speed.max()),
                                        alpha=trail_alpha)
                    lc.set_array(speed[:-1])
                    ax.add_collection(lc)
                else:
                    ax.plot(xi[:, 0], xi[:, 1], color=colors[i], alpha=trail_alpha)

            if show_start:
                if D == 2:
                    ax.scatter(*xi[0], s=80, color=colors[i], zorder=5, marker='o',
                               edgecolors='white', linewidths=0.8)
                else:
                    ax.scatter(*xi[0], s=80, color=colors[i], marker='o')

            if show_end:
                if D == 2:
                    ax.scatter(*xi[-1], s=120, color=colors[i], zorder=5, marker='*',
                               edgecolors='white', linewidths=0.8)
                else:
                    ax.scatter(*xi[-1], s=120, color=colors[i], marker='*')

        if D == 2:
            ax.set_aspect("equal")
        ax.set_title(title)
        ax.legend(loc="upper right", fontsize=8)
        plt.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
        if show:
            plt.show()
        return fig

    @staticmethod
    def plot_energy(
            track: dict,
            figsize: tuple = (10, 4),
            save_path: Optional[str | Path] = None,
            show: bool = True,
    ) -> plt.Figure:
        """Plot kinetic, potential, and total energy over time (energy conservation check)."""
        if track.get("energies") is None:
            raise ValueError("No energy data. Re-run with energy_check=True.")
        energies = track["energies"]
        times = track["times"]
        fig, axes = plt.subplots(1, 2, figsize=figsize)
        ax1, ax2 = axes
        ax1.plot(times, energies[:, 0], label="KE", color="tomato")
        ax1.plot(times, energies[:, 1], label="PE", color="steelblue")
        ax1.plot(times, energies[:, 2], label="Total", color="black", lw=2, ls="--")
        ax1.set_xlabel("Time")
        ax1.set_ylabel("Energy")
        ax1.set_title("Energy Components")
        ax1.legend()
        drift = np.abs((energies[:, 2] - energies[0, 2]) / (np.abs(energies[0, 2]) + 1e-12))
        ax2.semilogy(times, drift, color="purple")
        ax2.set_xlabel("Time")
        ax2.set_ylabel("|ΔE/E₀|")
        ax2.set_title("Energy Drift (RK4 quality)")
        plt.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
        if show:
            plt.show()
        return fig

    # ── Utilities ─────────────────────────────────────────────────────────────

    def compute_observables(self, track: dict) -> dict:
        """
        Compute useful physical observables from a trajectory.

        Returns
        -------
        dict with:
          'com_pos'       : (T, dim) centre-of-mass position
          'ang_momentum'  : (T,) total angular momentum (2D: scalar, 3D: vector)
          'lin_momentum'  : (T, dim) total linear momentum
          'pairwise_dist' : (T, N, N) pairwise distances at each timestep
          'separation_min': minimum pairwise separation over time (collision proxy)
        """
        pos = track["positions"]  # (T, N, D)
        vel = track["velocities"]  # (T, N, D)
        T, N, D = pos.shape
        m = self.masses

        com = (m[None, :, None] * pos).sum(axis=1) / m.sum()
        lin_mom = (m[None, :, None] * vel).sum(axis=1)

        if D == 2:
            ang_mom = (m[None, :] * (pos[:, :, 0] * vel[:, :, 1]
                                     - pos[:, :, 1] * vel[:, :, 0])).sum(axis=1)
        else:
            # cross product broadcast over time: (T, N, 3)
            rxv = np.cross(pos, vel)  # (T, N, 3)
            ang_mom = (m[None, :, None] * rxv).sum(axis=1)  # (T, 3)

        # Vectorised pairwise distances: (T, N, 1, D) - (T, 1, N, D) → (T, N, N)
        diff = pos[:, :, np.newaxis, :] - pos[:, np.newaxis, :, :]
        pw_dist = np.sqrt((diff * diff).sum(axis=-1))

        ii, jj = np.triu_indices(N, k=1)
        sep_min = pw_dist[:, ii, jj].min()

        return {
            "com_pos": com,
            "ang_momentum": ang_mom,
            "lin_momentum": lin_mom,
            "pairwise_dist": pw_dist,
            "separation_min": sep_min,
        }

    def __repr__(self):
        return (f"NBodySimulator(n={self.n}, dim={self.dim}, "
                f"masses={self.masses.tolist()}, "
                f"force_fn={self.force_fn.__name__}, "
                f"numba={'on' if _NUMBA else 'off'})")


# ═══════════════════════════════════════════════════════════════════════════════
# Example usage
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":

    print("=== Example 1: Classic Figure-8 Three-Body ===")
    sim = NBodySimulator(n=3, dim=2, masses=[1.0, 1.0, 1.0])
    pos0, vel0 = sim.random_initial_conditions(preset="figure8")
    track = sim.simulate(pos0, vel0, t_end=6.3, dt=0.005,
                         save_every=2, energy_check=True, verbose=True)
    print(f"Track shape: {track['positions'].shape}")
    NBodySimulator.plot_tracks(track, title="Figure-8 Three-Body", color_by="time")
    NBodySimulator.plot_energy(track)

    print("\n=== Example 2: Random 5-body gravity ===")
    sim5 = NBodySimulator(n=5, dim=2, force_fn=gravity_fast)
    pos0, vel0 = sim5.random_initial_conditions(seed=42, pos_scale=2.0, vel_scale=0.3)
    track5 = sim5.simulate(pos0, vel0, t_end=20, dt=0.01, save_every=5, verbose=True)
    NBodySimulator.plot_tracks(track5, title="5-Body Random", color_by="speed")

    print("\n=== Example 3: Save and reload ===")
    NBodySimulator.save(track5, "/tmp/five_body", fmt="npz")
    loaded = NBodySimulator.load("/tmp/five_body.npz")
    print(f"Loaded positions shape: {loaded['positions'].shape}")

    print("\n=== Example 4: Batch generation ===")
    pos_batch, vel_batch = sim.batch_initial_conditions(n_trajectories=10, seed=0)
    print(f"Batch shapes: {pos_batch.shape}, {vel_batch.shape}")

    print("\n=== Example 5: Custom force (inverse-cube) ===")


    def inv_cube_force(pos, masses, G=1.0, softening=1e-3):
        N, D = pos.shape
        acc = np.zeros((N, D))
        for i in range(N):
            for j in range(i + 1, N):
                dr = pos[j] - pos[i]
                dist = np.sqrt(np.dot(dr, dr) + softening ** 2)
                f = G * dr / dist ** 4  # 1/r^3 force
                acc[i] += masses[j] * f
                acc[j] -= masses[i] * f
        return acc


    sim_custom = NBodySimulator(n=3, dim=2, force_fn=inv_cube_force,
                                force_kwargs={"G": 0.5})
    pos0, vel0 = sim_custom.random_initial_conditions(seed=7)
    track_custom = sim_custom.simulate(pos0, vel0, t_end=15, dt=0.01)
    NBodySimulator.plot_tracks(track_custom, title="Inverse-Cube Force")

    # Uncomment for live animation:
    # track_live = sim.simulate(pos0, vel0, t_end=10, dt=0.005, live=True)
