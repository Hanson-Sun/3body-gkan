"""N-body gravitational simulator using symplectic (leapfrog) integrator."""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from typing import Optional, Callable


def gravity(positions: np.ndarray, masses: np.ndarray, G: float = 1.0,
            softening: float = 1e-2) -> np.ndarray:
    """
    Compute gravitational accelerations for all bodies.

    Parameters
    ----------
    positions : np.ndarray
        Positions of shape (n, dim)
    masses : np.ndarray
        Masses of shape (n,)
    G : float
        Gravitational constant
    softening : float
        Softening parameter to avoid singularities

    Returns
    -------
    np.ndarray
        Accelerations of shape (n, dim)
    """
    n = len(masses)
    dim = positions.shape[1]
    acc = np.zeros((n, dim))

    for i in range(n):
        for j in range(n):
            if i != j:
                r = positions[j] - positions[i]
                r_norm = np.sqrt(np.sum(r**2) + softening**2)
                acc[i] += G * masses[j] * r / r_norm**3

    return acc


class NBodySimulator:
    """
    N-body simulator using the leapfrog integrator.

    The leapfrog method is a second-order symplectic integrator that conserves
    energy well over long time periods.

    Parameters
    ----------
    masses : np.ndarray
        Masses of the bodies, shape (n,)
    force_fn : Callable, optional
        Force function with signature: f(positions, masses, **kwargs) -> accelerations
        If None, uses gravitational force (default)
    **force_kwargs
        Additional keyword arguments passed to force_fn (e.g., G=1.0, softening=1e-2)
    """

    def __init__(self, masses: np.ndarray,
                 force_fn: Optional[Callable] = None,
                 **force_kwargs):
        self.masses = np.asarray(masses)
        self.n = len(masses)
        self.force_fn = force_fn or gravity
        self.force_kwargs = force_kwargs if force_kwargs else {'G': 1.0, 'softening': 1e-2}

    def _accelerations(self, positions: np.ndarray) -> np.ndarray:
        """Compute accelerations from positions."""
        return self.force_fn(positions, self.masses, **self.force_kwargs)

    def leapfrog_step(self, pos: np.ndarray, vel: np.ndarray, dt: float) -> tuple:
        """
        Single leapfrog integration step.

        Leapfrog integration:
        1. v(t + dt/2) = v(t) + a(t) * dt/2
        2. x(t + dt) = x(t) + v(t + dt/2) * dt
        3. v(t + dt) = v(t + dt/2) + a(t + dt) * dt/2

        Parameters
        ----------
        pos : np.ndarray
            Current positions, shape (n, dim)
        vel : np.ndarray
            Current velocities, shape (n, dim)
        dt : float
            Time step

        Returns
        -------
        tuple
            New positions and velocities
        """
        # Half-step velocity
        acc = self._accelerations(pos)
        vel_half = vel + 0.5 * dt * acc

        # Full-step position
        pos_new = pos + dt * vel_half

        # Full-step velocity
        acc_new = self._accelerations(pos_new)
        vel_new = vel_half + 0.5 * dt * acc_new

        return pos_new, vel_new

    def simulate(self, pos0: np.ndarray, vel0: np.ndarray,
                 t_end: float, dt: float, save_every: int = 1) -> dict:
        """
        Simulate n-body system.

        Parameters
        ----------
        pos0 : np.ndarray
            Initial positions, shape (n, dim)
        vel0 : np.ndarray
            Initial velocities, shape (n, dim)
        t_end : float
            End time
        dt : float
            Time step
        save_every : int
            Save every k-th step

        Returns
        -------
        dict
            Dictionary with 'times', 'positions', 'velocities'
        """
        n_steps = int(t_end / dt)
        n_save = n_steps // save_every

        pos, vel = pos0.copy(), vel0.copy()
        dim = pos.shape[1]

        # Pre-allocate arrays
        positions = np.zeros((n_save, self.n, dim))
        velocities = np.zeros((n_save, self.n, dim))
        times = np.zeros(n_save)

        save_idx = 0
        for step in range(n_steps):
            if step % save_every == 0:
                positions[save_idx] = pos
                velocities[save_idx] = vel
                times[save_idx] = step * dt
                save_idx += 1

            pos, vel = self.leapfrog_step(pos, vel, dt)

        return {
            'times': times,
            'positions': positions,
            'velocities': velocities
        }

    def simulate_batch(self, pos_batch: np.ndarray, vel_batch: np.ndarray,
                       t_end: float, dt: float, save_every: int = 1) -> list:
        """
        Simulate multiple trajectories in batch.

        Parameters
        ----------
        pos_batch : np.ndarray
            Initial positions, shape (n_traj, n, dim)
        vel_batch : np.ndarray
            Initial velocities, shape (n_traj, n, dim)
        t_end : float
            End time
        dt : float
            Time step
        save_every : int
            Save every k-th step

        Returns
        -------
        list
            List of simulation results (one per trajectory)
        """
        results = []
        for pos0, vel0 in zip(pos_batch, vel_batch):
            result = self.simulate(pos0, vel0, t_end, dt, save_every)
            results.append(result)
        return results

    def random_initial_conditions(self, dim: int = 2,
                                   pos_scale: float = 1.0,
                                   vel_scale: float = 0.5,
                                   seed: Optional[int] = None) -> tuple:
        """
        Generate random initial conditions.

        Parameters
        ----------
        dim : int
            Spatial dimension
        pos_scale : float
            Scale for positions
        vel_scale : float
            Scale for velocities
        seed : int, optional
            Random seed

        Returns
        -------
        tuple
            Initial positions and velocities
        """
        if seed is not None:
            np.random.seed(seed)

        pos = pos_scale * (2 * np.random.rand(self.n, dim) - 1)
        vel = vel_scale * (2 * np.random.rand(self.n, dim) - 1)

        return pos, vel

    def batch_initial_conditions(self, n_trajectories: int, dim: int = 2,
                                 pos_scale: float = 1.0, vel_scale: float = 0.5,
                                 seed: int = 0) -> tuple:
        """
        Generate batch of random initial conditions.

        Parameters
        ----------
        n_trajectories : int
            Number of trajectories
        dim : int
            Spatial dimension
        pos_scale : float
            Scale for positions
        vel_scale : float
            Scale for velocities
        seed : int
            Random seed for reproducibility (default: 0)

        Returns
        -------
        tuple
            Batched initial positions and velocities
        """
        np.random.seed(seed)

        pos_batch = []
        vel_batch = []
        for _ in range(n_trajectories):
            pos, vel = self.random_initial_conditions(dim, pos_scale, vel_scale)
            pos_batch.append(pos)
            vel_batch.append(vel)

        return np.array(pos_batch), np.array(vel_batch)

    def simulate_live(self, pos0: np.ndarray, vel0: np.ndarray,
                      t_end: float, dt: float, interval: int = 20) -> None:
        """
        Simulate with live visualization.

        Parameters
        ----------
        pos0 : np.ndarray
            Initial positions, shape (n, dim)
        vel0 : np.ndarray
            Initial velocities, shape (n, dim)
        t_end : float
            End time
        dt : float
            Time step
        interval : int
            Animation interval in milliseconds
        """
        if pos0.shape[1] != 2:
            raise ValueError("Live visualization only supports 2D simulations")

        n_steps = int(t_end / dt)
        pos, vel = pos0.copy(), vel0.copy()

        # Setup plot
        fig, ax = plt.subplots(figsize=(8, 8))

        # Compute plot limits based on initial positions
        margin = 2.0
        x_min, x_max = pos[:, 0].min() - margin, pos[:, 0].max() + margin
        y_min, y_max = pos[:, 1].min() - margin, pos[:, 1].max() + margin
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)

        # Scale marker sizes by mass
        sizes = 100 * (self.masses / self.masses.max())

        # Initialize scatter plot
        scatter = ax.scatter(pos[:, 0], pos[:, 1], s=sizes, alpha=0.8)

        # Trail lines
        trails = [ax.plot([], [], '-', alpha=0.3, linewidth=1)[0] for _ in range(self.n)]
        trail_length = min(100, n_steps // 10)
        trail_data = [[] for _ in range(self.n)]

        time_text = ax.text(0.02, 0.98, '', transform=ax.transAxes,
                           verticalalignment='top')
        energy_text = ax.text(0.02, 0.93, '', transform=ax.transAxes,
                             verticalalignment='top')

        # Compute initial energy (only for gravitational systems)
        def compute_energy(p, v):
            # Kinetic energy
            ke = 0.5 * np.sum(self.masses[:, None] * v**2)
            # Potential energy (gravitational)
            pe = 0.0
            G = self.force_kwargs.get('G', 1.0)
            softening = self.force_kwargs.get('softening', 1e-2)
            for i in range(self.n):
                for j in range(i + 1, self.n):
                    r = p[j] - p[i]
                    r_norm = np.sqrt(np.sum(r**2) + softening**2)
                    pe -= G * self.masses[i] * self.masses[j] / r_norm
            return ke + pe

        initial_energy = compute_energy(pos, vel)

        def init():
            return scatter, *trails, time_text, energy_text

        def update(frame):
            nonlocal pos, vel

            # Integrate
            pos, vel = self.leapfrog_step(pos, vel, dt)

            # Update scatter
            scatter.set_offsets(pos)

            # Update trails
            for i in range(self.n):
                trail_data[i].append(pos[i].copy())
                if len(trail_data[i]) > trail_length:
                    trail_data[i].pop(0)
                if len(trail_data[i]) > 1:
                    trail_array = np.array(trail_data[i])
                    trails[i].set_data(trail_array[:, 0], trail_array[:, 1])

            # Update text
            time_text.set_text(f'Time: {frame * dt:.2f}')
            current_energy = compute_energy(pos, vel)
            energy_error = abs((current_energy - initial_energy) / initial_energy) * 100
            energy_text.set_text(f'Energy error: {energy_error:.3f}%')

            return scatter, *trails, time_text, energy_text

        anim = FuncAnimation(fig, update, init_func=init, frames=n_steps,
                            interval=interval, blit=True)

        plt.title('N-Body Simulation (Leapfrog Integrator)')
        plt.show()


def demo():
    """Demo: 3-body problem with live visualization."""
    # Setup with default gravity
    masses = np.array([1.0, 1.0, 1.0])
    sim = NBodySimulator(masses, G=1.0, softening=0.1)

    # Initial conditions (figure-8 like configuration)
    pos0 = np.array([
        [-1.0, 0.0],
        [1.0, 0.0],
        [0.0, 0.0]
    ])
    vel0 = np.array([
        [0.3, 0.5],
        [0.3, 0.5],
        [-0.6, -1.0]
    ])

    # Run with live visualization
    sim.simulate_live(pos0, vel0, t_end=20.0, dt=0.01, interval=10)


def demo_custom_force():
    """Demo: Using a custom force function (repulsive force)."""

    def repulsive_force(positions: np.ndarray, masses: np.ndarray,
                       k: float = 1.0, softening: float = 1e-2) -> np.ndarray:
        """Repulsive force proportional to 1/r^2."""
        n = len(masses)
        dim = positions.shape[1]
        acc = np.zeros((n, dim))

        for i in range(n):
            for j in range(n):
                if i != j:
                    r = positions[j] - positions[i]
                    r_norm = np.sqrt(np.sum(r**2) + softening**2)
                    # Repulsive force (negative sign compared to gravity)
                    acc[i] -= k * masses[j] * r / r_norm**3

        return acc

    # Setup with custom force
    masses = np.array([1.0, 1.0, 1.0])
    sim = NBodySimulator(masses, force_fn=repulsive_force, k=2.0, softening=0.1)

    # Initial conditions
    pos0 = np.array([
        [-0.5, 0.0],
        [0.5, 0.0],
        [0.0, 0.5]
    ])
    vel0 = np.array([
        [0.1, 0.2],
        [-0.1, 0.2],
        [0.0, -0.4]
    ])

    sim.simulate_live(pos0, vel0, t_end=10.0, dt=0.01, interval=10)


if __name__ == '__main__':
    demo()
