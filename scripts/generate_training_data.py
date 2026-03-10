"""Generate training and validation datasets for N-body dynamics."""

import argparse
from pathlib import Path
import numpy as np
from nbody_gkan.data.dataset import create_dataset_from_simulator
from nbody_gkan.nbody import NBodySimulator


def main():
    parser = argparse.ArgumentParser(description="Generate N-body training data")
    parser.add_argument("--n_bodies", type=int, default=3, help="Number of bodies")
    parser.add_argument("--n_train", type=int, default=1000, help="Training trajectories")
    parser.add_argument("--n_val", type=int, default=200, help="Validation trajectories")
    parser.add_argument("--t_end", type=float, default=5.0, help="End time")
    parser.add_argument("--dt", type=float, default=0.01, help="Timestep")
    parser.add_argument("--output_dir", type=str, default="data", help="Output directory")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create simulator
    masses = np.ones(args.n_bodies)
    sim = NBodySimulator(masses)

    # Generate training data
    print(f"Generating {args.n_train} training trajectories...")
    create_dataset_from_simulator(
        sim, args.n_train, args.t_end, args.dt,
        output_path=output_dir / "train.npz", seed=42
    )

    # Generate validation data
    print(f"Generating {args.n_val} validation trajectories...")
    create_dataset_from_simulator(
        sim, args.n_val, args.t_end, args.dt,
        output_path=output_dir / "val.npz", seed=123
    )

    print(f"\nDone! Data saved to {output_dir}")


if __name__ == "__main__":
    main()
