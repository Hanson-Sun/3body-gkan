"""
Generate training and validation datasets for N-body dynamics.

Usage:
    python scripts/generate_training_data.py --n_train 1000 --n_val 200
    python scripts/generate_training_data.py --n_bodies 5 --preset random
"""

import argparse
from pathlib import Path

import numpy as np
from nbody_gkan.data.dataset import create_dataset_from_simulator
from nbody_gkan.nbody import NBodySimulator


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate N-body training and validation data"
    )

    # System parameters
    parser.add_argument(
        "--n_bodies", type=int, default=3, help="Number of bodies"
    )
    parser.add_argument(
        "--dim", type=int, default=2, help="Spatial dimension (2 or 3)"
    )

    # Data generation
    parser.add_argument(
        "--n_train", type=int, default=1000, help="Number of training trajectories"
    )
    parser.add_argument(
        "--n_val", type=int, default=200, help="Number of validation trajectories"
    )
    parser.add_argument(
        "--t_end", type=float, default=5.0, help="End time for each trajectory"
    )
    parser.add_argument(
        "--dt", type=float, default=0.01, help="Simulation timestep"
    )
    parser.add_argument(
        "--save_every", type=int, default=1, help="Save every k-th step"
    )

    # Initial conditions
    parser.add_argument(
        "--preset",
        type=str,
        default="random",
        choices=["random", "circular", "figure8", "solar"],
        help="Initial condition preset",
    )
    parser.add_argument(
        "--pos_scale", type=float, default=1.0, help="Position scale"
    )
    parser.add_argument(
        "--vel_scale", type=float, default=0.5, help="Velocity scale"
    )
    parser.add_argument(
        "--no_zero_momentum", action="store_true", default=False,
        help="Disable zero net momentum (enabled by default)"
    )

    # Physics
    parser.add_argument(
        "--G", type=float, default=1.0, help="Gravitational constant"
    )
    parser.add_argument(
        "--softening", type=float, default=1e-2, help="Softening parameter"
    )

    # Output
    parser.add_argument(
        "--output_dir", type=str, default="data", help="Output directory"
    )
    parser.add_argument(
        "--train_file", type=str, default="train.npz", help="Training data filename"
    )
    parser.add_argument(
        "--val_file", type=str, default="val.npz", help="Validation data filename"
    )

    # Seeds
    parser.add_argument(
        "--train_seed", type=int, default=42, help="Training data seed"
    )
    parser.add_argument(
        "--val_seed", type=int, default=123, help="Validation data seed"
    )

    return parser.parse_args()


def main():
    """Main data generation function."""
    args = parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("N-Body Data Generation")
    print("=" * 60)
    print(f"System: {args.n_bodies} bodies in {args.dim}D")
    print(f"Physics: G={args.G}, softening={args.softening}")
    print(f"Training trajectories: {args.n_train}")
    print(f"Validation trajectories: {args.n_val}")
    print(f"Time span: [0, {args.t_end}] with dt={args.dt}")
    print(f"Initial conditions: {args.preset}, pos_scale={args.pos_scale}, vel_scale={args.vel_scale}")
    print(f"Output directory: {output_dir}")
    print("=" * 60)

    # Set masses (all equal for simplicity)
    masses = np.ones(args.n_bodies)

    # Create simulator
    sim = NBodySimulator(
        n=args.n_bodies,
        dim=args.dim,
        masses=masses,
        force_kwargs={"G": args.G, "softening": args.softening},
    )

    print(f"\nSimulator: {sim}")

    # Generate training data
    print(f"\n[1/2] Generating training data...")
    train_path = output_dir / args.train_file
    create_dataset_from_simulator(
        sim=sim,
        n_trajectories=args.n_train,
        t_end=args.t_end,
        dt=args.dt,
        save_every=args.save_every,
        output_path=train_path,
        seed=args.train_seed,
        preset=args.preset,
        pos_scale=args.pos_scale,
        vel_scale=args.vel_scale,
        zero_momentum=not args.no_zero_momentum,
    )

    # Generate validation data
    print(f"\n[2/2] Generating validation data...")
    val_path = output_dir / args.val_file
    create_dataset_from_simulator(
        sim=sim,
        n_trajectories=args.n_val,
        t_end=args.t_end,
        dt=args.dt,
        save_every=args.save_every,
        output_path=val_path,
        seed=args.val_seed,
        preset=args.preset,
        pos_scale=args.pos_scale,
        vel_scale=args.vel_scale,
        zero_momentum=not args.no_zero_momentum,
    )

    print("\n" + "=" * 60)
    print("Data generation complete!")
    print("=" * 60)
    print(f"Training data: {train_path}")
    print(f"Validation data: {val_path}")

    # Summary statistics
    train_data = np.load(train_path)
    print(f"\nDataset statistics:")
    print(f"  Training samples: {train_data['positions'].shape[0] * train_data['positions'].shape[1]}")
    print(f"  Training shape: {train_data['positions'].shape}")
    print(f"  Validation samples: {args.n_val * train_data['positions'].shape[1]}")
    print(f"  Node features: {2 * args.dim + 1} (pos, vel, mass)")
    print(f"  Target dimension: {args.dim} (accelerations)")


if __name__ == "__main__":
    main()
