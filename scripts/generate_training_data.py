"""Generate training and validation datasets for N-body dynamics."""

import argparse
from pathlib import Path
from typing import Optional

import numpy as np

import nbody_gkan.nbody_force_fns as nbody_force_fns
from nbody_gkan.data.dataset import create_dataset_from_simulator
from nbody_gkan.nbody import NBodySimulator


FORCE_FN_MAP = {
    "gravity":        nbody_force_fns.gravity,
    "linear_gravity": nbody_force_fns.linear_gravity,
    "cubic_gravity":  nbody_force_fns.cubic_gravity,
    "linear_spring":  nbody_force_fns.linear_spring,
    "hooke_pairwise": nbody_force_fns.hooke_pairwise,
}


def main(yaml_params: Optional[dict] = None, output_dir: Optional[str] = None) -> None:
    parser = argparse.ArgumentParser(description="Generate N-body training data")
    parser.add_argument("--n_bodies",   type=int,   default=3,         help="Number of bodies")
    parser.add_argument("--n_train",    type=int,   default=1000,      help="Training trajectories")
    parser.add_argument("--n_val",      type=int,   default=200,       help="Validation trajectories")
    parser.add_argument("--t_end",      type=float, default=5.0,       help="End time")
    parser.add_argument("--dt",         type=float, default=0.01,      help="Timestep")
    parser.add_argument("--force_fn",   type=str,   default="gravity", help="Force function to use")
    parser.add_argument("--mass_min",   type=float, default=0.5,       help="Minimum body mass")
    parser.add_argument("--mass_max",   type=float, default=2.0,       help="Maximum body mass")
    parser.add_argument("--mass_seed",  type=int,   default=7,         help="Random seed for sampling masses")
    parser.add_argument("--output_dir", type=str,   default="data",    help="Output directory")
    args = parser.parse_args([] if yaml_params is not None else None)

    if yaml_params is not None:
        for key, val in yaml_params.items():
            setattr(args, key, val)
        if output_dir is not None:
            args.output_dir = output_dir

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    force_fn = FORCE_FN_MAP.get(args.force_fn)
    mass_rng = np.random.default_rng(args.mass_seed)
    masses = mass_rng.uniform(args.mass_min, args.mass_max, size=args.n_bodies)
    print(
        f"Using masses sampled in [{args.mass_min:.3f}, {args.mass_max:.3f}] "
        f"with seed {args.mass_seed}: {np.array2string(masses, precision=3)}"
    )

    sim = NBodySimulator(masses, force_fn=force_fn)

    for split, n, seed in [("train", args.n_train, 42), ("val", args.n_val, 123)]:
        print(f"Generating {n} {split} trajectories...")
        create_dataset_from_simulator(
            sim, n, args.t_end, args.dt,
            output_path=out / f"{split}.npz",
            seed=seed,
        )

    print(f"\nDone! Data saved to {out}")


if __name__ == "__main__":
    main()
