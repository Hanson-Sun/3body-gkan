"""Generate training and validation datasets for N-body dynamics."""
from typing import Optional
import argparse
from pathlib import Path
import numpy as np
from nbody_gkan.data.dataset import create_dataset_from_simulator
from nbody_gkan.nbody import NBodySimulator
import nbody_gkan.nbody_force_fns as nbody_force_fns


def main(yaml_params: Optional[dict] = None, output_dir: Optional[str] = None):
    parser = argparse.ArgumentParser(description="Generate N-body training data")
    parser.add_argument("--n_bodies", type=int, default=3, help="Number of bodies")
    parser.add_argument("--n_train", type=int, default=1000, help="Training trajectories")
    parser.add_argument("--n_val", type=int, default=200, help="Validation trajectories")
    parser.add_argument("--t_end", type=float, default=5.0, help="End time")
    parser.add_argument("--dt", type=float, default=0.01, help="Timestep")
    parser.add_argument("--force_fn", type=str, default="gravity", help="Force function to use")
    parser.add_argument("--output_dir", type=str, default="data", help="Output directory")
    args = parser.parse_args([] if yaml_params is not None else None)

    if yaml_params is not None:
        args.n_bodies = yaml_params.get("n_bodies", args.n_bodies)
        args.n_train = yaml_params.get("n_train", args.n_train)
        args.n_val = yaml_params.get("n_val", args.n_val)
        args.t_end = yaml_params.get("t_end", args.t_end)
        args.dt = yaml_params.get("dt", args.dt)
        args.force_fn = yaml_params.get("force_fn", args.force_fn)
        args.output_dir = output_dir if output_dir is not None else args.output_dir

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    force_fn_map = {
        "gravity": nbody_force_fns.gravity,
        "linear_gravity": nbody_force_fns.linear_gravity,
        "cubic_gravity": nbody_force_fns.cubic_gravity,
        "linear_spring": nbody_force_fns.linear_spring
    } 
    

    # Create simulator
    masses = np.ones(args.n_bodies)
    sim = NBodySimulator(masses, force_fn=force_fn_map.get(args.force_fn))

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