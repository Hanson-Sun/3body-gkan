"""Run all experiments defined in experiments.yaml."""

import argparse
from pathlib import Path

import yaml
from tqdm import tqdm

import generate_training_data
import train_comparison
import visualize_comparison


def main(overwrite: bool = True) -> None:
    with open("experiments.yaml") as f:
        experiments = yaml.safe_load(f)

    for exp in tqdm(experiments.values(), desc="Experiments"):
        name = exp["name"]
        checkpoint_dir = Path("checkpoints") / name
        data_dir       = Path("data")        / name
        output_dir     = Path("outputs")     / name

        for d in (checkpoint_dir, data_dir, output_dir):
            d.mkdir(parents=True, exist_ok=True)

        tqdm.write("\n══════════════════════════════════════════════")
        tqdm.write(f"   Running Experiment: {name}")
        tqdm.write("══════════════════════════════════════════════")

        # Evaluate per-phase skip flags independently
        skip_data  = not overwrite and any(data_dir.iterdir())
        skip_train = not overwrite and any(checkpoint_dir.iterdir())
        skip_vis   = not overwrite and any(output_dir.iterdir())

        if skip_data:
            tqdm.write(f"Data directory '{data_dir}' is not empty — skipping data generation.")
        else:
            tqdm.write("Generating training data...")
            generate_training_data.main(exp["data_params"], str(data_dir))

        if skip_train:
            tqdm.write(f"Checkpoint directory '{checkpoint_dir}' is not empty — skipping training.")
        else:
            tqdm.write("Training models...")
            train_comparison.main(exp["train_params"], str(checkpoint_dir), str(data_dir))

        if skip_vis:
            tqdm.write(f"Output directory '{output_dir}' is not empty — skipping visualization.")
        else:
            tqdm.write("Visualizing results...")
            visualize_comparison.main(
                exp["visualization_params"],
                output_dir=str(output_dir),
                checkpoint_dir=str(checkpoint_dir),
                data_file=str(data_dir / "train.npz"),
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run experiments from experiments.yaml.")
    parser.add_argument(
        "--no-overwrite",
        dest="overwrite",
        action="store_false",
        help="Skip phases whose output directories are already non-empty.",
    )
    args = parser.parse_args()
    main(overwrite=args.overwrite)
