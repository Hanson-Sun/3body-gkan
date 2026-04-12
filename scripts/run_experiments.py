"""Run all experiments defined in experiments.yaml."""

import argparse
from pathlib import Path

import yaml
from tqdm.auto import tqdm

import generate_training_data
import train_comparison
import visualize_comparison


def main(overwrite: bool = True, selected: list[str] | None = None) -> None:
    with open("experiments.yaml") as f:
        experiments = yaml.safe_load(f)

    if selected:
        missing = [name for name in selected if name not in experiments]
        if missing:
            available = ", ".join(experiments.keys())
            raise SystemExit(
                f"Unknown experiment(s): {', '.join(missing)}\n"
                f"Available: {available}"
            )
        items = [(k, experiments[k]) for k in selected]
    else:
        items = list(experiments.items())

    for _key, exp in tqdm(items, desc="Experiments"):
        name = exp["name"]
        checkpoint_dir = Path(exp["checkpoint_dir"]) / name if "checkpoint_dir" in exp else Path("checkpoints") / name
        data_dir       = Path(exp["data_dir"]) if "data_dir" in exp else Path("data") / name
        output_dir     = Path(exp["output_dir"]) / name if "output_dir" in exp else Path("outputs") / name

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
        "experiments",
        nargs="*",
        help="Run only these experiment keys (default: all). "
             "E.g.: python run_experiments.py experiment_1a_analytical_min",
    )
    parser.add_argument(
        "--no-overwrite",
        dest="overwrite",
        action="store_false",
        help="Skip phases whose output directories are already non-empty.",
    )
    args = parser.parse_args()
    main(overwrite=args.overwrite, selected=args.experiments or None)
