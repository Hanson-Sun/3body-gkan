'''This should run experiments from experiments.yaml'''

import yaml
import generate_training_data
import train_comparison
import visualize_comparison
import pathlib
from tqdm import tqdm

def main(overwrite: bool = True):
    with open("experiments.yaml", "r") as f:
        experiments = yaml.safe_load(f)

    for exp_key, exp in tqdm(experiments.items()):
        tqdm.write("")
        tqdm.write("══════════════════════════════════════════════")
        tqdm.write(f"   Running Experiment: {exp['name']}")
        tqdm.write("══════════════════════════════════════════════")

        checkpoint_dir = f"checkpoints/{exp['name']}"
        data_dir = f"data/{exp['name']}"
        output_dir = f"outputs/{exp['name']}"

        pathlib.Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
        pathlib.Path(data_dir).mkdir(parents=True, exist_ok=True)
        pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)

        if not overwrite:
            if any(pathlib.Path(checkpoint_dir).iterdir()):
                tqdm.write(f"Checkpoint directory {checkpoint_dir} is not empty, skipping training.")
                continue
            if any(pathlib.Path(data_dir).iterdir()):
                tqdm.write(f"Data directory {data_dir} is not empty, skipping data generation.")
                continue
            if any(pathlib.Path(output_dir).iterdir()):
                tqdm.write(f"Output directory {output_dir} is not empty, skipping visualization.")
                continue

        tqdm.write("Generating training data...")
        generate_training_data.main(exp['data_params'], data_dir)
        tqdm.write("Training models...")
        train_comparison.main(exp['train_params'], checkpoint_dir, data_dir)
        tqdm.write("Visualizing results...")
        visualize_comparison.main(
            exp['visualization_params'],
            output_dir,
            checkpoint_dir,
            data_dir
        )

if __name__ == "__main__":
    main()
