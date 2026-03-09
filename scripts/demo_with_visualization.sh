#!/usr/bin/env bash
# Demo with live visualization - compares both models on the SAME test trajectories
# Uses the new compare_models.py script for TRUE fair comparison

set -e  # Exit on error

echo "================================================================"
echo "Demo: Compare Models with Live Visualization"
echo "================================================================"
echo ""
echo "This script trains both models and evaluates them on the"
echo "EXACT SAME test trajectories (guaranteed fair comparison)."
echo ""

# Check if data exists
if [ ! -f "data/train.npz" ] || [ ! -f "data/val.npz" ]; then
    echo "Error: Training data not found!"
    echo "Please run: python scripts/generate_training_data.py"
    exit 1
fi

echo "Found training data ✓"
echo ""

# Fixed parameters for fair comparison
TEST_SEED=42
ROLLOUT_STEPS=100
N_TEST=3
EPOCHS=3
MSG_DIM=
HIDDEN=20

echo "Parameters:"
echo "  Test seed: $TEST_SEED (ensures identical initial conditions)"
echo "  Rollout steps: $ROLLOUT_STEPS"
echo "  Test trajectories: $N_TEST"
echo "  Training epochs: $EPOCHS"
echo "  Hidden dimension: $HIDDEN"
echo ""

echo "================================================================"
echo "Training and Comparing Models"
echo "================================================================"
echo ""
echo "Using compare_models.py to ensure IDENTICAL test trajectories..."
echo ""

uv run python scripts/compare_models.py \
    --train \
    --epochs $EPOCHS \
    --hidden $HIDDEN \
    --rollout_steps $ROLLOUT_STEPS \
    --n_test_trajectories $N_TEST \
    --test_seed $TEST_SEED \
    --live_visualization \
    --output_dir results_comparison

echo ""
echo "================================================================"
echo "Comparison Complete!"
echo "================================================================"
echo ""
echo "Both models were evaluated on IDENTICAL test trajectories (seed=$TEST_SEED)"
echo ""
echo "Results saved to: results_comparison/"
echo ""
echo "Generated files:"
echo "  - model_comparison.png: Side-by-side error comparison"
echo "  - baseline_trajectory_*.png: Baseline GNN predictions"
echo "  - kan_trajectory_*.png: Graph-KAN predictions"
echo "  - kan_edge_functions.png: Learned KAN edge functions"
echo "  - test_indices.npy: Test trajectory indices (for reproducibility)"
echo ""
echo "Next steps:"
echo "1. View model_comparison.png to see which model performs better"
echo "2. Check KAN edge functions to see if 1/r² gravity was learned"
echo "3. Visualize B-splines per feature:"
echo "   python scripts/visualize_kan_splines.py --checkpoint checkpoints/graph_kan_comparison/best.pt"
echo "4. Run with more epochs for better results (edit EPOCHS variable)"
