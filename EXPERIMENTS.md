# Experiments configuration (all tunables)

Use these keys either via CLI flags in `scripts/train_comparison.py` or via `experiments.yaml` (mapped into the same args). Paths are workspace-relative.

## Data
- `train_data`, `val_data`: NPZ files with datasets.
- `checkpoint_dir`: Where to write model checkpoints and plots.

## Baseline GNN (OGN)
- `hidden`: Hidden width for MLP blocks.
- `msg_dim`: Message size for the GNN.
- `train_baseline` (`--train-baseline` / `--no-train-baseline`): Toggle baseline training.

## GraphKAN architecture
- `kan_msg_width`, `kan_node_width`: pykan width specs (must include input/output dims; use `[linear, mult]` to add mult nodes).
- `kan_msg_dim`: Optional override for message output dim (must match `kan_msg_width` output if set).
- `kan_msg_mult_arity`, `kan_node_mult_arity`: Multiplication arity; accept an int or per-layer list-of-lists (e.g., `2` or `[[],[2,3],[]]`).
- `kan_grid_size`: B-spline grid intervals (`grid`).
- `kan_base_fun`: Base residual function (e.g., `identity` for linear init).
- `kan_noise_scale`: Spline noise scale (set `0` with `identity` for exact linear start).
- `kan_scale_base_mu`, `kan_scale_base_sigma`: Mean/std for base-function scale init (set `1,0` for deterministic identity).

## GraphKAN optimization and regularization
- `kan_lbfgs_lr`, `kan_lbfgs_max_iter`: LBFGS learning rate and per-step iterations.
- `kan_lamb_l1`, `kan_lamb_entropy`: PyKAN regularization strengths (edge-forward).
- `kan_adam_warmup_epochs`: Number of Adam warmup epochs before LBFGS.
- `kan_grid_update_freq`, `kan_grid_update_warmup`, `kan_max_grid_updates`: Grid-update scheduling (frequency, start offset, max count).
- `kan_gradient_clip`: Clip value during Adam warmup (`<=0` disables).
- `kan_square_loss`: Loss for KAN training (`false`→L1, `true`→MSE).

## Training loop (shared)
- `epochs`: Total training epochs.
- `batch_size`: Batch size for baseline GNN.
- `kan_batch_size`: Batch size for GraphKAN.
- `lr`: Adam learning rate (baseline and KAN warmup).
- `lamb`: Overall loss penalty term (passed to trainers).

## Notes
- YAML keys mirror these names but are grouped as `gnn_hp`, `gkan_hp`, and `training_hp` inside an experiment block; `data_params` can override data paths/counts when used by generators.
- Width specs must match required input/output dims: msg net starts with `2 * n_f`, node net starts with `n_f + msg_dim` and ends with dataset dim.
