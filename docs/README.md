# Graph-KAN for N-Body Physics Discovery

## Overview

This project applies **Graph Kolmogorov-Arnold Networks (Graph-KANs)** to the n-body problem with the goal of
discovering Newton's law of universal gravitation in an interpretable way. We replace the Multi-Layer Perceptrons (MLPs)
in a Graph Neural Network with Kolmogorov-Arnold Network (KAN) layers, which learn univariate spline functions that can
be directly visualized and interpreted.

**Key Research Question:** Can KANs enable more interpretable physics discovery than standard GNNs?

## Background & Motivation

### The Baseline: GNNs for Symbolic Physics

Our work builds on the
paper ["Discovering Symbolic Models from Deep Learning with Inductive Biases"](https://arxiv.org/abs/2006.11287) (
Cranmer et al.), which demonstrated that Graph Neural Networks can learn physical laws like F = Gm₁m₂/r² from trajectory
data. Their approach:

1. **Train a GNN** to predict accelerations from positions/velocities/masses
2. **Analyze learned messages** between nodes (representing pairwise forces)
3. **Apply symbolic regression (PySR)** to extract closed-form equations

**Limitation:** The learned representations in MLP-based GNNs are opaque. You need a separate symbolic regression step
to interpret what the network learned.

### Our Innovation: KANs for Direct Interpretability

**Kolmogorov-Arnold Networks (KANs)** replace the traditional `Linear → Activation` pattern with learnable univariate
B-spline functions on each connection. Key advantages:

- **Direct visualization:** Each learned spline can be plotted (e.g., does it look like 1/r²?)
- **Univariate structure:** Forces decomposition into simpler functions
- **Theoretically grounded:** Based on Kolmogorov-Arnold representation theorem

**The Dream:** Train the Graph-KAN, plot the learned splines, and directly see "this function looks like 1/r²" without
needing symbolic regression.

## Architecture

### Message Passing Framework

Both the baseline GNN and our Graph-KAN use the same message passing structure:

```
For each edge (i → j):
  1. message() : Compute message from node j to node i
  2. aggregate(): Sum messages (respects force superposition)
  3. update()  : Predict acceleration for node i

Node features: [position, velocity, mass]
```

This naturally maps to n-body physics:

- **Edges** = pairwise gravitational interactions
- **Messages** = forces between bodies
- **Aggregation (sum)** = force superposition principle
- **Update output** = acceleration (F/m)

### The Only Difference: MLPs vs KANs

| Component        | Baseline GNN                   | Graph-KAN                      |
|------------------|--------------------------------|--------------------------------|
| Message function | `[x_i, x_j] → MLP → messages`  | `[x_i, x_j] → KAN → messages`  |
| Update function  | `[x, aggr_msgs] → MLP → accel` | `[x, aggr_msgs] → KAN → accel` |
| Aggregation      | Sum                            | Sum                            |
| Loss             | L1 on accelerations            | L1 on accelerations            |

**MLP structure:**

```
Linear(14 → 300) → ReLU →
Linear(300 → 300) → ReLU →
Linear(300 → 300) → ReLU →
Linear(300 → 100)
```

**KAN structure:**

```
KANLayer(14 → 300) →
KANLayer(300 → 300) →
KANLayer(300 → 100)
```

Each `KANLayer` learns a B-spline function for every input-output connection, making the transformation interpretable.

## Code Structure

```
nbody_gkan/                    # Main package
├── models/
│   ├── kan_layer.py          # Core KAN layer with B-splines
│   ├── graph_kan.py           # Graph-KAN architecture
│   ├── baseline_gnn.py        # Baseline MLP-based GNN (for comparison)
│   └── ordinary_mixin.py      # Position augmentation & loss
├── training/
│   └── trainer.py             # Training loop with checkpointing
├── data/
│   └── dataset.py             # PyG Dataset for n-body trajectories
├── nbody.py                   # N-body simulator
└── device.py                  # Device management utilities

scripts/
├── generate_training_data.py  # Generate n-body trajectories
└── train.py                   # Train models

tests/                         # Comprehensive test suite
symbolic_deep_learning/        # Original baseline code from paper
```

### Key Files Explained

#### `nbody_gkan/models/kan_layer.py`

Implements the KAN layer using B-splines:

```python
class KANLayer(nn.Module):
# For each (input_i, output_j) connection:
#   output_j = base_activation(input_i) * weight_ij  # Residual path
#            + Σ spline_coeff[k] * B_k(input_i)      # Spline path
```

**Key design choices:**

- **B-splines of order 3** (cubic): Smooth, twice differentiable
- **Grid range [-5, 5]**: Wide enough for raw physical inputs
- **No normalization**: Preserves interpretability (splines learn functions of physical quantities directly)
- **Base activation (SiLU)**: Residual path for multiplicative interactions

#### `nbody_gkan/models/graph_kan.py`

The Graph-KAN architecture:

```python
class GraphKAN(MessagePassing):
    def message(self, x_i, x_j):
        # Concatenate all features from both nodes
        return self.msg_fnc(torch.cat([x_i, x_j], dim=1))

    def update(self, aggr_out, x):
        # Concatenate node features with aggregated messages
        return self.node_fnc(torch.cat([x, aggr_out], dim=1))
```

**Important:** We deliberately match the baseline's input structure (`[x_i, x_j]`) rather than engineering features like
`[dr, |dr|, masses]`. This:

1. Ensures fair comparison
2. Lets the network learn what's relevant
3. Maintains simplicity

#### `nbody_gkan/models/ordinary_mixin.py`

Adds training-specific functionality:

```python
class OrdinaryMixin:
    def just_derivative(self, g, augment=False):
        # Apply position augmentation if training
        # Return predicted accelerations

    def loss(self, g, augment=True, square=False):
        # L1 or L2 loss on accelerations
```

**Position augmentation:** Randomly translates all positions by the same offset during training. This teaches the
network translation invariance (forces depend on relative positions, not absolute coordinates).

#### `nbody_gkan/nbody.py`

Vectorized n-body simulator using Runge-Kutta integration:

- Computes gravitational forces: `F = G * m_i * m_j / (r² + ε²)`
- Uses softening parameter `ε` to prevent singularities
- Supports adaptive timesteps, energy tracking, visualization

#### `nbody_gkan/data/dataset.py`

PyTorch Geometric dataset for n-body trajectories:

```python
data = Data(
    x=node_features,     # [pos, vel, mass]
    y=accelerations,     # Ground truth
    edge_index=edges,    # Fully connected graph
)
```

Automatically computes ground truth accelerations from positions and masses.

## Key Design Decisions

### 1. Matching the Baseline Architecture

We deliberately match the baseline paper's architecture exactly (input structure, number of layers, hidden dimensions)
with the *only* difference being KAN layers vs MLPs. This ensures:

- **Fair comparison:** Differences in performance are due to KANs, not other choices
- **Reproducibility:** Easy to ablate and understand what matters
- **Simplicity:** Minimal moving parts

### 2. Translation Invariance is Learned, Not Structural

The network sees **absolute positions** `[x_i, x_j]`, not relative positions `[dr]`. It must learn that forces depend
only on `r = x_j - x_i` through:

- **Position augmentation** during training (random global translations)
- **Network capacity** to discover the invariance

This is consistent with the baseline paper's approach.

### 3. Narrow vs Wide KANs

KANs are parameter-dense (each connection has ~8 spline coefficients). We provide a tunable `hidden` parameter:

```python
# Narrow KAN (conventional wisdom for KANs)
model = OrdinaryGraphKAN(..., hidden=None)  # Uses input_dim

# Wide KAN (match baseline capacity)
model = OrdinaryGraphKAN(..., hidden=300)
```

Start narrow and widen as needed during experimentation.

### 4. No Feature Engineering

Unlike some approaches that hand-engineer inputs like `[dr, |dr|, masses]`, we use raw `[x_i, x_j]` (all node features).
This:

- Matches the baseline
- Tests whether KANs can learn relevant features
- Avoids biasing the network toward specific functional forms

## Getting Started

### 1. Generate Training Data

```bash
uv run python scripts/generate_training_data.py \
    --n_train 1000 \
    --n_val 200 \
    --n_bodies 3 \
    --t_end 5.0 \
    --dt 0.01 \
    --output_dir data/
```

This creates `data/train.npz` and `data/val.npz` with n-body trajectories.

### 2. Train a Model

```bash
# Train Graph-KAN
uv run python scripts/train.py \
    --model graph_kan \
    --hidden_dim 300 \
    --epochs 100 \
    --msg_dim 100

# Train baseline GNN
uv run python scripts/train.py \
    --model baseline_gnn \
    --hidden_dim 300 \
    --epochs 100 \
    --msg_dim 100
```

### 3. Run Tests

```bash
uv run pytest tests/ -v
```

Comprehensive test suite covering:

- B-spline properties (partition of unity, smoothness)
- Model forward passes
- Training loop
- Translation sensitivity
- Dataset generation

## Relationship to Baseline Paper

| Aspect               | Baseline Paper            | Our Implementation                    |
|----------------------|---------------------------|---------------------------------------|
| **Core idea**        | GNN for physics discovery | Same + KANs for interpretability      |
| **Architecture**     | MLP-based GNN             | KAN-based GNN                         |
| **Message passing**  | `[x_i, x_j] → MLP → msg`  | `[x_i, x_j] → KAN → msg`              |
| **Training**         | L1 loss + position aug    | Same                                  |
| **Interpretability** | Messages → PySR           | Direct spline visualization (planned) |
| **Code location**    | `symbolic_deep_learning/` | `src/models/graph_kan.py`             |

We maintain a copy of their code in `symbolic_deep_learning/` for reference and reproducibility.

## Current Status & Next Steps

### ✅ Completed

- [x] KAN layer implementation with B-splines
- [x] Graph-KAN architecture matching baseline structure
- [x] Training pipeline with position augmentation
- [x] Comprehensive test suite
- [x] Interface parity with baseline GNN

### 🚧 In Progress / TODO

- [ ] Train on multi-body systems and evaluate
- [ ] Spline visualization utilities
- [ ] Extract learned functions for interpretation
- [ ] Compare interpretability: KAN splines vs symbolic regression
- [ ] Grid refinement for sharper learned functions
- [ ] Experiment with different spline orders/grid sizes

## References

- **Original paper:
  ** [Discovering Symbolic Models from Deep Learning with Inductive Biases](https://arxiv.org/abs/2006.11287) (Cranmer
  et al., 2020)
- **KANs:** [KAN: Kolmogorov-Arnold Networks](https://arxiv.org/abs/2404.19756) (Liu et al., 2024)
- **PyTorch Geometric:** [Documentation](https://pytorch-geometric.readthedocs.io/)

## Questions?

Key questions to explore when reading the code:

1. **Why KANs for physics?** See "Background & Motivation" section
2. **How does message passing work?** See "Architecture" section
3. **What's in a KAN layer?** Read `src/models/kan_layer.py` + docstrings
4. **How does training work?** See `src/training/trainer.py` and `scripts/train.py`
5. **How do we ensure fair comparison?** See "Key Design Decisions" section

Welcome to the project! 🚀