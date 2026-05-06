# KAN it learn? Interpretable N-Body Physics

We investigate whether Kolmogorov-Arnold Networks (KANs) can be integrated into graph neural networks (GNNs) to produce interpretable models of $n$-body physical systems. We introduce Graph KANs (GKANs), a hybrid architecture that replaces MLP components in message-passing GNNs with learnable spline-based edge functions. We evaluate GKANs on simulated $n$-body dynamics under several force laws and assess both predictive performance and interpretability through sparsity and network visualization. Our results are largely negative: GKANs are consistently harder to optimize than MLP baselines, and standard regularization techniques fail to produce interpretable sparse networks. We attribute this primarily to the interaction between spline-based edge functions and message passing, which amplifies gradient instability. Despite this, we find that physics-informed feature engineering can recover baseline GNN performance with a 99.5\% reduction in model size, and that smoother force laws are substantially easier to learn---suggesting that GKANs may still hold promise under the right inductive biases.

**Please see the [project_report](https://github.com/Hanson-Sun/3body-gkan/tree/main/project_report) folder for the full report.**

## Code

To set up the project, use `uv`
```
uv init
uv sync
```

To run experiments, use
```
uv run run_experiments.py
```

To define/change experiments, modify `experiments.yaml`
