# NBody with Graph KANs 

Learn symbolic relationship between n-body systems from tracjectory data, utilizing a Graph KAN architecture. The goal is to extract interpretable, symbolic equations that govern the dynamics of the system, and to demonstrate that these equations generalize better than black-box models.

We can compare this with previous research on symbolic regression from deep learning, which typically uses MLP-based GNNs and a separate symbolic regression step. The key novelty here is that the KAN architecture allows us to directly learn symbolic functions without needing a separate regression step, which should improve interpretability and potentially generalization.

So there is no guarantee that it will be better but hopefully we can gain some interesting insights about the tradeoffs between interpretability and performance in this specific application. We can also compare the performance of the Graph KAN with other physics-informed GNN architectures, such as Hamiltonian GNNs or momentum-conserving GNNs, to see if the KAN architecture provides any advantages in terms of accuracy or generalization.

## Tools

So we can pick a KAN library
- pykan (classic, Python, not super optimized, **has symbolic regression built in**)
- efficient-kan (C++, optimized for speed, but less user-friendly)
- GraphKAN (Python, optimized for graph data, but less mature)
- KAGNN (might be the goat?)

it "should" be not too hard to roll our own GKAN implementation with pytorch + efficient-kan.

Ideally have GPU accelerated pytorch or else this will be slow af to train...

# TODO 

- [x] n body simulation data generation (allow us to specify the force law between trajectories)
    - there might be some stability issues with close interactions, might be problematic
- [x] n body simulation visualizer
- [x] convert trajectory data into graph format for GNN
- [x] GNN baseline implementation and training
- [x] Graph KAN implementation and training
- [x] Implement sparse initialization for the KAN layers
- [x] Add multiply layer to KAN architecture (copy MultKan from pykan)
- [x] design experiments to compare generalization performance of GNN vs Graph KAN
  - [x] Test some feature engineering approaches
  - [x] Try with minimal model complexity
  - [x] Try with more complex model and see if sparsity works
  - [x] Look at some approaches from KAN 2.0 paper
- [x] interpretability analysis of learned functions

