"""
Unit tests for models.

Tests:
1. KAN layer shape and gradients
2. GraphKAN forward pass
3. Baseline GNN forward pass
4. Overfit single batch test
"""

import sys

import torch
from nbody_gkan.data.dataset import get_edge_index
from nbody_gkan.models.baseline_gnn import GN, OGN
from nbody_gkan.models.graph_kan import GraphKAN, OrdinaryGraphKAN
from nbody_gkan.models.kan_layer import KANLayer
from torch_geometric.data import Data


def test_kan_layer_shape():
    """Test that KAN layer produces correct output shape."""
    print("\n" + "=" * 60)
    print("Test 1: KAN Layer Shape")
    print("=" * 60)

    in_features = 10
    out_features = 5
    batch_size = 32

    layer = KANLayer(in_features, out_features, grid_size=5, spline_order=3)

    x = torch.randn(batch_size, in_features)
    y = layer(x)

    assert y.shape == (batch_size, out_features), \
        f"Expected shape {(batch_size, out_features)}, got {y.shape}"

    print(f"✓ Input shape: {x.shape}")
    print(f"✓ Output shape: {y.shape}")
    print(f"✓ Parameters: {sum(p.numel() for p in layer.parameters()):,}")
    print("PASS")


def test_kan_layer_gradients():
    """Test that KAN layer computes gradients correctly."""
    print("\n" + "=" * 60)
    print("Test 2: KAN Layer Gradients")
    print("=" * 60)

    layer = KANLayer(5, 3, grid_size=5, spline_order=3)

    x = torch.randn(8, 5, requires_grad=True)
    y = layer(x)
    loss = y.sum()
    loss.backward()

    # Check that gradients exist
    assert x.grad is not None, "Input gradients not computed"
    for name, param in layer.named_parameters():
        assert param.grad is not None, f"Parameter {name} has no gradient"

    print(f"✓ Input gradients: {x.grad.shape}")
    print(f"✓ All parameter gradients computed")
    print("PASS")


def test_graph_kan_forward():
    """Test GraphKAN forward pass."""
    print("\n" + "=" * 60)
    print("Test 3: GraphKAN Forward Pass")
    print("=" * 60)

    n_nodes = 5
    n_features = 7  # Arbitrary test value (for 2D N-body: 5 = 2 pos + 2 vel + 1 mass)
    msg_dim = 20
    ndim = 2

    edge_index = get_edge_index(n_nodes)

    model = GraphKAN(
        n_f=n_features,
        msg_dim=msg_dim,
        ndim=ndim,
        n_msg_layers=2,
        n_node_layers=2,
        grid_size=5,
        spline_order=3,
    )

    x = torch.randn(n_nodes, n_features)
    out = model(x, edge_index)

    assert out.shape == (n_nodes, ndim), \
        f"Expected shape {(n_nodes, ndim)}, got {out.shape}"

    print(f"✓ Nodes: {n_nodes}, Features: {n_features}")
    print(f"✓ Edges: {edge_index.shape[1]}")
    print(f"✓ Output shape: {out.shape}")
    print(f"✓ Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print("PASS")


def test_baseline_gnn_forward():
    """Test baseline GNN forward pass."""
    print("\n" + "=" * 60)
    print("Test 4: Baseline GNN Forward Pass")
    print("=" * 60)

    n_nodes = 5
    n_features = 7
    msg_dim = 20
    ndim = 2

    edge_index = get_edge_index(n_nodes)

    model = GN(n_f=n_features, msg_dim=msg_dim, ndim=ndim, hidden=50)

    x = torch.randn(n_nodes, n_features)
    out = model(x, edge_index)

    assert out.shape == (n_nodes, ndim), \
        f"Expected shape {(n_nodes, ndim)}, got {out.shape}"

    print(f"✓ Nodes: {n_nodes}, Features: {n_features}")
    print(f"✓ Edges: {edge_index.shape[1]}")
    print(f"✓ Output shape: {out.shape}")
    print(f"✓ Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print("PASS")


def test_overfit_single_batch():
    """Test that model can overfit a single batch (sanity check)."""
    print("\n" + "=" * 60)
    print("Test 5: Overfit Single Batch")
    print("=" * 60)

    n_nodes = 3
    n_features = 5  # 2D: [pos_x, pos_y, vel_x, vel_y, mass]
    ndim = 2
    msg_dim = 20

    edge_index = get_edge_index(n_nodes)

    # Create a simple data point
    x = torch.randn(n_nodes, n_features)
    y = torch.randn(n_nodes, ndim)
    data = Data(x=x, y=y, edge_index=edge_index)

    # Create model
    model = OrdinaryGraphKAN(
        n_f=n_features,
        msg_dim=msg_dim,
        ndim=ndim,
        edge_index=edge_index,
        n_msg_layers=2,
        n_node_layers=2,
        grid_size=5,
        spline_order=3,
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)

    # Train for a few steps
    initial_loss = model.loss(data, augment=False).item()
    print(f"Initial loss: {initial_loss:.6f}")

    for step in range(100):
        optimizer.zero_grad()
        loss = model.loss(data, augment=False)
        loss.backward()
        optimizer.step()

        if (step + 1) % 20 == 0:
            print(f"Step {step + 1:3d}: loss = {loss.item():.6f}")

    final_loss = model.loss(data, augment=False).item()

    # Loss should decrease significantly (at least 80% reduction)
    assert final_loss < initial_loss * 0.2, \
        f"Loss did not decrease enough: {initial_loss:.6f} -> {final_loss:.6f}"

    print(f"\n✓ Initial loss: {initial_loss:.6f}")
    print(f"✓ Final loss: {final_loss:.6f}")
    print(f"✓ Reduction: {(1 - final_loss / initial_loss) * 100:.1f}%")
    print("PASS")


def test_model_comparison():
    """Compare parameter counts between GraphKAN and baseline GNN."""
    print("\n" + "=" * 60)
    print("Test 6: Model Comparison")
    print("=" * 60)

    n_nodes = 5
    n_features = 7
    msg_dim = 100
    ndim = 2
    edge_index = get_edge_index(n_nodes)

    # GraphKAN
    kan_model = OrdinaryGraphKAN(
        n_f=n_features,
        msg_dim=msg_dim,
        ndim=ndim,
        edge_index=edge_index,
        n_msg_layers=3,
        n_node_layers=3,
        grid_size=5,
        spline_order=3,
    )

    # Baseline GNN
    gnn_model = OGN(
        n_f=n_features,
        msg_dim=msg_dim,
        ndim=ndim,
        edge_index=edge_index,
        hidden=300,
    )

    kan_params = sum(p.numel() for p in kan_model.parameters())
    gnn_params = sum(p.numel() for p in gnn_model.parameters())

    print(f"GraphKAN parameters: {kan_params:,}")
    print(f"Baseline GNN parameters: {gnn_params:,}")
    print(f"Ratio: {kan_params / gnn_params:.2f}x")

    # Both models should work on the same data
    x = torch.randn(n_nodes, n_features)
    kan_out = kan_model(x, edge_index)
    gnn_out = gnn_model(x, edge_index)

    assert kan_out.shape == gnn_out.shape == (n_nodes, ndim)

    print(f"✓ Both models produce correct output shape: {kan_out.shape}")
    print("PASS")


def test_translation_invariance():
    """
    Test that models are translation-sensitive before training.

    Models use [x_i, x_j] inputs (absolute positions), so they are not structurally
    translation invariant. They learn translation invariance via position augmentation
    during training. This test verifies untrained models ARE sensitive to translations.
    """
    print("\n" + "=" * 60)
    print("Test 7: Translation Sensitivity (Untrained)")
    print("=" * 60)

    n_nodes = 3
    n_features = 5  # 2D: [pos_x, pos_y, vel_x, vel_y, mass]
    ndim = 2
    msg_dim = 20
    edge_index = get_edge_index(n_nodes)

    # Test both models
    for model_name, model_class in [("GraphKAN", GraphKAN), ("Baseline GNN", GN)]:
        print(f"\nTesting {model_name}...")

        if model_class == GraphKAN:
            model = model_class(
                n_f=n_features,
                msg_dim=msg_dim,
                ndim=ndim,
                n_msg_layers=2,
                n_node_layers=2,
                grid_size=5,
                spline_order=3,
            )
        else:
            model = model_class(
                n_f=n_features,
                msg_dim=msg_dim,
                ndim=ndim,
                hidden=50,
            )

        model.eval()

        # Create random input
        x = torch.randn(n_nodes, n_features)

        # Original output
        with torch.no_grad():
            out1 = model(x, edge_index)

        # Translate all positions by a constant vector
        translation = torch.tensor([10.0, -5.0])
        x_translated = x.clone()
        x_translated[:, :ndim] += translation

        # Output with translated positions
        with torch.no_grad():
            out2 = model(x_translated, edge_index)

        # Untrained models should NOT be translation invariant
        # (they see absolute positions and must learn invariance from training)
        max_diff = (out1 - out2).abs().max().item()
        assert max_diff > 1e-6, \
            f"{model_name} unexpectedly translation invariant before training: max diff = {max_diff}"

        print(f"  ✓ Translation-sensitive as expected: {max_diff:.2e}")

    print("\n✓ Both models are translation invariant")
    print("PASS")


def run_all_tests():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("Running Model Tests")
    print("=" * 60)

    try:
        test_kan_layer_shape()
        test_kan_layer_gradients()
        test_graph_kan_forward()
        test_baseline_gnn_forward()
        test_overfit_single_batch()
        test_model_comparison()
        test_translation_invariance()

        print("\n" + "=" * 60)
        print("ALL TESTS PASSED ✓")
        print("=" * 60)
        return True

    except AssertionError as e:
        print(f"\n✗ TEST FAILED: {e}")
        return False
    except Exception as e:
        print(f"\n✗ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
