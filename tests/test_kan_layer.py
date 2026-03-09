"""
Tests for B-spline KAN layer correctness.
"""

import numpy as np
import torch
from nbody_gkan.models import KANLayer


def test_bspline_dimension():
    """Test that B-spline basis has correct number of functions without padding."""
    print("\n" + "=" * 60)
    print("Test: B-spline Dimension Correctness")
    print("=" * 60)

    # Test various grid sizes and spline orders
    test_configs = [
        (5, 3),  # Default configuration
        (3, 2),  # Quadratic splines
        (5, 4),  # Quartic splines
        (10, 3),  # Finer grid
    ]

    for grid_size, spline_order in test_configs:
        in_features = 4
        out_features = 3
        batch_size = 16

        layer = KANLayer(
            in_features=in_features,
            out_features=out_features,
            grid_size=grid_size,
            spline_order=spline_order
        )

        n_coeffs = grid_size + spline_order
        n_knots = len(layer.grid)
        expected_knots = n_coeffs + spline_order

        print(f"\nGrid size: {grid_size}, Spline order: {spline_order}")
        print(f"  Expected n_coeffs: {n_coeffs}")
        print(f"  Expected n_knots: {expected_knots}")
        print(f"  Actual n_knots: {n_knots}")

        assert n_knots == expected_knots, (
            f"Knot count mismatch: expected {expected_knots}, got {n_knots}"
        )

        # Test forward pass doesn't trigger assertion
        x = torch.randn(batch_size, in_features)
        try:
            output = layer(x)
            print(f"  ✓ Forward pass successful, output shape: {output.shape}")
        except AssertionError as e:
            print(f"  ✗ Forward pass failed: {e}")
            raise

        assert output.shape == (batch_size, out_features)

    print("\n✓ All dimension tests passed")


def test_partition_of_unity():
    """Test that B-spline basis functions sum to 1 (partition of unity property)."""
    print("\n" + "=" * 60)
    print("Test: B-spline Partition of Unity")
    print("=" * 60)

    in_features = 5
    out_features = 3
    grid_size = 5
    spline_order = 3

    layer = KANLayer(
        in_features=in_features,
        out_features=out_features,
        grid_size=grid_size,
        spline_order=spline_order
    )

    # Test at various points in [-1, 1]
    test_points = torch.linspace(-0.99, 0.99, 50).unsqueeze(-1).expand(-1, in_features)

    # Compute basis functions directly using the private method
    with torch.no_grad():
        bases = layer._compute_bspline_basis(test_points)

    # Sum across basis functions (last dimension)
    basis_sums = bases.sum(dim=-1)  # Shape: (n_points, in_features)

    # Check if sum is close to 1 everywhere
    expected_sum = torch.ones_like(basis_sums)
    max_error = (basis_sums - expected_sum).abs().max().item()
    mean_error = (basis_sums - expected_sum).abs().mean().item()

    print(f"Max error from 1.0: {max_error:.6e}")
    print(f"Mean error from 1.0: {mean_error:.6e}")

    # Allow small numerical errors
    assert max_error < 1e-5, f"Partition of unity violated: max error {max_error}"

    print("✓ Partition of unity property verified")


def test_bspline_smoothness():
    """Test that B-spline basis functions are smooth (continuous derivatives)."""
    print("\n" + "=" * 60)
    print("Test: B-spline Smoothness")
    print("=" * 60)

    in_features = 3
    out_features = 2
    grid_size = 5
    spline_order = 3  # Cubic splines should have continuous 1st and 2nd derivatives

    layer = KANLayer(
        in_features=in_features,
        out_features=out_features,
        grid_size=grid_size,
        spline_order=spline_order
    )

    # Test points densely sampled
    x = torch.linspace(-0.99, 0.99, 200, requires_grad=True).unsqueeze(-1).expand(-1, in_features)

    # Forward pass
    output = layer(x)

    # Compute gradients
    grad_outputs = torch.ones_like(output)
    gradients = torch.autograd.grad(
        outputs=output,
        inputs=x,
        grad_outputs=grad_outputs,
        create_graph=True,
        retain_graph=True
    )[0]

    print(f"Output shape: {output.shape}")
    print(f"Gradient shape: {gradients.shape}")
    print(f"Gradient range: [{gradients.min().item():.3f}, {gradients.max().item():.3f}]")

    # Check that gradients exist and are finite
    assert torch.isfinite(gradients).all(), "Non-finite gradients detected"

    print("✓ Smoothness test passed (gradients are finite)")


def test_bspline_vs_scipy():
    """Compare B-spline computation against scipy.interpolate.BSpline (if available)."""
    print("\n" + "=" * 60)
    print("Test: Compare against scipy.interpolate.BSpline")
    print("=" * 60)

    try:
        from scipy.interpolate import BSpline
    except ImportError:
        print("⚠ scipy not available, skipping comparison test")
        return

    grid_size = 5
    spline_order = 3
    in_features = 1  # Test single feature for simplicity
    out_features = 1

    layer = KANLayer(
        in_features=in_features,
        out_features=out_features,
        grid_size=grid_size,
        spline_order=spline_order
    )

    # Get knot vector
    knots = layer.grid.numpy()
    n_coeffs = grid_size + spline_order

    print(f"Knots: {knots}")
    print(f"Number of basis functions: {n_coeffs}")

    # Test points
    x_test = np.linspace(-0.9, 0.9, 20)
    x_torch = torch.tensor(x_test, dtype=torch.float32).unsqueeze(-1)

    # Compute bases with our implementation
    with torch.no_grad():
        our_bases = layer._compute_bspline_basis(x_torch).squeeze(1).numpy()  # (n_points, n_coeffs)

    # Compute bases with scipy (manually for each basis function)
    scipy_bases = np.zeros((len(x_test), n_coeffs))
    for i in range(n_coeffs):
        # Create coefficient vector with 1 at position i
        coeffs = np.zeros(n_coeffs)
        coeffs[i] = 1.0
        # Create B-spline
        spl = BSpline(knots, coeffs, spline_order - 1)  # scipy uses degree, not order
        # Evaluate
        scipy_bases[:, i] = spl(x_test)

    # Compare
    max_diff = np.abs(our_bases - scipy_bases).max()
    mean_diff = np.abs(our_bases - scipy_bases).mean()

    print(f"\nComparison with scipy:")
    print(f"  Max difference: {max_diff:.6e}")
    print(f"  Mean difference: {mean_diff:.6e}")

    # Allow for small numerical differences
    assert max_diff < 1e-5, f"Large difference from scipy: {max_diff}"

    print("✓ Our implementation matches scipy.interpolate.BSpline")


def test_kan_layer_gradients():
    """Test that gradients flow correctly through KAN layer."""
    print("\n" + "=" * 60)
    print("Test: KAN Layer Gradient Flow")
    print("=" * 60)

    in_features = 4
    out_features = 3
    batch_size = 8

    layer = KANLayer(in_features, out_features, grid_size=5, spline_order=3)
    x = torch.randn(batch_size, in_features, requires_grad=True)

    output = layer(x)
    loss = output.sum()
    loss.backward()

    # Check gradients exist
    assert x.grad is not None, "No gradient for input"
    assert layer.spline_weight.grad is not None, "No gradient for spline weights"
    assert layer.base_weight.grad is not None, "No gradient for base weights"

    # Check gradients are non-zero
    assert x.grad.abs().sum() > 0, "Input gradient is zero"
    assert layer.spline_weight.grad.abs().sum() > 0, "Spline weight gradient is zero"
    assert layer.base_weight.grad.abs().sum() > 0, "Base weight gradient is zero"

    print(f"Input gradient norm: {x.grad.norm().item():.6f}")
    print(f"Spline weight gradient norm: {layer.spline_weight.grad.norm().item():.6f}")
    print(f"Base weight gradient norm: {layer.base_weight.grad.norm().item():.6f}")
    print("✓ Gradients flow correctly")


if __name__ == "__main__":
    test_bspline_dimension()
    test_partition_of_unity()
    test_bspline_smoothness()
    test_bspline_vs_scipy()
    test_kan_layer_gradients()
    print("\n" + "=" * 60)
    print("All B-spline tests passed!")
    print("=" * 60)
