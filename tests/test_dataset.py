"""Tests for NBodyDataset."""

import numpy as np
import pytest
import torch

from nbody_gkan.data.dataset import NBodyDataset, get_edge_index
from nbody_gkan.nbody import gravity


@pytest.fixture
def test_data_path(tmp_path):
    """Create a small test dataset."""
    n_traj, T, n, dim = 2, 10, 3, 2
    positions = np.random.randn(n_traj, T, n, dim) * 0.5
    velocities = np.random.randn(n_traj, T, n, dim) * 0.1
    masses = np.ones(n)

    data_path = tmp_path / "test.npz"
    np.savez(data_path, positions=positions, velocities=velocities, masses=masses)
    return data_path


def test_dataset_loads_file(test_data_path):
    """Test that dataset loads .npz file correctly."""
    dataset = NBodyDataset(test_data_path)
    assert len(dataset) == 20  # 2 trajectories × 10 timesteps


def test_dataset_shapes(test_data_path):
    """Test that data shapes are correct."""
    dataset = NBodyDataset(test_data_path)
    data = dataset[0]

    # x: [pos, vel, mass] = [2+2+1] = 5 features per node
    assert data.x.shape == (3, 5)
    # y: accelerations = 2D
    assert data.y.shape == (3, 2)
    # edge_index: fully connected (3 nodes, 6 edges)
    assert data.edge_index.shape == (2, 6)


def test_dataset_shapes_without_velocity(test_data_path):
    """Test that disabling velocity changes node feature width."""
    dataset = NBodyDataset(test_data_path, include_velocity=False)
    data = dataset[0]

    # x: [pos, mass] = [2+1] = 3 features per node
    assert data.x.shape == (3, 3)
    assert dataset.n_node_features == 3
    assert dataset.include_velocity is False


def test_dataset_feature_spec_augmentation_is_rejected(test_data_path):
    """Feature augmentations are intentionally unsupported."""
    with pytest.raises(ValueError, match="feature_spec\.augment is not supported"):
        NBodyDataset(
            test_data_path,
            feature_spec={
                "include": ["pos", "mass"],
                "augment": ["speed"],
            },
        )


def test_edge_index_fully_connected():
    """Test that edge index creates fully connected graph."""
    edge_index = get_edge_index(3)

    # Should have n*(n-1) edges
    assert edge_index.shape == (2, 6)

    # Check no self-loops
    sources = edge_index[0].numpy()
    targets = edge_index[1].numpy()
    assert not np.any(sources == targets)

    # Check bidirectional
    for i in range(3):
        for j in range(3):
            if i != j:
                # Should have edge i->j
                assert np.any((sources == i) & (targets == j))


def test_acceleration_computation(test_data_path):
    """Test that accelerations are computed correctly."""
    dataset = NBodyDataset(test_data_path, G=1.0, softening=1e-2)
    data = dataset[0]

    # Extract positions and masses
    pos = data.x[:, :2].numpy()  # First 2 features
    masses = data.x[:, 4].numpy()  # Last feature

    # Recompute acceleration
    acc_expected = gravity(pos, masses, G=1.0, softening=1e-2)
    acc_actual = data.y.numpy()

    np.testing.assert_allclose(acc_actual, acc_expected, rtol=1e-5)


def test_node_features_structure(test_data_path):
    """Test that node features have correct structure."""
    dataset = NBodyDataset(test_data_path)
    data = dataset[0]

    # Features: [pos_x, pos_y, vel_x, vel_y, mass]
    x = data.x

    # Check positions (first 2 columns)
    pos = x[:, :2]
    assert pos.shape == (3, 2)

    # Check velocities (columns 2-4)
    vel = x[:, 2:4]
    assert vel.shape == (3, 2)

    # Check masses (last column)
    mass = x[:, 4:5]
    assert mass.shape == (3, 1)
    assert torch.allclose(mass, torch.ones_like(mass))  # All masses = 1


def test_node_features_structure_without_velocity(test_data_path):
    """Test node feature layout when velocity inputs are disabled."""
    dataset = NBodyDataset(test_data_path, include_velocity=False)
    data = dataset[0]

    # Features: [pos_x, pos_y, mass]
    x = data.x

    # Check positions (first 2 columns)
    pos = x[:, :2]
    assert pos.shape == (3, 2)

    # Check masses (last column)
    mass = x[:, 2:3]
    assert mass.shape == (3, 1)
    assert torch.allclose(mass, torch.ones_like(mass))


def test_dataset_dtype(test_data_path):
    """Test that tensors have correct dtype."""
    dataset = NBodyDataset(test_data_path)
    data = dataset[0]

    assert data.x.dtype == torch.float32
    assert data.y.dtype == torch.float32
    assert data.edge_index.dtype == torch.int64


def test_dataset_iteration(test_data_path):
    """Test that we can iterate over the dataset."""
    dataset = NBodyDataset(test_data_path)

    count = 0
    for data in dataset:
        assert hasattr(data, 'x')
        assert hasattr(data, 'y')
        assert hasattr(data, 'edge_index')
        count += 1

    assert count == len(dataset)


def test_different_g_values(test_data_path):
    """Test that different G values affect accelerations."""
    dataset1 = NBodyDataset(test_data_path, G=1.0)
    dataset2 = NBodyDataset(test_data_path, G=2.0)

    data1 = dataset1[0]
    data2 = dataset2[0]

    # Same positions and velocities
    assert torch.allclose(data1.x, data2.x)

    # Different accelerations (should be scaled by G)
    assert not torch.allclose(data1.y, data2.y)
    # data2.y should be approximately 2x data1.y
    assert torch.allclose(data2.y, data1.y * 2.0, rtol=0.01)


def test_edge_index_consistency(test_data_path):
    """Test that edge_index is the same for all samples."""
    dataset = NBodyDataset(test_data_path)

    edge_index_0 = dataset[0].edge_index
    for i in range(1, min(5, len(dataset))):
        assert torch.equal(dataset[i].edge_index, edge_index_0)
