"""Tests for Trainer."""

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from nbody_gkan.data.dataset import NBodyDataset, get_edge_index
from nbody_gkan.models.graph_kan import OrdinaryGraphKAN
from nbody_gkan.training.trainer import Trainer


@pytest.fixture
def small_dataset(tmp_path):
    """Create a small dataset for training tests."""
    n_traj, T, n, dim = 2, 10, 3, 2
    positions = np.random.randn(n_traj, T, n, dim) * 0.5
    velocities = np.random.randn(n_traj, T, n, dim) * 0.1
    masses = np.ones(n)

    data_path = tmp_path / "train.npz"
    np.savez(data_path, positions=positions, velocities=velocities, masses=masses)
    return NBodyDataset(data_path)


@pytest.fixture
def simple_model():
    """Create a simple model for testing."""
    edge_index = get_edge_index(3)
    model = OrdinaryGraphKAN(
        n_f=5,
        msg_dim=10,
        ndim=2,
        edge_index=edge_index,
        n_msg_layers=2,
        n_node_layers=2,
        grid_size=3,
        spline_order=2,
    )
    return model


def test_trainer_initialization(small_dataset, simple_model):
    """Test that Trainer initializes correctly."""
    loader = DataLoader(small_dataset, batch_size=4, collate_fn=small_dataset.collate_fn)
    optimizer = torch.optim.Adam(simple_model.parameters(), lr=1e-3)

    trainer = Trainer(
        model=simple_model,
        train_loader=loader,
        optimizer=optimizer,
        device=torch.device('cpu'),
    )

    assert trainer.model == simple_model
    assert trainer.train_loader == loader
    assert trainer.optimizer == optimizer
    assert trainer.device == torch.device('cpu')
    assert trainer.epoch == 0


def test_trainer_creates_checkpoint_dir(small_dataset, simple_model, tmp_path):
    """Test that Trainer creates checkpoint directory."""
    loader = DataLoader(small_dataset, batch_size=4, collate_fn=small_dataset.collate_fn)
    checkpoint_dir = tmp_path / "test_checkpoints"
    optimizer = torch.optim.Adam(simple_model.parameters(), lr=1e-3)

    trainer = Trainer(
        model=simple_model,
        train_loader=loader,
        optimizer=optimizer,
        device=torch.device('cpu'),
        checkpoint_dir=checkpoint_dir,
    )

    assert checkpoint_dir.exists()
    assert checkpoint_dir.is_dir()


def test_training_step_executes(small_dataset, simple_model):
    """Test that a single training step executes without errors."""
    loader = DataLoader(small_dataset, batch_size=4, collate_fn=small_dataset.collate_fn)

    # Need to create optimizer for training
    optimizer = torch.optim.Adam(simple_model.parameters(), lr=1e-3)

    trainer = Trainer(
        model=simple_model,
        train_loader=loader,
        optimizer=optimizer,
        device=torch.device('cpu'),
    )

    # Train for 1 epoch
    trainer.train(n_epochs=1, augment=False)

    # Access history from trainer
    history = trainer.history
    assert 'train_loss' in history
    assert len(history['train_loss']) == 1
    assert isinstance(history['train_loss'][0], float)


def test_loss_decreases(small_dataset, simple_model):
    """Test that loss decreases over training."""
    loader = DataLoader(small_dataset, batch_size=4, collate_fn=small_dataset.collate_fn)

    optimizer = torch.optim.Adam(simple_model.parameters(), lr=1e-2)

    trainer = Trainer(
        model=simple_model,
        train_loader=loader,
        optimizer=optimizer,
        device=torch.device('cpu'),
    )

    # Train for multiple epochs
    trainer.train(n_epochs=20, augment=False)

    # Loss should decrease
    history = trainer.history
    initial_loss = history['train_loss'][0]
    final_loss = history['train_loss'][-1]

    assert final_loss < initial_loss, \
        f"Loss did not decrease: {initial_loss:.6f} -> {final_loss:.6f}"


def test_checkpointing(small_dataset, simple_model, tmp_path):
    """Test that checkpointing works correctly."""
    loader = DataLoader(small_dataset, batch_size=4, collate_fn=small_dataset.collate_fn)
    checkpoint_dir = tmp_path / "checkpoints"

    optimizer = torch.optim.Adam(simple_model.parameters(), lr=1e-3)

    trainer = Trainer(
        model=simple_model,
        train_loader=loader,
        optimizer=optimizer,
        device=torch.device('cpu'),
        checkpoint_dir=checkpoint_dir,
    )

    # Train and save checkpoint
    trainer.train(n_epochs=2, augment=False, save_every=1)

    # Check that checkpoint files exist
    assert checkpoint_dir.exists()
    checkpoints = list(checkpoint_dir.glob("*.pt"))
    assert len(checkpoints) > 0


def test_validation_loop(small_dataset, simple_model):
    """Test that validation loop executes."""
    train_loader = DataLoader(small_dataset, batch_size=4, collate_fn=small_dataset.collate_fn)
    val_loader = DataLoader(small_dataset, batch_size=4, collate_fn=small_dataset.collate_fn)

    optimizer = torch.optim.Adam(simple_model.parameters(), lr=1e-3)

    trainer = Trainer(
        model=simple_model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        device=torch.device('cpu'),
    )

    # Train with validation
    trainer.train(n_epochs=2, augment=False)

    history = trainer.history
    assert 'val_loss' in history
    assert len(history['val_loss']) == 2
    assert all(isinstance(loss, float) for loss in history['val_loss'])


def test_gradient_clipping(small_dataset, simple_model):
    """Test that gradient clipping doesn't break training."""
    loader = DataLoader(small_dataset, batch_size=4, collate_fn=small_dataset.collate_fn)

    optimizer = torch.optim.Adam(simple_model.parameters(), lr=1e-3)

    trainer = Trainer(
        model=simple_model,
        train_loader=loader,
        optimizer=optimizer,
        device=torch.device('cpu'),
    )

    # Train with gradient clipping
    trainer.train(n_epochs=2, augment=False, gradient_clip=1.0)

    history = trainer.history
    assert len(history['train_loss']) == 2


def test_model_moves_to_device(small_dataset, simple_model):
    """Test that model is moved to the specified device."""
    loader = DataLoader(small_dataset, batch_size=4, collate_fn=small_dataset.collate_fn)
    optimizer = torch.optim.Adam(simple_model.parameters(), lr=1e-3)

    trainer = Trainer(
        model=simple_model,
        train_loader=loader,
        optimizer=optimizer,
        device=torch.device('cpu'),
    )

    # Check that model parameters are on CPU
    for param in trainer.model.parameters():
        assert param.device.type == 'cpu'


def test_history_tracking(small_dataset, simple_model):
    """Test that training history is tracked correctly."""
    loader = DataLoader(small_dataset, batch_size=4, collate_fn=small_dataset.collate_fn)

    optimizer = torch.optim.Adam(simple_model.parameters(), lr=1e-3)

    trainer = Trainer(
        model=simple_model,
        train_loader=loader,
        optimizer=optimizer,
        device=torch.device('cpu'),
    )

    n_epochs = 3
    trainer.train(n_epochs=n_epochs, augment=False)

    history = trainer.history
    # Check history structure
    assert 'train_loss' in history
    assert 'val_loss' in history
    assert 'lr' in history

    # Check lengths
    assert len(history['train_loss']) == n_epochs
    assert len(history['lr']) == n_epochs
