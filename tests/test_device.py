"""Tests for device selection utilities."""

import torch
from nbody_gkan.device import get_device


def test_device_returns_torch_device():
    """Test that get_device returns a torch.device object."""
    device = get_device()
    assert isinstance(device, torch.device)


def test_device_is_valid():
    """Test that returned device is valid (cpu, cuda, or mps)."""
    device = get_device()
    assert device.type in ['cpu', 'cuda', 'mps']


def test_cpu_always_available():
    """Test that CPU device is always available as fallback."""
    device = get_device()
    # Should not raise an error
    tensor = torch.zeros(1).to(device)
    assert tensor.device.type in ['cpu', 'cuda', 'mps']


def test_device_priority():
    """Test that device selection follows priority: MPS > CUDA > CPU."""
    device = get_device()

    # Verify the device selected matches availability
    if torch.backends.mps.is_available():
        assert device.type == 'mps', "MPS should be selected when available"
    elif torch.cuda.is_available():
        assert device.type == 'cuda', "CUDA should be selected when available and MPS isn't"
    else:
        assert device.type == 'cpu', "CPU should be selected as fallback"
