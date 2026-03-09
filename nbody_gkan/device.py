"""Device utilities for handling MPS/CUDA/CPU selection."""

import torch


def get_device() -> torch.device:
    """
    Automatically detect and return the best available device.

    Priority order:
    1. MPS (Metal Performance Shaders) on Apple Silicon
    2. CUDA on NVIDIA GPUs
    3. CPU as fallback

    Returns
    -------
    torch.device
        The selected device
    """
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")
