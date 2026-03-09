"""
Self-contained B-spline Kolmogorov-Arnold Network (KAN) layer.

A KAN layer replaces the traditional Linear → Activation pattern with learnable
univariate spline functions on each input-output connection.
"""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class KANLayer(nn.Module):
    """
    Kolmogorov-Arnold Network layer using B-splines.

    Each connection between input and output has its own learnable univariate
    spline function, enabling interpretable function learning.

    Parameters
    ----------
    in_features : int
        Number of input features
    out_features : int
        Number of output features
    grid_size : int, optional (default=5)
        Number of grid intervals for the B-spline basis
    spline_order : int, optional (default=3)
        B-spline order (degree + 1). Order 3 = cubic splines
    base_activation : nn.Module or None, optional (default=nn.SiLU())
        Base activation function for residual path (set to None to disable)
    """

    def __init__(
            self,
            in_features: int,
            out_features: int,
            grid_size: int = 5,
            spline_order: int = 3,
            base_activation: Optional[nn.Module] = None,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.grid_size = grid_size
        self.spline_order = spline_order

        # Base linear transformation (residual path for stability)
        if base_activation is None:
            base_activation = nn.SiLU()
        self.base_activation = base_activation
        self.base_weight = nn.Parameter(torch.randn(out_features, in_features) * 0.1)

        # B-spline parameters
        # Each (in, out) pair has grid_size + spline_order coefficients
        n_coeffs = grid_size + spline_order
        self.spline_weight = nn.Parameter(
            torch.randn(out_features, in_features, n_coeffs) * 0.1
        )

        # Fixed grid: uniform spacing from -5 to 5, extended for B-spline basis
        # Wider range to accommodate raw physical inputs without normalization
        # For n_coeffs basis functions of order k, we need n_coeffs + k knots
        # B-spline formula: n_basis = n_knots - order
        # Therefore: n_knots = n_coeffs + spline_order
        n_knots = n_coeffs + spline_order
        grid_range = 5.0
        grid_ext = torch.linspace(
            -grid_range - (spline_order - 1) * 2.0 * grid_range / grid_size,
            grid_range + (spline_order - 1) * 2.0 * grid_range / grid_size,
            n_knots,
        )
        self.register_buffer("grid", grid_ext)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch_size, in_features)

        Returns
        -------
        torch.Tensor
            Output tensor of shape (batch_size, out_features)
        """
        batch_size = x.shape[0]

        # Base transformation: Linear(x) activated
        base_output = F.linear(self.base_activation(x), self.base_weight)

        # Spline transformation on raw inputs (no normalization)
        # This preserves interpretability: splines directly learn functions of physical quantities
        # Compute B-spline basis for each input
        # x: (batch, in_features) → (batch, in_features, n_coeffs)
        basis = self._compute_bspline_basis(x)  # (batch, in_features, n_coeffs)

        # Weighted sum of basis functions for each output
        # basis: (batch, in_features, n_coeffs)
        # spline_weight: (out_features, in_features, n_coeffs)
        # Want: (batch, out_features)
        spline_output = torch.einsum(
            "bin,oin->bo", basis, self.spline_weight
        )  # (batch, out_features)

        return base_output + spline_output

    def _compute_bspline_basis(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute B-spline basis functions.

        Parameters
        ----------
        x : torch.Tensor
            Input values, shape (batch_size, in_features)

        Returns
        -------
        torch.Tensor
            B-spline basis values, shape (batch_size, in_features, n_coeffs)
        """
        batch_size, in_features = x.shape
        n_coeffs = self.grid_size + self.spline_order

        # Expand x for broadcasting: (batch, in_features, 1)
        x_expanded = x.unsqueeze(-1)  # (batch, in_features, 1)

        # B-spline basis using Cox-de Boor recursion
        # Start with order 0 (piecewise constant)
        bases = []
        for i in range(len(self.grid) - 1):
            # Indicator function: 1 if grid[i] <= x < grid[i+1]
            mask = (x_expanded >= self.grid[i]) & (x_expanded < self.grid[i + 1])
            bases.append(mask.float().squeeze(-1))  # (batch, in_features)

        bases = torch.stack(bases, dim=-1)  # (batch, in_features, n_intervals)

        # Recursive computation for higher orders
        for k in range(1, self.spline_order):
            new_bases = []
            n_bases = len(self.grid) - k - 1
            for i in range(n_bases):
                # Left term
                denom1 = self.grid[i + k] - self.grid[i]
                if denom1 > 1e-8:
                    left = (x - self.grid[i]) / denom1 * bases[..., i]
                else:
                    left = torch.zeros_like(bases[..., i])

                # Right term
                denom2 = self.grid[i + k + 1] - self.grid[i + 1]
                if denom2 > 1e-8:
                    right = (self.grid[i + k + 1] - x) / denom2 * bases[..., i + 1]
                else:
                    right = torch.zeros_like(bases[..., i + 1])

                new_bases.append(left + right)

            bases = torch.stack(new_bases, dim=-1)  # (batch, in_features, n_bases)

        # Verify output has exactly n_coeffs basis functions
        if bases.shape[-1] != n_coeffs:
            raise RuntimeError(
                f"B-spline basis computation error: expected {n_coeffs} basis functions, "
                f"got {bases.shape[-1]}. This indicates a bug in the grid construction."
            )

        return bases

    def __repr__(self):
        return (
            f"KANLayer(in_features={self.in_features}, "
            f"out_features={self.out_features}, "
            f"grid_size={self.grid_size}, "
            f"spline_order={self.spline_order})"
        )
