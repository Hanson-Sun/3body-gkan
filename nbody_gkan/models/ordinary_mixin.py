"""
Mixin class for ODE-style graph networks with position augmentation and loss.

This mixin provides common functionality for OrdinaryGraphKAN and OGN.
"""

import torch


class OrdinaryMixin:
    """
    Mixin for ordinary differential equation-style graph networks.

    Provides:
    - Position augmentation for training
    - Loss computation (L1 or L2)

    Classes using this mixin must:
    - Have a `ndim` attribute (spatial dimension)
    - Implement `propagate()` method from MessagePassing
    """

    def just_derivative(
            self,
            g,
            augment: bool = False,
            augmentation: float = 3.0,
    ) -> torch.Tensor:
        """
        Compute derivative (acceleration) with optional position augmentation.

        Position augmentation matches original OGN: applies the SAME random translation
        to ALL nodes in the input (regardless of batching). This teaches translation
        invariance while preserving relative positions.

        Parameters
        ----------
        g : torch_geometric.data.Data
            Graph data with node features g.x and edge_index.
        augment : bool, optional (default=False)
            Whether to apply position augmentation
        augmentation : float, optional (default=3.0)
            Scale of augmentation noise

        Returns
        -------
        torch.Tensor
            Predicted accelerations, shape (n_nodes, ndim)
        """
        x = g.x
        ndim = self.ndim

        if augment:
            # Match original OGN: single random translation for ALL nodes
            # This teaches translation invariance (physics doesn't depend on absolute position)
            x = x.clone()
            aug_noise = torch.randn(1, ndim, device=x.device) * augmentation
            aug_noise_per_node = aug_noise.expand(len(x), ndim)
            x[:, :ndim] = x[:, :ndim] + aug_noise_per_node

        edge_index = g.edge_index

        return self.propagate(edge_index, size=(x.size(0), x.size(0)), x=x)

    def loss(
            self,
            g,
            augment: bool = True,
            square: bool = False,
            augmentation: float = 3.0,
            lamb: float = 0,  # 0 is off
            **kwargs,  # Absorb extra arguments for API compatibility
    ) -> torch.Tensor:
        """
        Compute loss between predicted and true accelerations.

        Parameters
        ----------
        g : torch_geometric.data.Data
            Graph data with node features g.x, targets g.y, and edge_index
        augment : bool, optional (default=True)
            Whether to apply position augmentation during training
        square : bool, optional (default=False)
            Use L2 loss if True, L1 loss if False
        augmentation : float, optional (default=3.0)
            Scale of augmentation noise

        Returns
        -------
        torch.Tensor
            Scalar loss value
        """
        pred = self.just_derivative(g, augment=augment, augmentation=augmentation)

        if square:
            task_loss = torch.mean((g.y - pred) ** 2)
        else:
            task_loss = torch.mean(torch.abs(g.y - pred))
    
        if lamb > 0.0 and hasattr(self, 'regularization'):
            reg = self.regularization()
            return task_loss + lamb * reg
        
        return task_loss