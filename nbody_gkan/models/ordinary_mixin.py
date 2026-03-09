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

        Parameters
        ----------
        g : torch_geometric.data.Data
            Graph data with node features g.x and edge_index.
            If batched, must have g.batch attribute.
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
            # Add random noise to positions - different noise for each graph in batch
            # This preserves relative positions within each graph and teaches translation invariance
            x = x.clone()

            # Handle batched graphs (PyG Batch) vs single graphs (Data)
            if hasattr(g, 'batch'):
                # Batched case: apply different translation to each graph
                n_graphs = g.batch.max().item() + 1
                # Generate different random translation for each graph
                aug_noise = torch.randn(n_graphs, ndim, device=x.device) * augmentation
                # Index into aug_noise using batch tensor to get per-node translation
                aug_noise_per_node = aug_noise[g.batch]  # (n_nodes, ndim)
            else:
                # Single graph case: same translation for all nodes
                aug_noise_per_node = torch.randn(1, ndim, device=x.device) * augmentation
                aug_noise_per_node = aug_noise_per_node.repeat(len(x), 1)

            # Apply augmentation to positions (first ndim features)
            x[:, :ndim] = x[:, :ndim] + aug_noise_per_node

        edge_index = g.edge_index

        return self.propagate(edge_index, size=(x.size(0), x.size(0)), x=x)

    def loss(
            self,
            g,
            augment: bool = True,
            square: bool = False,
            augmentation: float = 3.0,
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
            return torch.mean((g.y - pred) ** 2)
        else:
            return torch.mean(torch.abs(g.y - pred))
