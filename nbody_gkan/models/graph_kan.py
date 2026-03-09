"""
Graph-KAN: Graph Neural Network with KAN layers instead of MLPs.

Mirrors the architecture of the baseline GNN but replaces Linear+ReLU
layers with KAN layers for interpretable function learning.
"""

from typing import Optional

import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing

from .kan_layer import KANLayer
from .ordinary_mixin import OrdinaryMixin


class GraphKAN(MessagePassing):
    """
    Graph Neural Network using KAN layers for message and node update functions.

    This is a direct analog of the baseline GN class, but with KAN layers
    replacing the MLP message and node functions.

    Parameters
    ----------
    n_f : int
        Number of node features
    msg_dim : int
        Dimension of message vectors
    ndim : int
        Dimension of output (e.g., acceleration dimension)
    hidden : int or None, optional (default=None)
        Hidden layer dimension. If None, uses input dimension (narrow KAN).
        For fair comparison with baseline, set to 300.
    n_msg_layers : int, optional (default=3)
        Number of KAN layers in message function
    n_node_layers : int, optional (default=3)
        Number of KAN layers in node update function
    grid_size : int, optional (default=5)
        KAN: Number of B-spline grid intervals
    spline_order : int, optional (default=3)
        KAN: B-spline order
    aggr : str, optional (default='add')
        Aggregation method for messages
    """

    def __init__(
            self,
            n_f: int,
            msg_dim: int,
            ndim: int,
            hidden: Optional[int] = None,
            n_msg_layers: int = 3,
            n_node_layers: int = 3,
            grid_size: int = 5,
            spline_order: int = 3,
            aggr: str = "add",
    ):
        super().__init__(aggr=aggr)

        self.ndim = ndim

        # Message function: [x_i, x_j] (2*n_f) → msg_dim
        # Matches baseline GNN - concatenate all features from both nodes
        msg_input_dim = 2 * n_f
        self.msg_fnc = self._build_kan_network(
            msg_input_dim, msg_dim, n_msg_layers, grid_size, spline_order, hidden
        )

        # Node update function: [x, aggr_msgs] (n_f + msg_dim) → ndim
        # Matches baseline GNN - concatenate node features with aggregated messages
        node_input_dim = n_f + msg_dim
        self.node_fnc = self._build_kan_network(
            node_input_dim, ndim, n_node_layers, grid_size, spline_order, hidden
        )

    def _build_kan_network(
            self,
            in_dim: int,
            out_dim: int,
            n_layers: int,
            grid_size: int,
            spline_order: int,
            hidden_dim: Optional[int] = None,
    ) -> nn.Sequential:
        """
        Build a sequential KAN network.

        Parameters
        ----------
        in_dim : int
            Input dimension
        out_dim : int
            Output dimension
        n_layers : int
            Number of layers
        grid_size : int
            B-spline grid size
        spline_order : int
            B-spline order
        hidden_dim : int or None
            Hidden layer dimension. If None, uses in_dim (narrow network).

        Returns
        -------
        nn.Sequential
            Sequential KAN network
        """
        if n_layers == 1:
            return nn.Sequential(KANLayer(in_dim, out_dim, grid_size, spline_order))

        # Use hidden_dim if provided, otherwise use in_dim (narrow KAN)
        h_dim = hidden_dim if hidden_dim is not None else in_dim

        layers = []
        # First layer: in_dim → h_dim
        layers.append(KANLayer(in_dim, h_dim, grid_size, spline_order))

        # Hidden layers: h_dim → h_dim
        for _ in range(n_layers - 2):
            layers.append(KANLayer(h_dim, h_dim, grid_size, spline_order))

        # Output layer: h_dim → out_dim
        layers.append(KANLayer(h_dim, out_dim, grid_size, spline_order))

        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the graph network.

        Parameters
        ----------
        x : torch.Tensor
            Node features, shape (n_nodes, n_f)
        edge_index : torch.Tensor
            Edge indices, shape (2, n_edges)

        Returns
        -------
        torch.Tensor
            Updated node features, shape (n_nodes, ndim)
        """
        return self.propagate(edge_index, size=(x.size(0), x.size(0)), x=x)

    def message(self, x_i: torch.Tensor, x_j: torch.Tensor) -> torch.Tensor:
        """
        Compute messages from node j to node i.

        Matches baseline GNN: simply concatenate all features from both nodes.

        Parameters
        ----------
        x_i : torch.Tensor
            Features of receiving nodes, shape (n_edges, n_f)
        x_j : torch.Tensor
            Features of sending nodes, shape (n_edges, n_f)

        Returns
        -------
        torch.Tensor
            Messages, shape (n_edges, msg_dim)
        """
        # Concatenate all features: [x_i, x_j]
        tmp = torch.cat([x_i, x_j], dim=1)  # (n_edges, 2*n_f)
        return self.msg_fnc(tmp)

    def update(
            self, aggr_out: torch.Tensor, x: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Update node features using aggregated messages.

        Matches baseline GNN: concatenate all node features with aggregated messages.

        Parameters
        ----------
        aggr_out : torch.Tensor
            Aggregated messages, shape (n_nodes, msg_dim)
        x : torch.Tensor, optional
            Original node features, shape (n_nodes, n_f)

        Returns
        -------
        torch.Tensor
            Updated node features, shape (n_nodes, ndim)
        """
        # Concatenate [x, aggregated_messages]
        tmp = torch.cat([x, aggr_out], dim=1)  # (n_nodes, msg_dim+n_f)
        return self.node_fnc(tmp)


class OrdinaryGraphKAN(OrdinaryMixin, GraphKAN):
    """
    Ordinary Graph-KAN with position augmentation and loss computation.

    This is analogous to the OGN (Ordinary Graph Network) in the baseline,
    but using KAN layers.

    Parameters
    ----------
    n_f : int
        Number of node features
    msg_dim : int
        Dimension of message vectors
    ndim : int
        Dimension of output (spatial dimension for accelerations)
    edge_index : torch.Tensor
        Fixed edge indices for the graph, shape (2, n_edges)
    hidden : int or None, optional (default=None)
        Hidden layer dimension. If None, uses input dimension (narrow KAN).
        For fair comparison with baseline, set to 300.
    n_msg_layers : int, optional (default=3)
        Number of KAN layers in message function
    n_node_layers : int, optional (default=3)
        Number of KAN layers in node update function
    grid_size : int, optional (default=5)
        KAN: Number of B-spline grid intervals
    spline_order : int, optional (default=3)
        KAN: B-spline order
    aggr : str, optional (default='add')
        Aggregation method
    """

    def __init__(
            self,
            n_f: int,
            msg_dim: int,
            ndim: int,
            edge_index: torch.Tensor,
            hidden: Optional[int] = None,
            n_msg_layers: int = 3,
            n_node_layers: int = 3,
            grid_size: int = 5,
            spline_order: int = 3,
            aggr: str = "add",
    ):
        super().__init__(
            n_f, msg_dim, ndim, hidden, n_msg_layers, n_node_layers, grid_size, spline_order, aggr
        )
        self.ndim = ndim
        self.register_buffer("edge_index", edge_index)

    # just_derivative() and loss() are inherited from OrdinaryMixin
