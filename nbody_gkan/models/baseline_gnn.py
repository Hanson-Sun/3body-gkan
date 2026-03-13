"""
Baseline MLP-based Graph Neural Network models.

Ported from symbolic_deep_learning/models.py with:
- Type hints
- Docstrings
- Same interface as Graph-KAN for fair comparison
"""

from typing import Optional

import torch
from torch.nn import Sequential as Seq, Linear as Lin, ReLU
from torch_geometric.nn import MessagePassing

from .ordinary_mixin import OrdinaryMixin
from .graph_mixin import GraphMixin


class GN(MessagePassing, GraphMixin):
    """
    Graph Network with MLP message and node update functions.

    This is a baseline model using traditional MLPs (Linear + ReLU layers)
    for message passing and node updates. Architecture matches the original
    OGN from symbolic_deep_learning: 4 layers, 300 hidden dimension.

    Parameters
    ----------
    n_f : int
        Number of node features
    msg_dim : int
        Dimension of message vectors
    ndim : int
        Dimension of output (e.g., acceleration dimension)
    hidden : int, optional (default=300)
        Hidden layer dimension
    aggr : str, optional (default='add')
        Aggregation method for messages ('add', 'mean', 'max')
    """

    def __init__(
            self, n_f: int, msg_dim: int, ndim: int, hidden: int = 300, aggr: str = "add"
    ):
        super().__init__(aggr=aggr)
        
        self.n_f = n_f
        self.msg_dim = msg_dim
        self.ndim = ndim
        self.hidden = hidden

        # Message function: [x_i, x_j] (2*n_f) → msg_dim
        # 4 layers, configurable hidden dimension (default 300) - matches original
        self.msg_fnc = Seq(
            Lin(2 * n_f, hidden),
            ReLU(),
            Lin(hidden, hidden),
            ReLU(),
            Lin(hidden, hidden),
            ReLU(),
            Lin(hidden, msg_dim),
        )

        # Node update function: [x, aggr_msgs] (n_f + msg_dim) → ndim
        # 4 layers, configurable hidden dimension (default 300) - matches original
        self.node_fnc = Seq(
            Lin(msg_dim + n_f, hidden),
            ReLU(),
            Lin(hidden, hidden),
            ReLU(),
            Lin(hidden, hidden),
            ReLU(),
            Lin(hidden, ndim),
        )

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

        Matches original paper: simply concatenate all features from both nodes.

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
        
    def summary(self):
        super().summary()


class OGN(OrdinaryMixin, GN):
    """
    Ordinary Graph Network with position augmentation and loss computation.

    Extends the base GN with:
    - Stored edge indices
    - Position augmentation for training
    - Loss computation

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
    hidden : int, optional (default=300)
        Hidden layer dimension
    aggr : str, optional (default='add')
        Aggregation method
    """

    def __init__(
            self,
            n_f: int,
            msg_dim: int,
            ndim: int,
            edge_index: torch.Tensor,
            hidden: int = 300,
            aggr: str = "add",
    ):
        super().__init__(n_f, msg_dim, ndim, hidden, aggr=aggr)
        self.ndim = ndim
        self.register_buffer("edge_index", edge_index)

    # just_derivative() and loss() are inherited from OrdinaryMixin
