"""
Graph-KAN: Graph Neural Network with KAN layers instead of MLPs.

Uses pykan's KANLayer implementation with native support for activation
storage and symbolic regression.
"""

from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import sympy as sp
from tqdm import tqdm
from kan import KANLayer
from torch_geometric.nn import MessagePassing

from .ordinary_mixin import OrdinaryMixin
from .graph_mixin import GraphMixin


class GraphKAN(MessagePassing, GraphMixin):
    """
    Graph Neural Network using KAN layers for message and node update functions.

    Architecture matches the baseline GNN: 4 layers, 300 hidden dimension.
    Uses pykan's KANLayer with B-spline basis functions.

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
    grid_size : int, optional (default=5)
        KAN: Number of B-spline grid intervals (must be >= 1)
    spline_order : int, optional (default=3)
        KAN: B-spline order (must be >= 1)
    aggr : str, optional (default='add')
        Aggregation method for messages
    sparse_init : bool, optional (default=False)
        KAN: Whether to use sparse initialization for layer connectivity

    Notes
    -----
    - Hidden dimension defaults to 300 to match baseline OGN
    - Architecture is 4 layers to match baseline exactly
    - KAN-specific parameters (grid_size, spline_order) are the only additions beyond baseline
    """

    def __init__(
            self,
            n_f: int,
            msg_dim: int,
            ndim: int,
            hidden: int = 300,
            node_hidden: int = 300,
            grid_size: int = 5,
            spline_order: int = 3,
            aggr: str = "add",
            hidden_layers: int = 0,
            node_hidden_layers: int = 0,
            lamb_l1: float = 1.0,
            lamb_entropy: float = 2.0,
            sparse_init: bool = False,
        ):
        super().__init__(aggr=aggr)
        self.n_f = n_f
        self.msg_dim = msg_dim
        self.ndim = ndim
        self.hidden = hidden
        self.node_hidden = node_hidden
        self.grid_size = grid_size
        self.spline_order = spline_order
        self.hidden_layers = hidden_layers
        self.node_hidden_layers = node_hidden_layers
        self.lamb_l1 = lamb_l1
        self.lamb_entropy = lamb_entropy
        self.sparse_init = sparse_init

        # Message function: [x_i, x_j] (2*n_f) → msg_dim
        # 4 layers, configurable hidden (default 300) - matches baseline
        msg_input_dim = 2 * n_f
        self.msg_layers = self._build_kan_network(
            msg_input_dim, msg_dim, hidden, grid_size, spline_order, hidden_layers, sparse_init
        )

        # Node update function: [x, aggr_msgs] (n_f + msg_dim) → ndim
        # 4 layers, configurable hidden (default 300) - matches baseline
        node_input_dim = n_f + msg_dim
        self.node_layers = self._build_kan_network(
            node_input_dim, ndim, node_hidden, grid_size, spline_order, node_hidden_layers, sparse_init
        )

    def _build_kan_network(
            self,
            in_dim: int,
            out_dim: int,
            hidden: int,
            grid_size: int,
            spline_order: int,
            hidden_layers: int = 0,
            sparse_init: bool = True,
    ) -> nn.ModuleList:
        """
        Build a 4-layer KAN network with configurable hidden dimension (matches baseline).

        Parameters
        ----------
        in_dim : int
            Input dimension
        out_dim : int
            Output dimension
        hidden : int
            Hidden layer dimension
        grid_size : int
            B-spline grid size (num parameter in pykan)
        spline_order : int
            B-spline order (k parameter in pykan)

        Returns
        -------
        nn.ModuleList
            List of 4 KAN layers
        """
        layers = []
        # Layer 1: in_dim → hidden
        layers.append(KANLayer(in_dim=in_dim, out_dim=hidden, num=grid_size, k=spline_order, sparse_init=sparse_init))
        for _ in range(hidden_layers):
            layers.append(KANLayer(in_dim=hidden, out_dim=hidden, num=grid_size, k=spline_order, sparse_init=sparse_init))
        # Final layer: hidden → out_dim
        layers.append(KANLayer(in_dim=hidden, out_dim=out_dim, num=grid_size, k=spline_order, sparse_init=sparse_init))

        return nn.ModuleList(layers)

    def _forward_kan_layers(
            self,
            x: torch.Tensor,
            layers: nn.ModuleList,
    ) -> torch.Tensor:
        """
        Forward through KAN layers.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor
        layers : nn.ModuleList
            List of KAN layers

        Returns
        -------
        torch.Tensor
            Output tensor
        """
        for layer in layers:
            # pykan's KANLayer returns (y, preacts, postacts, postspline)
            # We only need the output y
            y, _, _, _ = layer(x)
            x = y
        return x

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
        return self._forward_kan_layers(tmp, self.msg_layers)

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
        return self._forward_kan_layers(tmp, self.node_layers)

    def update_grids(self, data_loader, device:torch.device | str = 'cpu', max_batches: int = 10):
        """
        Update KAN grids based on training data distribution.

        This is an OPTIONAL KAN-specific optimization feature. Grid updates adapt
        the B-spline grid boundaries to match the actual input distributions seen
        during training, which can improve KAN performance.

        Each layer's grid is updated based on the input distribution it actually sees
        during forward pass, not just the network input.

        Control via training script: --grid_update_freq N (0 to disable)

        Parameters
        ----------
        data_loader : DataLoader
            Training data loader
        device : str or torch.device
            Device to run on
        max_batches : int, optional (default=10)
            Maximum number of batches to use for grid adaptation
        """
        from torch_scatter import scatter_add

        self.eval()
        with torch.no_grad():
            # Collect inputs for FIRST layer only (we'll propagate for subsequent layers)
            msg_inputs_layer0 = []
            node_inputs_layer0 = []

            # Pass 1: collect msg inputs
            for i, batch in enumerate(data_loader):
                if i >= max_batches:
                    break
                batch = batch.to(device)
                x = batch.x
                edge_index = batch.edge_index
                row, col = edge_index
                msg_input = torch.cat([x[row], x[col]], dim=1)
                msg_inputs_layer0.append(msg_input)

            # Update message grids first
            if msg_inputs_layer0:
                layer_input = torch.cat(msg_inputs_layer0, dim=0)

                # Guard against zero-variance dimensions (e.g. constant mass)
                std = layer_input.std(dim=0, keepdim=True)
                zero_var_dims = std < 1e-6
                if zero_var_dims.any():
                    zero_dims = zero_var_dims.squeeze().nonzero().flatten().tolist()
                    tqdm.write(f"  [Grid Update] Jittering zero-variance msg dims: {zero_dims}")
                    jitter = torch.randn_like(layer_input) * 1e-4
                    layer_input = layer_input + jitter * zero_var_dims

                for layer in self.msg_layers:
                    layer.update_grid_from_samples(layer_input)
                    layer_input, _, _, _ = layer(layer_input)

            # Pass 2: collect node inputs AFTER message grids are updated
            for i, batch in enumerate(data_loader):
                if i >= max_batches:
                    break
                batch = batch.to(device)
                x = batch.x
                edge_index = batch.edge_index
                row, col = edge_index
                msg_input = torch.cat([x[row], x[col]], dim=1)
                msg_out = self._forward_kan_layers(msg_input, self.msg_layers)
                aggr_msg = scatter_add(msg_out, row, dim=0, dim_size=x.size(0))
                node_inputs_layer0.append(torch.cat([x, aggr_msg], dim=1))

            # Now update node grids with fresh inputs
            if node_inputs_layer0:
                layer_input = torch.cat(node_inputs_layer0, dim=0)

                # Guard against zero-variance dimensions in node inputs
                std = layer_input.std(dim=0, keepdim=True)
                zero_var_dims = std < 1e-6
                if zero_var_dims.any():
                    zero_dims = zero_var_dims.squeeze().nonzero().flatten().tolist()
                    tqdm.write(f"  [Grid Update] Jittering zero-variance node dims: {zero_dims}")
                    jitter = torch.randn_like(layer_input) * 1e-4
                    layer_input = layer_input + jitter * zero_var_dims

                for layer in self.node_layers:
                    layer.update_grid_from_samples(layer_input)
                    layer_input, _, _, _ = layer(layer_input)

        self.train()

    def suggest_symbolic(self, data_loader, device='cpu', lib=None,
                        max_batches=10, threshold=0.8) -> dict:
        if lib is None:
            lib = ['x', 'x^2', 'x^3', '1/x', '1/x^2',
                'sqrt(x)', 'log(x)', 'exp(x)', 'abs(x)', 'sin(x)']
    
        self.eval()
        with torch.no_grad():
            for i, batch in enumerate(data_loader):
                if i >= max_batches:
                    break
                batch = batch.to(device)
                # Call GraphKAN.forward directly — bypasses OrdinaryMixin
                # which may not trigger message/update and populate acts_scale
                self.propagate(
                    batch.edge_index,
                    size=(batch.x.size(0), batch.x.size(0)),
                    x=batch.x
                )
    
        # Diagnostic — log whether acts_scale got populated
        for i, layer in enumerate(self.msg_layers):
            if not hasattr(layer, 'acts_scale'):
                tqdm.write(f"  Warning: msg layer {i} has no acts_scale — "
                        f"forward pass may not have reached it")
        for i, layer in enumerate(self.node_layers):
            if not hasattr(layer, 'acts_scale'):
                tqdm.write(f"  Warning: node layer {i} has no acts_scale — "
                        f"forward pass may not have reached it")
    
        suggestions = {'msg_layers': {}, 'node_layers': {}}
    
        def _suggest_for_layers(layers, layer_key):
            for layer_idx, layer in enumerate(layers):
                suggestions[layer_key][layer_idx] = {}
                for in_i in range(layer.in_dim):
                    for out_i in range(layer.out_dim):
                        best_fn     = None
                        best_r2     = -float('inf')
                        best_coeffs = None
    
                        x_sample = torch.linspace(
                            layer.grid[in_i, layer.k].item(),
                            layer.grid[in_i, -layer.k - 1].item(),
                            steps=100,
                        ).unsqueeze(1).to(device)
    
                        with torch.no_grad():
                            y_spline, _, _, _ = layer(
                                x_sample.expand(-1, layer.in_dim)
                            )
                        y_edge = y_spline[:, out_i].cpu().numpy()
                        x_np   = x_sample.squeeze().cpu().numpy()
    
                        for fn_str in lib:
                            try:
                                x_sym  = sp.Symbol('x')
                                fn_sym = sp.sympify(fn_str)
                                fn_lam = sp.lambdify(x_sym, fn_sym, 'numpy')
    
                                with np.errstate(invalid='ignore', divide='ignore'):
                                    y_cand = fn_lam(x_np).astype(float)
    
                                if not np.isfinite(y_cand).all():
                                    continue
                                A      = np.stack([y_cand, np.ones_like(y_cand)], axis=1)
                                coeffs, _, _, _ = np.linalg.lstsq(A, y_edge, rcond=None)
                                y_fit  = A @ coeffs
                                ss_res = np.sum((y_edge - y_fit) ** 2)
                                ss_tot = np.sum((y_edge - y_edge.mean()) ** 2)
                                r2     = 1 - ss_res / (ss_tot + 1e-10)
                                if r2 > best_r2:
                                    best_r2     = r2
                                    best_fn     = fn_str
                                    best_coeffs = coeffs
                            except Exception:
                                continue
    
                        if best_fn is not None and best_r2 >= threshold:
                            suggestions[layer_key][layer_idx][(in_i, out_i)] = {
                                'fn': best_fn,
                                'r2': best_r2,
                                'a':  best_coeffs[0],
                                'b':  best_coeffs[1],
                            }
    
        _suggest_for_layers(self.msg_layers,  'msg_layers')
        _suggest_for_layers(self.node_layers, 'node_layers')
        self.train()
        return suggestions
        
    def print_symbolic_suggestions(self, suggestions: dict):
        """Pretty print suggestions from suggest_symbolic()"""
        for layer_key in ['msg_layers', 'node_layers']:
            print(f"\n{layer_key}:")
            for layer_idx, edges in suggestions[layer_key].items():
                print(f"  Layer {layer_idx}:")
                if not edges:
                    print("    No strong symbolic matches found")
                for (in_i, out_i), info in edges.items():
                    print(f"    edge ({in_i}→{out_i}): "
                        f"{info['a']:.3f} * {info['fn']} + {info['b']:.3f}  "
                        f"(R²={info['r2']:.4f})")


    def regularization(
        self,
    ) -> torch.Tensor:
        """
        Compute KAN sparsity regularization over all msg and node layers.
        Mirrors pykan's MultKAN.get_reg() but operates on raw KANLayer instances.
        """
        reg = torch.tensor(0.0, device=next(self.parameters()).device)

        all_layers = list(self.msg_layers) + list(self.node_layers)

        for layer in all_layers:
            # acts_scale shape: (out_dim, in_dim)
            # only available after a forward pass has been run
            if not hasattr(layer, 'acts_scale'):
                continue

            acts = layer.acts_scale  # mean |activation| per spline edge

            # L1 — total activation magnitude, pushes edges toward zero
            l1 = torch.sum(torch.mean(acts, dim=0))

            # Entropy — encourages each output to depend on ONE input strongly
            # low entropy = one dominant spline = interpretable
            acts_normalized = acts / (torch.sum(acts, dim=0, keepdim=True) + 1e-8)
            entropy = -torch.sum(
                acts_normalized * torch.log(acts_normalized + 1e-8), dim=0
            ).mean()

            reg += self.lamb_l1 * l1 + self.lamb_entropy * entropy

        return reg

    def summary(self):
        super().summary()

        print(f"  Grid size:     {self.grid_size}")
        print(f"  Spline order:  {self.spline_order}")
        print(f"  KAN layers:    {len(self.msg_layers)} msg, "
                f"{len(self.node_layers)} node")
        print(f"  L1 regularization: {self.lamb_l1}")
        print(f"  Entropy regularization: {self.lamb_entropy}")
        print()
        print("  msg_layers:")
        for i, layer in enumerate(self.msg_layers):
            n = sum(p.numel() for p in layer.parameters())
            print(f"    [{i}]  {layer.in_dim:>4} → {layer.out_dim:<4}  "
                    f"params: {n:,}")
        print("  node_layers:")
        for i, layer in enumerate(self.node_layers):
            n = sum(p.numel() for p in layer.parameters())
            print(f"    [{i}]  {layer.in_dim:>4} → {layer.out_dim:<4}  "
                    f"params: {n:,}")
        print("=" * 60)




class OrdinaryGraphKAN(OrdinaryMixin, GraphKAN):
    """
    Ordinary Graph-KAN with position augmentation and loss computation.

    This is analogous to the OGN (Ordinary Graph Network) in the baseline,
    but using KAN layers. Architecture matches baseline: 4 layers, 300 hidden.

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
    grid_size : int, optional (default=5)
        KAN: Number of B-spline grid intervals (must be >= 1)
    spline_order : int, optional (default=3)
        KAN: B-spline order (must be >= 1)
    aggr : str, optional (default='add')
        Aggregation method
    sparse_init : bool, optional (default=False)
        KAN: Whether to use sparse initialization for layer connectivity
    """

    def __init__(
            self,
            n_f: int,
            msg_dim: int,
            ndim: int,
            edge_index: torch.Tensor,
            hidden: int = 300,
            node_hidden: int = 300,
            grid_size: int = 5,
            spline_order: int = 3,
            aggr: str = "add",
            hidden_layers: int = 0,
            node_hidden_layers: int = 0,
            lamb_l1: float = 1.0,
            lamb_entropy: float = 2.0,
            sparse_init: bool = False
    ):
        super().__init__(
            n_f, msg_dim, ndim, hidden, node_hidden, grid_size, spline_order, aggr, hidden_layers, node_hidden_layers, lamb_l1, lamb_entropy, sparse_init
        )
        self.register_buffer("edge_index", edge_index)

    # just_derivative() and loss() are inherited from OrdinaryMixin
