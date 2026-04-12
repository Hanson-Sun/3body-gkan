"""
Graph-KAN: Graph Neural Network with KAN layers instead of MLPs.

Uses pykan's MultKAN implementation (KAN alias). Widths are required and
use pykan's native ``width`` format (e.g., ``[in_dim, [h, m], out]``); any
multiplication nodes are encoded inline via ``[linear, mult]`` entries.
"""

from typing import Optional, Sequence, Any
import json
import copy
import warnings

import torch
import torch.nn as nn
from kan import KAN
from torch_geometric.nn import MessagePassing

from .ordinary_mixin import OrdinaryMixin
from .graph_mixin import GraphMixin
from .symbolic_mixin import SymbolicGraphKANMixin


class GraphKAN(SymbolicGraphKANMixin, MessagePassing, GraphMixin):
    """
    Graph Neural Network using KAN layers for message and node update functions.

    Widths are required and use pykan's native ``width`` format. Multiplicative
    node counts come from ``[linear, mult]`` entries in the width lists; arities
    remain explicit via ``msg_mult_arity`` / ``node_mult_arity``.

    Parameters
    ----------
    n_f : int
        Number of node features
    msg_width : Sequence
        pykan width spec for the message subnet. Must start with ``2 * n_f``
        and end with the message output dimension.
    node_width : Sequence
        pykan width spec for the node subnet. Must start with ``n_f + msg_dim``
        and end with the output dimension ``ndim``.
    msg_dim : int, optional
        Optional explicit message output dimension; inferred from ``msg_width``.
    ndim : int, optional
        Optional explicit node output dimension; inferred from ``node_width``.
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
    - Width specifications must include input and output dimensions.
    - Multiplication nodes are encoded inline via ``[linear, mult]`` entries.
    - KAN-specific parameters (grid_size, spline_order) remain configurable.
    """

    def __init__(
            self,
            n_f: int,
            msg_width: Sequence,
            node_width: Sequence,
            msg_mult_arity: int | Sequence[int] | Sequence[Sequence[int]] = 2,
            node_mult_arity: int | Sequence[int] | Sequence[Sequence[int]] = 2,
            grid_size: int = 5,
            spline_order: int = 3,
            aggr: str = "add",
            lamb_l1: float = 1.0,
            lamb_entropy: float = 2.0,
            sparse_init: bool = True,
            base_fun: Optional[str] = None,
            noise_scale: Optional[float] = None,
            scale_base_mu: Optional[float] = None,
            scale_base_sigma: Optional[float] = None,
            msg_dim: Optional[int] = None,
            ndim: Optional[int] = None,
            **_: Any,
        ):
        super().__init__(aggr=aggr)
        self.n_f = n_f
        self.grid_size = self._validate_positive_int(grid_size, "grid_size")
        self.spline_order = self._validate_positive_int(spline_order, "spline_order")
        self.lamb_l1 = float(lamb_l1)
        self.lamb_entropy = float(lamb_entropy)
        self.sparse_init = bool(sparse_init)

        inferred_msg_dim = self._extract_output_dim(msg_width, "msg_width")
        if msg_dim is None:
            msg_dim = inferred_msg_dim
        elif msg_dim != inferred_msg_dim:
            raise ValueError(
                f"msg_dim ({msg_dim}) must match msg_width output ({inferred_msg_dim})."
            )
        inferred_ndim = self._extract_output_dim(node_width, "node_width")
        if ndim is None:
            ndim = inferred_ndim
        elif ndim != inferred_ndim:
            raise ValueError(
                f"ndim ({ndim}) must match node_width output ({inferred_ndim})."
            )
        self.msg_dim = msg_dim
        self.ndim = ndim
        self._grid_updates_disabled = False
        msg_input_dim = 2 * n_f

        # Message function: [x_i, x_j] (2*n_f) → msg_dim
        self.msg_width = self._normalize_width_spec(
            width=msg_width,
            in_dim=msg_input_dim,
            out_dim=self.msg_dim,
            label="msg_width",
        )
        self.msg_mult_arity = self._normalize_mult_arity(
            mult_arity=msg_mult_arity,
            width=self.msg_width,
            label="msg_mult_arity",
        )
        self.msg_mult_nodes = self._max_mult_nodes_in_width(self.msg_width)
        self.msg_kan = self._build_multkan_network(
            width=self.msg_width,
            grid_size=grid_size,
            spline_order=spline_order,
            mult_arity=self.msg_mult_arity,
            sparse_init=sparse_init,
            base_fun=base_fun,
            noise_scale=noise_scale,
            scale_base_mu=scale_base_mu,
            scale_base_sigma=scale_base_sigma,
        )
        self.msg_layers = self.msg_kan.act_fun

        # Node update function: [x, aggr_msgs] (n_f + msg_dim) → ndim
        node_input_dim = n_f + msg_dim
        self.node_width = self._normalize_width_spec(
            width=node_width,
            in_dim=node_input_dim,
            out_dim=self.ndim,
            label="node_width",
        )
        self.node_mult_arity = self._normalize_mult_arity(
            mult_arity=node_mult_arity,
            width=self.node_width,
            label="node_mult_arity",
        )
        self.node_mult_nodes = self._max_mult_nodes_in_width(self.node_width)
        self.node_kan = self._build_multkan_network(
            width=self.node_width,
            grid_size=grid_size,
            spline_order=spline_order,
            mult_arity=self.node_mult_arity,
            sparse_init=sparse_init,
            base_fun=base_fun,
            noise_scale=noise_scale,
            scale_base_mu=scale_base_mu,
            scale_base_sigma=scale_base_sigma,
        )
        self.node_layers = self.node_kan.act_fun

    @staticmethod
    def _validate_positive_int(value: int, name: str) -> int:
        if isinstance(value, bool):
            raise ValueError(f"{name} must be an integer, got boolean {value!r}.")
        try:
            ivalue = int(value)
        except Exception as exc:
            raise ValueError(f"{name} must be an integer, got {value!r}.") from exc
        if isinstance(value, (float, complex)) and not float(value).is_integer():
            raise ValueError(f"{name} must be an integer, got {value!r}.")
        if ivalue <= 0:
            raise ValueError(f"{name} must be > 0 (got {ivalue}).")
        return ivalue

    @staticmethod
    def _validate_nonnegative_int(value: int, name: str) -> int:
        if isinstance(value, bool):
            raise ValueError(f"{name} must be an integer, got boolean {value!r}.")
        try:
            ivalue = int(value)
        except Exception as exc:
            raise ValueError(f"{name} must be an integer, got {value!r}.") from exc
        if isinstance(value, (float, complex)) and not float(value).is_integer():
            raise ValueError(f"{name} must be an integer, got {value!r}.")
        if ivalue < 0:
            raise ValueError(f"{name} must be >= 0 (got {ivalue}).")
        return ivalue

    @staticmethod
    def _extract_output_dim(width: Sequence, label: str) -> int:
        if width is None:
            raise ValueError(f"{label} is required and must be a sequence of layer sizes.")
        if not isinstance(width, Sequence) or isinstance(width, (str, bytes)):
            raise ValueError(f"{label} must be a sequence, got {type(width).__name__}.")
        if len(width) < 2:
            raise ValueError(f"{label} must include at least input and output dimensions.")
        last = width[-1]
        if isinstance(last, Sequence) and not isinstance(last, (str, bytes)):
            if len(last) != 2:
                raise ValueError(
                    f"{label} last entry must be a scalar or [linear, mult]; got {last!r}."
                )
            return int(last[0])
        return int(last)

    @staticmethod
    def _normalize_width_spec(
        width: Sequence,
        in_dim: int,
        out_dim: int,
        label: str,
    ) -> list:
        """Validate and normalize a pykan-style width specification."""
        if width is None:
            raise ValueError(f"{label} is required.")
        if isinstance(width, str):
            raise ValueError(
                f"{label} must be a sequence like [in, [hidden, mult], out]; "
                "got string input."
            )
        if not isinstance(width, Sequence):
            raise ValueError(f"{label} must be a sequence, got {type(width).__name__}.")

        if len(width) < 2:
            raise ValueError(f"{label} must include at least input and output dimensions.")

        normalized: list = []
        for idx, entry in enumerate(width):
            if isinstance(entry, Sequence) and not isinstance(entry, (str, bytes)):
                if len(entry) != 2:
                    raise ValueError(
                        f"{label}[{idx}] must be a 2-item sequence [width, mult_nodes]; "
                        f"got {entry!r}."
                    )
                linear, mult = entry
                linear_v = GraphKAN._validate_positive_int(linear, f"{label}[{idx}][0]")
                mult_v = GraphKAN._validate_nonnegative_int(mult, f"{label}[{idx}][1]")
                normalized.append([linear_v, mult_v])
            else:
                linear_v = GraphKAN._validate_positive_int(entry, f"{label}[{idx}]")
                normalized.append(linear_v)

        first = normalized[0][0] if isinstance(normalized[0], Sequence) else normalized[0]
        if first != in_dim:
            raise ValueError(
                f"{label} must start with input dimension {in_dim} (got {normalized[0]})."
            )
        last = normalized[-1][0] if isinstance(normalized[-1], Sequence) else normalized[-1]
        if last != out_dim:
            raise ValueError(
                f"{label} must end with output dimension {out_dim} (got {normalized[-1]})."
            )

        return copy.deepcopy(normalized)

    @staticmethod
    def _call_prune_method(
        subnet: nn.Module,
        method_name: str,
        threshold: Optional[float],
        log_history: bool = True,
    ) -> nn.Module:
        if threshold is None:
            return subnet
        method = getattr(subnet, method_name, None)
        if method is None:
            return subnet
        result = method(threshold=float(threshold), log_history=log_history)
        return subnet if result is None else result

    def _refresh_subnet_metadata(self):
        self.msg_layers = self.msg_kan.act_fun
        self.node_layers = self.node_kan.act_fun
        self.msg_width = copy.deepcopy(getattr(self.msg_kan, "width", self.msg_width))
        self.node_width = copy.deepcopy(getattr(self.node_kan, "width", self.node_width))
        self.msg_mult_nodes = self._max_mult_nodes_in_width(self.msg_width)
        self.node_mult_nodes = self._max_mult_nodes_in_width(self.node_width)

    def prune_subnets(
        self,
        edge_threshold: Optional[float] = 3e-2,
        node_threshold: Optional[float] = None,
        log_history: bool = True,
    ) -> dict[str, Any]:
        """
        Prune GraphKAN message and node subnets using pykan pruning APIs.

        Parameters
        ----------
        edge_threshold : float or None
            Threshold for ``prune_edge`` on both subnets. Set None to skip.
        node_threshold : float or None
            Threshold for ``prune_node`` on both subnets. Set None to skip.
        log_history : bool
            Forwarded to pykan pruning methods.

        Returns
        -------
        dict
            Summary of updated subnet widths.
        """
        self.msg_kan = self._call_prune_method(
            self.msg_kan,
            "prune_edge",
            edge_threshold,
            log_history=log_history,
        )
        self.node_kan = self._call_prune_method(
            self.node_kan,
            "prune_edge",
            edge_threshold,
            log_history=log_history,
        )

        self.msg_kan = self._call_prune_method(
            self.msg_kan,
            "prune_node",
            node_threshold,
            log_history=log_history,
        )
        self.node_kan = self._call_prune_method(
            self.node_kan,
            "prune_node",
            node_threshold,
            log_history=log_history,
        )

        self._refresh_subnet_metadata()
        return {
            "msg_width": copy.deepcopy(self.msg_width),
            "node_width": copy.deepcopy(self.node_width),
        }

    @staticmethod
    def _max_mult_nodes_in_width(width: Sequence) -> int:
        max_mult = 0
        for entry in width:
            if isinstance(entry, Sequence) and not isinstance(entry, (str, bytes)) and len(entry) == 2:
                try:
                    mult_val = int(entry[1])
                except Exception:
                    continue
                if mult_val > max_mult:
                    max_mult = mult_val
        return max_mult

    @staticmethod
    def _layer_mult_counts(width: Sequence) -> list[int]:
        counts: list[int] = []
        for entry in width:
            if isinstance(entry, Sequence) and not isinstance(entry, (str, bytes)) and len(entry) == 2:
                try:
                    counts.append(int(entry[1]))
                except Exception:
                    counts.append(0)
            else:
                counts.append(0)
        return counts

    @staticmethod
    def _normalize_mult_arity_layer(entry: Any, mult_nodes: int, label: str) -> list[int]:
        if mult_nodes <= 0:
            return []

        def _canonicalize_arity(raw_value: Any, arity_label: str) -> int:
            val = GraphKAN._validate_positive_int(raw_value, arity_label)
            # pykan MultKAN forward assumes multiplicative arity >= 2.
            # Arity=1 can trigger an UnboundLocalError in upstream code.
            if val == 1:
                warnings.warn(
                    f"{arity_label}=1 is unsupported by pykan multiplication nodes; "
                    "promoting to 2."
                )
                return 2
            return val

        if isinstance(entry, bool):
            raise ValueError(f"{label} must be an int or list of ints; got boolean {entry!r}.")
        if isinstance(entry, (int, float)) and not isinstance(entry, bool):
            if isinstance(entry, (float, complex)) and not float(entry).is_integer():
                raise ValueError(f"{label} must be an int or list of ints; got {entry!r}.")
            val = _canonicalize_arity(int(entry), label)
            return [val] * mult_nodes
        if isinstance(entry, str):
            try:
                parsed = json.loads(entry)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{label} must be an int or list of ints; got {entry!r}.") from exc
            return GraphKAN._normalize_mult_arity_layer(parsed, mult_nodes, label)
        if not (isinstance(entry, Sequence) and not isinstance(entry, (str, bytes))):
            raise ValueError(f"{label} must be an int or list (len {mult_nodes}); got {type(entry).__name__}.")

        layer_arities = [
            _canonicalize_arity(val, f"{label}[{idx}]")
            for idx, val in enumerate(entry)
        ]
        if len(layer_arities) != mult_nodes:
            raise ValueError(
                f"{label} length ({len(layer_arities)}) must equal number of multiplication nodes ({mult_nodes})."
            )
        return layer_arities

    @staticmethod
    def _normalize_mult_arity(mult_arity: Any, width: Sequence, label: str) -> int | list[list[int]]:
        layer_counts = GraphKAN._layer_mult_counts(width)
        has_mult_nodes = any(count > 0 for count in layer_counts)

        if mult_arity is None:
            raise ValueError(f"{label} is required; got None.")
        if isinstance(mult_arity, bool):
            raise ValueError(f"{label} must be an int or list, got boolean {mult_arity!r}.")
        if isinstance(mult_arity, (int, float)) and not isinstance(mult_arity, bool):
            if isinstance(mult_arity, (float, complex)) and not float(mult_arity).is_integer():
                raise ValueError(f"{label} must be an int or list-of-lists; got {mult_arity!r}.")
            arity = GraphKAN._validate_positive_int(int(mult_arity), label)
            if has_mult_nodes and arity == 1:
                warnings.warn(
                    f"{label}=1 is unsupported by pykan multiplication nodes; promoting to 2."
                )
                arity = 2
            return arity
        if isinstance(mult_arity, str):
            try:
                parsed = json.loads(mult_arity)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"{label} must be an int or list-of-lists; failed to parse {mult_arity!r}."
                ) from exc
            return GraphKAN._normalize_mult_arity(parsed, width, label)
        if not (isinstance(mult_arity, Sequence) and not isinstance(mult_arity, (str, bytes))):
            raise ValueError(f"{label} must be an int or list-of-lists; got {type(mult_arity).__name__}.")

        if len(mult_arity) != len(layer_counts):
            raise ValueError(
                f"{label} length ({len(mult_arity)}) must match width length ({len(layer_counts)})."
            )

        normalized: list[list[int]] = []
        for layer_idx, (entry, mult_nodes) in enumerate(zip(mult_arity, layer_counts)):
            if mult_nodes == 0:
                if entry in (None, [], ()):  # allow empty/None for layers without mult nodes
                    normalized.append([])
                    continue
                if isinstance(entry, Sequence) and not isinstance(entry, (str, bytes)) and len(entry) == 0:
                    normalized.append([])
                    continue
                raise ValueError(
                    f"{label}[{layer_idx}] must be empty because layer has 0 multiplication nodes (got {entry!r})."
                )

            normalized.append(
                GraphKAN._normalize_mult_arity_layer(entry, mult_nodes, f"{label}[{layer_idx}]")
            )

        return normalized

    def _build_multkan_network(
            self,
            width: list,
            grid_size: int,
            spline_order: int,
            mult_arity: int | Sequence,
            sparse_init: bool = True,
            base_fun: Optional[str] = None,
            noise_scale: Optional[float] = None,
            scale_base_mu: Optional[float] = None,
            scale_base_sigma: Optional[float] = None,
    ) -> nn.Module:
        """
        Build a pykan MultKAN numerical network.

        Parameters
        ----------
        width : list
            MultKAN width specification
        grid_size : int
            B-spline grid size (num parameter in pykan)
        spline_order : int
            B-spline order (k parameter in pykan)
        mult_arity : int or list of lists
            Multiplication arity (number of inputs) per multiplication node. Can
            be a single int (broadcast) or a list-of-lists matching the width
            specification.
        sparse_init : bool
            Whether to sparsify connections at initialization.

        Returns
        -------
        nn.Module
            KAN/MultKAN model
        """
        kw = {
            "width": width,
            "grid": grid_size,
            "k": spline_order,
            "mult_arity": mult_arity,
            "sparse_init": sparse_init,
            "symbolic_enabled": False,
            "save_act": True,
            "auto_save": False,
        }
        if base_fun is not None:
            kw["base_fun"] = base_fun
        if noise_scale is not None:
            kw["noise_scale"] = noise_scale
        if scale_base_mu is not None:
            kw["scale_base_mu"] = scale_base_mu
        if scale_base_sigma is not None:
            kw["scale_base_sigma"] = scale_base_sigma

        model = KAN(**kw)
        # Keep symbolic branch disabled and frozen; GraphKAN uses numerical branch only.
        for param in model.symbolic_fun.parameters():
            param.requires_grad_(False)
        return model

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
        return self.msg_kan(tmp)

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
        return self.node_kan(tmp)

    def update_grids(
        self,
        data_loader,
        device: torch.device | str = 'cpu',
        max_batches: int = 10,
        jitter_scale: float = 1e-2,
        corr_threshold: float = 0.9,
    ):
        """
        Update message/node MultKAN grids from sampled training batches.

        pykan handles per-layer grid adaptation internally in
        ``update_grid_from_samples``. We only provide representative first-layer
        inputs for each subnet:
        - message subnet input: [x_i, x_j]
        - node subnet input: [x, aggregated_messages]

        Parameters
        ----------
        data_loader : DataLoader
            Training data loader
        device : str or torch.device
            Device to run on
        max_batches : int, optional (default=10)
            Maximum number of batches to use for grid adaptation
        jitter_scale : float, optional (default=1e-2)
            Scale for Gaussian jitter to decorrelate highly aligned features
        corr_threshold : float, optional (default=0.9)
            Apply jitter only to features whose absolute correlation exceeds this
            threshold with any other feature
        """
        from torch_scatter import scatter_add

        if self._grid_updates_disabled:
            return

        was_training = self.training
        self.eval()
        # Back up numeric parameters so we can restore on failure/NaN
        backups = {
            'msg': copy.deepcopy(self.msg_kan.state_dict()),
            'node': copy.deepcopy(self.node_kan.state_dict()),
        }
        with torch.no_grad():
            def _with_jitter(t: torch.Tensor) -> torch.Tensor:
                if jitter_scale <= 0:
                    return t
                if t.shape[0] < 2:
                    return t
                t_f = t.float()
                eps = 1e-6
                std = t_f.std(dim=0, keepdim=True)
                centered = t_f - t_f.mean(dim=0, keepdim=True)
                denom = std + eps
                # Correlation matrix; diagonal is ~1, off-diagonals show alignment
                normed = centered / denom
                corr = (normed.T @ normed) / (t_f.shape[0] - 1 + eps)
                corr = corr - torch.eye(corr.size(0), device=t.device)
                max_corr = corr.abs().max(dim=0).values
                mask = max_corr > corr_threshold
                if not mask.any():
                    return t
                noise = torch.randn_like(t_f) * jitter_scale * (std + eps)
                noise[:, ~mask] = 0
                return t + noise

            msg_inputs = []
            for i, batch in enumerate(data_loader):
                if i >= max_batches:
                    break
                batch = batch.to(device)
                x = batch.x
                row, col = batch.edge_index
                msg_inputs.append(torch.cat([x[row], x[col]], dim=1))

            if msg_inputs:
                msg_samples = _with_jitter(torch.cat(msg_inputs, dim=0))
                try:
                    self.msg_kan.update_grid_from_samples(msg_samples)
                except Exception as exc:
                    print(f"[GridUpdate] Message grid update failed: {exc}")
                    self.msg_kan.load_state_dict(backups['msg'])
                    self._grid_updates_disabled = True
                    self.train(was_training)
                    return

            node_inputs = []
            for i, batch in enumerate(data_loader):
                if i >= max_batches:
                    break
                batch = batch.to(device)
                x = batch.x
                row, col = batch.edge_index
                msg_input = torch.cat([x[row], x[col]], dim=1)
                msg_out = self.msg_kan(msg_input)
                aggr_msg = scatter_add(msg_out, row, dim=0, dim_size=x.size(0))
                node_inputs.append(torch.cat([x, aggr_msg], dim=1))

            if node_inputs:
                node_samples = _with_jitter(torch.cat(node_inputs, dim=0))
                try:
                    self.node_kan.update_grid_from_samples(node_samples)
                except Exception as exc:
                    print(f"[GridUpdate] Node grid update failed: {exc}")
                    self.node_kan.load_state_dict(backups['node'])
                    self._grid_updates_disabled = True
                    self.train(was_training)
                    return

            # Detect NaNs post-update; if present, restore and disable further updates
            has_nan = (
                self._has_nan_params(self.msg_kan) or
                self._has_nan_params(self.node_kan)
            )
            if has_nan:
                print("[GridUpdate] NaN detected after grid update; restoring backup and disabling future updates.")
                self.msg_kan.load_state_dict(backups['msg'])
                self.node_kan.load_state_dict(backups['node'])
                self._grid_updates_disabled = True

        self.train(was_training)

    @staticmethod
    def _count_parameters(module: nn.Module, trainable_only: bool = False) -> int:
        if trainable_only:
            return sum(p.numel() for p in module.parameters() if p.requires_grad)
        return sum(p.numel() for p in module.parameters())

    @staticmethod
    def _count_subnet_layer_parameters(subnet: nn.Module, layer_idx: int) -> tuple[int, int, int]:
        numeric = 0
        symbolic = 0

        if hasattr(subnet, 'act_fun') and len(subnet.act_fun) > layer_idx:
            numeric = sum(p.numel() for p in subnet.act_fun[layer_idx].parameters())
        if hasattr(subnet, 'symbolic_fun') and len(subnet.symbolic_fun) > layer_idx:
            symbolic = sum(p.numel() for p in subnet.symbolic_fun[layer_idx].parameters())

        return numeric, symbolic, numeric + symbolic


    def regularization(
        self,
    ) -> torch.Tensor:
        """
        Compute KAN sparsity regularization over message and node MultKANs.
        Uses pykan's native edge-forward regularization metric.
        """
        device = next(self.parameters()).device
        reg = torch.tensor(0.0, device=device)

        for subnet in (self.msg_kan, self.node_kan):
            if not hasattr(subnet, 'acts_scale'):
                continue

            reg = reg + subnet.get_reg(
                reg_metric='edge_forward_sum',
                lamb_l1=self.lamb_l1,
                lamb_entropy=self.lamb_entropy,
                lamb_coef=0.0,
                lamb_coefdiff=0.0,
            )

        return reg

    @staticmethod
    def _has_nan_params(module: nn.Module) -> bool:
        for p in module.parameters():
            if torch.isnan(p).any():
                return True
        return False

    def summary(self):
        super().summary()

        msg_total = self._count_parameters(self.msg_kan)
        msg_trainable = self._count_parameters(self.msg_kan, trainable_only=True)
        node_total = self._count_parameters(self.node_kan)
        node_trainable = self._count_parameters(self.node_kan, trainable_only=True)
        model_total = self._count_parameters(self)
        model_trainable = self._count_parameters(self, trainable_only=True)

        print(f"  Grid size:     {self.grid_size}")
        print(f"  Spline order:  {self.spline_order}")
        print(f"  Msg mult:      {self.msg_mult_nodes} (arity={self.msg_mult_arity})")
        print(f"  Node mult:     {self.node_mult_nodes} (arity={self.node_mult_arity})")
        print(f"  Msg width:     {self.msg_width}")
        print(f"  Node width:    {self.node_width}")
        print(f"  Msg params:    {msg_trainable:,} trainable / {msg_total:,} total")
        print(f"  Node params:   {node_trainable:,} trainable / {node_total:,} total")
        print(f"  Model params:  {model_trainable:,} trainable / {model_total:,} total")
        print(f"  KAN layers:    {len(self.msg_layers)} msg, "
                f"{len(self.node_layers)} node")
        print(f"  L1 regularization: {self.lamb_l1}")
        print(f"  Entropy regularization: {self.lamb_entropy}")
        print()
        print("  msg_layers:")
        for i, layer in enumerate(self.msg_layers):
            n_num, n_sym, n_total = self._count_subnet_layer_parameters(self.msg_kan, i)
            print(
                f"    [{i}]  {layer.in_dim:>4} -> {layer.out_dim:<4}  "
                f"num: {n_num:,}  sym: {n_sym:,}  total: {n_total:,}"
            )
        print("  node_layers:")
        for i, layer in enumerate(self.node_layers):
            n_num, n_sym, n_total = self._count_subnet_layer_parameters(self.node_kan, i)
            print(
                f"    [{i}]  {layer.in_dim:>4} -> {layer.out_dim:<4}  "
                f"num: {n_num:,}  sym: {n_sym:,}  total: {n_total:,}"
            )
        print("=" * 60)




class OrdinaryGraphKAN(OrdinaryMixin, GraphKAN):
    """
    Ordinary Graph-KAN with position augmentation and loss computation.

    This mirrors the baseline OGN but replaces MLPs with KAN subnets. Widths
    must be provided in pykan format for both message and node networks.

    Parameters
    ----------
    n_f : int
        Number of node features
    msg_width : Sequence
        pykan width spec for the message subnet (starts with ``2 * n_f``).
    node_width : Sequence
        pykan width spec for the node subnet (starts with ``n_f + msg_dim``).
    edge_index : torch.Tensor
        Fixed edge indices for the graph, shape (2, n_edges)
    msg_dim : int, optional
        Explicit message dimension; inferred from ``msg_width`` when omitted.
    ndim : int, optional
        Explicit node output dimension; inferred from ``node_width`` when omitted.
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
        msg_width: Sequence,
        node_width: Sequence,
        edge_index: torch.Tensor,
        msg_mult_arity: int | Sequence[int] | Sequence[Sequence[int]] = 2,
        node_mult_arity: int | Sequence[int] | Sequence[Sequence[int]] = 2,
        grid_size: int = 5,
        spline_order: int = 3,
        aggr: str = "add",
        lamb_l1: float = 1.0,
        lamb_entropy: float = 2.0,
        sparse_init: bool = True,
        msg_dim: Optional[int] = None,
        ndim: Optional[int] = None,
        **kwargs: Any,
    ):
        super().__init__(
            n_f=n_f,
            msg_width=msg_width,
            node_width=node_width,
            msg_mult_arity=msg_mult_arity,
            node_mult_arity=node_mult_arity,
            grid_size=grid_size,
            spline_order=spline_order,
            aggr=aggr,
            lamb_l1=lamb_l1,
            lamb_entropy=lamb_entropy,
            sparse_init=sparse_init,
            msg_dim=msg_dim,
            ndim=ndim,
            **kwargs,
        )
        self.register_buffer("edge_index", edge_index)

    # just_derivative() and loss() are inherited from OrdinaryMixin
