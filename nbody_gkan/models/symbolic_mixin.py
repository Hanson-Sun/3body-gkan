"""
Symbolic regression helper mixin for GraphKAN-like models.

Expects subclasses to provide:
- msg_kan and node_kan MultKAN subnetworks with act_fun/symbolic_fun
- propagate(x, edge_index, ...) to populate activation caches
- training flag and .train() to toggle modes
"""

from __future__ import annotations

from typing import Any

import torch


class SymbolicGraphKANMixin:
    """Utility methods for running and reporting symbolic regression."""

    DEFAULT_SYMBOLIC_LIBRARY = (
        'x', 'x^2', 'x^3', '1/x', '1/x^2',
        'sqrt', 'log', 'abs', 'sin', 'cos', 'exp',
    )

    def suggest_symbolic(
        self,
        data_loader,
        device: torch.device | str = 'cpu',
        lib: list[str] | None = None,
        max_batches: int = 10,
        threshold: float = 0.8,
        fit_affine_after_select: bool = False,
    ) -> dict:
        """
        Suggest symbolic functions per edge using pykan's native API.

        Returns a dict with per-edge symbolic metadata for both subnets.
        Threshold is recorded per-edge via ``passes_threshold`` so downstream
        consumers can either inspect every edge or only high-confidence matches.
        """
        normalized_lib = self._normalize_symbolic_library(lib)

        was_training = self.training
        msg_symbolic_enabled = self.msg_kan.symbolic_enabled
        node_symbolic_enabled = self.node_kan.symbolic_enabled

        self.eval()
        # Symbolic fitting uses cached numeric activations; keep symbolic branch off here.
        self.msg_kan.symbolic_enabled = False
        self.node_kan.symbolic_enabled = False
        try:
            with torch.no_grad():
                for i, batch in enumerate(data_loader):
                    if i >= max_batches:
                        break
                    batch = batch.to(device)
                    self.propagate(
                        batch.edge_index,
                        size=(batch.x.size(0), batch.x.size(0)),
                        x=batch.x,
                    )

            suggestions = {'msg_layers': {}, 'node_layers': {}}

            def _suggest_for_subnet(subnet, layer_key):
                for layer_idx, layer in enumerate(subnet.act_fun):
                    layer_suggestions = {}
                    for in_i in range(layer.in_dim):
                        for out_i in range(layer.out_dim):
                            try:
                                best_fn, _, best_r2, best_c = subnet.suggest_symbolic(
                                    layer_idx,
                                    in_i,
                                    out_i,
                                    lib=normalized_lib,
                                    topk=1,
                                    verbose=False,
                                )
                            except Exception:
                                continue

                            r2_value = float(best_r2)
                            passes_threshold = r2_value >= threshold

                            # Optional extra affine refit for expression readability.
                            # This is expensive and not required for selecting best_fn/r2.
                            a_x, b_x, c, d = 1.0, 0.0, 1.0, 0.0
                            if fit_affine_after_select:
                                try:
                                    subnet.fix_symbolic(
                                        layer_idx,
                                        in_i,
                                        out_i,
                                        best_fn,
                                        fit_params_bool=True,
                                        verbose=False,
                                        log_history=False,
                                    )
                                    # Affine is [a_x, b_x, c, d] in c * f(a_x * x + b_x) + d.
                                    a_x, b_x, c, d = subnet.symbolic_fun[layer_idx].affine[
                                        out_i, in_i
                                    ].detach().tolist()
                                except Exception:
                                    pass
                                finally:
                                    try:
                                        subnet.unfix_symbolic(layer_idx, in_i, out_i, log_history=False)
                                    except Exception:
                                        pass

                            layer_suggestions[(in_i, out_i)] = {
                                'fn': str(best_fn),
                                'r2': r2_value,
                                # Keep legacy keys expected by downstream tools.
                                'a': float(c),
                                'b': float(d),
                                # Additional detail from pykan's affine fit.
                                'ax': float(a_x),
                                'bx': float(b_x),
                                'complexity': float(best_c),
                                'passes_threshold': bool(passes_threshold),
                            }

                    suggestions[layer_key][layer_idx] = layer_suggestions

            _suggest_for_subnet(self.msg_kan, 'msg_layers')
            _suggest_for_subnet(self.node_kan, 'node_layers')
            return suggestions
        finally:
            self.msg_kan.symbolic_enabled = msg_symbolic_enabled
            self.node_kan.symbolic_enabled = node_symbolic_enabled
            self.train(was_training)

    @staticmethod
    def _normalize_symbolic_library(lib: list[str] | None) -> list[str]:
        if lib is None:
            lib = list(SymbolicGraphKANMixin.DEFAULT_SYMBOLIC_LIBRARY)

        alias_map = {
            'sqrt(x)': 'sqrt',
            'log(x)': 'log',
            'exp(x)': 'exp',
            'abs(x)': 'abs',
            'sin(x)': 'sin',
            'cos(x)': 'cos',
            'tan(x)': 'tan',
        }

        normalized: list[str] = []
        for fn_name in lib:
            key = alias_map.get(str(fn_name).strip(), str(fn_name).strip())
            if key and key not in normalized:
                normalized.append(key)
        return normalized

    @staticmethod
    def _to_float(value, default: float = 0.0) -> float:
        if value is None:
            return default
        return float(value)

    @staticmethod
    def _serialize_mult_arity_value(value):
        """Make mult_arity JSON-serializable while preserving structure."""
        if value is None:
            return None
        if isinstance(value, bool):
            return int(value)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            if isinstance(value, float) and float(value).is_integer():
                return int(value)
            return float(value)
        if isinstance(value, (list, tuple)):
            return [SymbolicGraphKANMixin._serialize_mult_arity_value(v) for v in value]
        return value

    def _format_symbolic_expression(self, info: dict, precision: int = 4) -> str:
        fn = str(info.get('fn', '?'))
        a = self._to_float(info.get('a'), 1.0)
        b = self._to_float(info.get('b'), 0.0)
        ax = info.get('ax', None)
        bx = info.get('bx', None)

        if ax is not None and bx is not None:
            ax_f = self._to_float(ax, 1.0)
            bx_f = self._to_float(bx, 0.0)
            return (
                f"{a:.{precision}f} * {fn}({ax_f:.{precision}f} * x + "
                f"{bx_f:.{precision}f}) + {b:.{precision}f}"
            )

        return f"{a:.{precision}f} * {fn} + {b:.{precision}f}"

    def serialize_symbolic_suggestions(
        self,
        suggestions: dict,
        threshold: float,
        lib: list[str] | None = None,
    ) -> dict:
        """Convert symbolic suggestions to a JSON-serializable payload."""
        lib = [] if lib is None else list(lib)

        serializable = {
            'metadata': {
                'threshold': float(threshold),
                'library': lib,
                'msg_width': getattr(self, 'msg_width', None),
                'node_width': getattr(self, 'node_width', None),
                'msg_mult_nodes': int(getattr(self, 'msg_mult_nodes', 0)),
                'node_mult_nodes': int(getattr(self, 'node_mult_nodes', 0)),
                'msg_mult_arity': self._serialize_mult_arity_value(getattr(self, 'msg_mult_arity', 2)),
                'node_mult_arity': self._serialize_mult_arity_value(getattr(self, 'node_mult_arity', 2)),
            },
            'msg_layers': {},
            'node_layers': {},
        }

        for layer_key in ('msg_layers', 'node_layers'):
            for layer_idx, edges in suggestions.get(layer_key, {}).items():
                layer_out = {}
                for (in_i, out_i), info in edges.items():
                    a_val = self._to_float(info.get('a'), 1.0)
                    b_val = self._to_float(info.get('b'), 0.0)
                    entry = {
                        'fn': str(info.get('fn', '')),
                        'r2': round(self._to_float(info.get('r2'), 0.0), 6),
                        'a': round(a_val, 6),
                        'b': round(b_val, 6),
                        'ax': None,
                        'bx': None,
                        'complexity': None,
                        'expression': self._format_symbolic_expression(info, precision=4),
                    }

                    if info.get('ax') is not None:
                        entry['ax'] = round(self._to_float(info.get('ax'), 1.0), 6)
                    if info.get('bx') is not None:
                        entry['bx'] = round(self._to_float(info.get('bx'), 0.0), 6)
                    if info.get('complexity') is not None:
                        entry['complexity'] = round(self._to_float(info.get('complexity'), 0.0), 6)
                    if 'passes_threshold' in info:
                        entry['passes_threshold'] = bool(info.get('passes_threshold'))

                    layer_out[f'{in_i}->{out_i}'] = entry

                serializable[layer_key][str(layer_idx)] = layer_out

        return serializable

    def print_symbolic_suggestions(
        self,
        suggestions: dict,
        threshold: float = 0.0,
        max_edges_per_layer: int | None = 25,
    ):
        """Pretty print symbolic suggestions, optionally filtered by R^2."""
        for layer_key in ('msg_layers', 'node_layers'):
            print(f"\n{layer_key}:")
            layer_dict = suggestions.get(layer_key, {})
            if not layer_dict:
                print("  No suggestions found")
                continue

            for layer_idx, edges in layer_dict.items():
                print(f"  Layer {layer_idx}:")
                if not edges:
                    print("    No symbolic suggestions found")
                    continue

                filtered = [
                    ((in_i, out_i), info)
                    for (in_i, out_i), info in edges.items()
                    if self._to_float(info.get('r2'), 0.0) >= threshold
                ]

                if not filtered:
                    print(f"    No matches with R^2 >= {threshold:.3f}")
                    continue

                filtered.sort(
                    key=lambda item: self._to_float(item[1].get('r2'), 0.0),
                    reverse=True,
                )

                shown = filtered if max_edges_per_layer is None else filtered[:max_edges_per_layer]

                print(
                    f"    Showing {len(shown)}/{len(filtered)} edges "
                    f"with R^2 >= {threshold:.3f}"
                )

                for (in_i, out_i), info in shown:
                    expr = self._format_symbolic_expression(info, precision=3)
                    r2 = self._to_float(info.get('r2'), 0.0)
                    complexity = info.get('complexity', None)
                    complexity_str = (
                        f", c={self._to_float(complexity, 0.0):.1f}"
                        if complexity is not None
                        else ""
                    )

                    print(
                        f"    edge ({in_i}->{out_i}): {expr} "
                        f"(R^2={r2:.4f}{complexity_str})"
                    )

                if max_edges_per_layer is not None and len(filtered) > max_edges_per_layer:
                    print(f"    ... {len(filtered) - max_edges_per_layer} more edges omitted")
