"""Neural network models for learning N-body dynamics."""

from .baseline_gnn import GN, OGN
from .graph_kan import GraphKAN, OrdinaryGraphKAN
from .ordinary_mixin import OrdinaryMixin

__all__ = ["OrdinaryMixin", "GraphKAN", "OrdinaryGraphKAN", "GN", "OGN"]
