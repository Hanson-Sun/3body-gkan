"""
Configurable edge-level feature augmentation for message-passing GNNs.

Features are computed at forward-pass time from node features x_i and x_j,
so no dataset changes are needed. Shared intermediates (dx, r_sq) are
computed lazily to avoid redundant work when multiple features depend on them.
"""

from __future__ import annotations

import torch


# Maps feature name -> number of output dimensions (as a function of ndim).
_FEATURE_DIMS: dict[str, int | str] = {
    "rel_pos": "ndim",
    "dist_sq": 1,
    "inv_dist_cu": 1,
}

AVAILABLE_FEATURES = tuple(_FEATURE_DIMS.keys())


def edge_feature_dim(ndim: int, features: list[str] | None) -> int:
    """Return total number of augmented dimensions for *features*.

    Parameters
    ----------
    ndim : int
        Spatial dimension (e.g. 2 for 2D).
    features : list[str] or None
        Feature names to include. ``None`` or ``[]`` → 0.
    """
    if not features:
        return 0
    total = 0
    for name in features:
        d = _FEATURE_DIMS.get(name)
        if d is None:
            raise ValueError(
                f"Unknown edge feature {name!r}. "
                f"Available: {', '.join(AVAILABLE_FEATURES)}"
            )
        total += ndim if d == "ndim" else d
    return total


def compute_edge_features(
    x_i: torch.Tensor,
    x_j: torch.Tensor,
    ndim: int,
    features: list[str],
    softening: float = 1e-2,
) -> torch.Tensor | None:
    """Compute selected edge features from source/target node features.

    Parameters
    ----------
    x_i, x_j : Tensor, shape ``(n_edges, n_f)``
        Node features for the receiving (i) and sending (j) ends of each edge.
        Layout per node: ``[pos (ndim), vel (ndim), mass (1)]``.
    ndim : int
        Spatial dimension.
    features : list[str]
        Which features to compute (order is preserved in the output).
    softening : float
        Softening parameter for distance-based features.

    Returns
    -------
    Tensor, shape ``(n_edges, n_aug)`` or ``None`` if *features* is empty.
    """
    if not features:
        return None

    # Lazy intermediates — computed at most once.
    dx: torch.Tensor | None = None
    r_sq: torch.Tensor | None = None

    def _dx() -> torch.Tensor:
        nonlocal dx
        if dx is None:
            dx = x_j[:, :ndim] - x_i[:, :ndim]
        return dx

    def _r_sq() -> torch.Tensor:
        nonlocal r_sq
        if r_sq is None:
            r_sq = (_dx() ** 2).sum(dim=1, keepdim=True)
        return r_sq

    parts: list[torch.Tensor] = []
    for name in features:
        if name == "rel_pos":
            parts.append(_dx())
        elif name == "dist_sq":
            parts.append(_r_sq())
        elif name == "inv_dist_cu":
            parts.append(torch.pow(_r_sq() + softening, -1.5))
        else:
            raise ValueError(
                f"Unknown edge feature {name!r}. "
                f"Available: {', '.join(AVAILABLE_FEATURES)}"
            )

    return torch.cat(parts, dim=1)
