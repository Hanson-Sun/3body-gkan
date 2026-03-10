import numpy as np

def gravity(positions: np.ndarray, masses: np.ndarray, G: float = 1.0,
            softening: float = 1e-2) -> np.ndarray:
    """
    Compute gravitational accelerations for all bodies.

    Parameters
    ----------
    positions : np.ndarray
        Positions of shape (n, dim)
    masses : np.ndarray
        Masses of shape (n,)
    G : float
        Gravitational constant
    softening : float
        Softening parameter to avoid singularities

    Returns
    -------
    np.ndarray
        Accelerations of shape (n, dim)
    """
    n = len(masses)
    dim = positions.shape[1]
    acc = np.zeros((n, dim))

    for i in range(n):
        for j in range(n):
            if i != j:
                r = positions[j] - positions[i]
                r_norm = np.sqrt(np.sum(r**2) + softening**2)
                acc[i] += G * masses[j] * r / r_norm**3

    return acc


def linear_gravity(positions: np.ndarray, masses: np.ndarray, G: float = 1.0,
                   softening: float = 1e-2) -> np.ndarray:
    """
    Compute gravitational accelerations with linear (1/r) falloff.
    Parameters
    ----------
    positions : np.ndarray
        Positions of shape (n, dim)
    masses : np.ndarray
        Masses of shape (n,)
    G : float
        Gravitational constant
    softening : float
        Softening parameter to avoid singularities
    Returns
    -------
    np.ndarray
        Accelerations of shape (n, dim)
    """
    n = len(masses)
    dim = positions.shape[1]
    acc = np.zeros((n, dim))
    for i in range(n):
        for j in range(n):
            if i != j:
                r = positions[j] - positions[i]
                r_norm = np.sqrt(np.sum(r**2) + softening**2)
                acc[i] += G * masses[j] * r / r_norm**2
    return acc


def cubic_gravity(positions: np.ndarray, masses: np.ndarray, G: float = 1.0,
                  softening: float = 1e-2) -> np.ndarray:
    """
    Compute gravitational accelerations with cubic (1/r^3) falloff.
    Parameters
    ----------
    positions : np.ndarray
        Positions of shape (n, dim)
    masses : np.ndarray
        Masses of shape (n,)
    G : float
        Gravitational constant
    softening : float
        Softening parameter to avoid singularities
    Returns
    -------
    np.ndarray
        Accelerations of shape (n, dim)
    """
    n = len(masses)
    dim = positions.shape[1]
    acc = np.zeros((n, dim))
    for i in range(n):
        for j in range(n):
            if i != j:
                r = positions[j] - positions[i]
                r_norm = np.sqrt(np.sum(r**2) + softening**2)
                acc[i] += G * masses[j] * r / r_norm**4
    return acc


def linear_spring(positions: np.ndarray, masses: np.ndarray, k: float = 1.0,
                  rest_length: float = 1.0) -> np.ndarray:
    """
    Compute spring (Hooke's law) accelerations for all bodies.
    Parameters
    ----------
    positions : np.ndarray
        Positions of shape (n, dim)
    masses : np.ndarray
        Masses of shape (n,)
    k : float
        Spring constant
    rest_length : float
        Natural length of the spring between each pair
    Returns
    -------
    np.ndarray
        Accelerations of shape (n, dim)
    """
    n = len(masses)
    dim = positions.shape[1]
    acc = np.zeros((n, dim))
    for i in range(n):
        for j in range(n):
            if i != j:
                r = positions[j] - positions[i]
                r_norm = np.sqrt(np.sum(r**2))
                if r_norm > 0:
                    acc[i] += (k / masses[i]) * (r_norm - rest_length) * r / r_norm
    return acc