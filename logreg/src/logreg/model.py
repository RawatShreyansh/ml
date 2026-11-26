"""Core functions for logistic regression."""

import numpy as np
from numpy.typing import NDArray


def sigmoid(z: NDArray[np.float64]) -> NDArray[np.float64]:
    """Compute 1 / (1 + exp(-z)) elementwise, without overflowing."""
    e = np.exp(-np.abs(z))
    return np.where(z >= 0, 1.0 / (1.0 + e), e / (1.0 + e))
