"""Tests for logreg.model."""

import numpy as np

from logreg.model import sigmoid


def test_sigmoid_at_zero_is_half() -> None:
    np.testing.assert_allclose(sigmoid(np.array([0.0])), [0.5])


def test_sigmoid_is_symmetric() -> None:
    z = np.linspace(-10.0, 10.0, 41).astype(np.float64)
    np.testing.assert_allclose(sigmoid(-z), 1.0 - sigmoid(z))


def test_sigmoid_extreme_inputs_do_not_overflow() -> None:
    result = sigmoid(np.array([-1000.0, 1000.0]))
    np.testing.assert_allclose(result, [0.0, 1.0], atol=1e-12)
    assert np.all(np.isfinite(result))


def test_sigmoid_preserves_shape() -> None:
    z = np.zeros((3, 4))
    assert sigmoid(z).shape == z.shape


def test_sigmoid_is_bounded_and_increasing() -> None:
    z = np.linspace(-20.0, 20.0, 100).astype(np.float64)
    result = sigmoid(z)
    assert np.all((result >= 0.0) & (result <= 1.0))
    assert np.all(np.diff(result) > 0.0)
