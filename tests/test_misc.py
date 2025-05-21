"""Tests for misc module."""

from __future__ import annotations

import numpy as np
import pytest

import overreact as rx


def test_broaden_spectrum_works() -> None:
    """Ensure we can broad a simple spectrum."""
    x = np.linspace(50, 200, num=15)
    s = rx._misc.broaden_spectrum(x, [150, 100], [2, 1], scale=20.0)
    assert s == pytest.approx(
        [
            0.04316864,
            0.1427911,
            0.35495966,
            0.66562891,
            0.95481719,
            1.09957335,
            1.16005667,
            1.34925384,
            1.72176729,
            2.0,
            1.85988866,
            1.32193171,
            0.70860709,
            0.28544362,
            0.0863263,
        ],
    )


def test_number_points_central_diff() -> None:
    """Ensure that number of points is odd and less than number of divisions."""
    with pytest.raises(ValueError, match=r"^Number of points must be at least\s*"):
        rx._misc._central_diff_weights(Np=1, ndiv=2)

    with pytest.raises(ValueError, match=r"^The number of points must be odd"):
        rx._misc._central_diff_weights(Np=2, ndiv=1)


def test_derivative_order() -> None:
    """Ensures that 'order' is less than 'n', and, odd."""
    with pytest.raises(ValueError, match=r"^'order'\s*"):
        rx._misc._derivative(np.sin, x0=0, n=3, order=1)

    with pytest.raises(ValueError, match=r"\ must be odd.$"):
        rx._misc._derivative(np.sin, x0=0, n=2, order=4)


def test_first_derivative() -> None:
    """Confirms the right value for the first derivative of a function."""
    for order in [3, 5, 7, 9]:
        first_derivative = rx._misc._derivative(np.sin, x0=np.pi / 2, n=1, order=order)
        assert first_derivative == pytest.approx(np.cos(np.pi / 2))


# TODO(mrauen): I couldn't find a case in overreact where we use the second (or higher) derivatives. Therefore, I think we can delete this piece of code...Or maybe just leave it here for the future implementations (who knows)
def test_second_derivative() -> None:
    """Confirms the right value for the second derivative of a function."""
    for order in [3, 5, 7, 9]:
        second_derivative = rx._misc._derivative(np.sin, x0=0, n=2, order=order)
        assert second_derivative == pytest.approx(-np.sin(0))


def test_high_order_derivative() -> None:
    """Confirms the right value for the nth derivative of a function."""
    first_derivative_high_order = rx._misc._derivative(
        np.sin, x0=np.pi / 2, n=1, order=11,
    )
    assert first_derivative_high_order == pytest.approx(np.cos(np.pi / 2), rel=1e-3)

    second_derivative_high_order = rx._misc._derivative(
        np.sin, x0=np.pi / 2, n=2, order=11,
    )
    assert second_derivative_high_order == pytest.approx(-np.sin(np.pi / 2), rel=1e-3)


@rx._misc.copy_unhashable()
def cached_function(data):
    """Ensure 'copy_unhashable()' is working on the applied functions."""
    return data


def test_empty_data() -> None:
    """Check if an empty np.array is returned when empty structures are passed."""
    resultant_data = cached_function(((), ()))
    assert isinstance(resultant_data, np.ndarray)
    assert resultant_data.size == 0


def test_zero_dimension() -> None:
    """Check if an empty np.array is returned when structures without shape are passed."""
    resultant_shape = cached_function(((1, 2), ()))
    assert isinstance(resultant_shape, np.ndarray)
    assert resultant_shape.shape == (0,)
    assert resultant_shape.size == 0


def test_reshape_error() -> None:
    """Ensure a ValueError is raised when reshaping fails due to invalid data and shape mismatch."""
    with pytest.raises(ValueError, match=r"^Reshape error:\s*"):
        cached_function(((2, 2), (-1, 2)))
