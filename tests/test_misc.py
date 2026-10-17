"""Tests for misc module."""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st

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


@given(st.data())
def test_derivative_is_exact_for_polynomials(data) -> None:
    """Ensure central differences of order m are exact for polynomials of degree below m."""
    order = data.draw(st.sampled_from([3, 5, 7, 9, 11]))
    n = data.draw(st.integers(min_value=1, max_value=order - 1))
    coefficients = data.draw(
        st.lists(st.integers(min_value=-5, max_value=5), min_size=1, max_size=order),
    )
    x0 = data.draw(st.floats(min_value=-2.0, max_value=2.0))
    polynomial = np.polynomial.Polynomial(coefficients)
    assert rx._misc._derivative(polynomial, x0=x0, n=n, order=order) == pytest.approx(
        polynomial.deriv(n)(x0),
        rel=1e-6,
        abs=1e-6,
    )


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
