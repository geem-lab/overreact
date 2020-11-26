#!/usr/bin/env python3

"""Tests for simulate module."""

import numpy as np
import pytest

from overreact import core
from overreact import simulate


def test_get_dydt_calculates_reaction_rate():
    """Ensure get_dydt gives correct reaction rates."""
    scheme = core.Scheme(
        compounds=["A", "B"],
        reactions=["A -> B"],
        is_half_equilibrium=np.array([False]),
        A=np.array([[-1.0], [1.0]]),
        B=np.array([[-1.0], [1.0]]),
    )

    dydt = simulate.get_dydt(scheme, [2.0])

    # TODO(schneiderfelipe): if jax is used, dydt won't accept lists, only
    # ndarrays. Should we wrap the jitted code and use np.asanyarray in the
    # wrapping function?
    assert dydt(0.0, np.array([1.0, 0.0])) == pytest.approx([-2.0, 2.0])
    assert dydt(5.0, np.array([1.0, 0.0])) == pytest.approx([-2.0, 2.0])
    assert dydt(0.0, np.array([1.0, 1.0])) == pytest.approx([-2.0, 2.0])
    assert dydt(0.0, np.array([10.0, 0.0])) == pytest.approx([-20.0, 20.0])


def test_get_y_propagates_reaction_automatically():
    """Ensure get_y properly propagates reactions with automatic time span."""
    scheme = core.Scheme(
        compounds=["A", "B", "AB4"],
        reactions=["A + 4 B -> AB4", "AB4 -> A + 4 B"],
        is_half_equilibrium=np.array([True, True]),
        A=np.array([[-1.0, 1.0], [-4.0, 4.0], [1.0, -1.0]]),
        B=np.array([[-1.0, 0.0], [-4.0, 0.0], [1.0, 0.0]]),
    )
    y0 = [2.00, 2.00, 0.01]

    y, r = simulate.get_y(simulate.get_dydt(scheme, [1.0, 1.0]), y0=y0)

    assert y.t_min == 0.0
    assert y.t_max == 1000.0
    assert y(y.t_min) == pytest.approx(y0)
    assert y(y.t_max) == pytest.approx(
        [1.668212890625, 0.6728515625, 0.341787109375], 9e-5
    )
    assert r(y.t_min) == pytest.approx([-31.99, -127.96, 31.99])
    assert r(y.t_max) == pytest.approx([0.0, 0.0, 0.0], abs=2e-4)


def test_get_y_propagates_reaction_with_fixed_time():
    """Ensure get_y properly propagates reactions when given time span."""
    scheme = core.Scheme(
        compounds=["A", "B", "AB4"],
        reactions=["A + 4 B -> AB4", "AB4 -> A + 4 B"],
        is_half_equilibrium=np.array([True, True]),
        A=np.array([[-1.0, 1.0], [-4.0, 4.0], [1.0, -1.0]]),
        B=np.array([[-1.0, 0.0], [-4.0, 0.0], [1.0, 0.0]]),
    )
    y0 = [2.00, 2.00, 0.01]
    t_span = [0.0, 200.0]

    y, r = simulate.get_y(simulate.get_dydt(scheme, [1.0, 1.0]), y0=y0, t_span=t_span)

    assert y.t_min == t_span[0]
    assert y.t_max == t_span[-1]
    assert y(y.t_min) == pytest.approx(y0)
    assert y(y.t_max) == pytest.approx(
        [1.668212890625, 0.6728515625, 0.341787109375], 9e-5
    )
    assert r(y.t_min) == pytest.approx([-31.99, -127.96, 31.99])
    assert r(y.t_max) == pytest.approx([0.0, 0.0, 0.0], abs=3e-6)


def test_get_y_conservation_in_equilibria():
    """Ensure get_y properly conserves matter in a toy equilibrium."""
    scheme = core.parse_reactions("A <=> B")
    y0 = [1, 0]

    y, r = simulate.get_y(simulate.get_dydt(scheme, [1, 1]), y0=y0)
    t = np.linspace(y.t_min, y.t_max, num=100)

    assert y.t_min == 0.0
    assert y.t_max == 10.0
    assert y(y.t_min) == pytest.approx(y0)
    assert y(y.t_max) == pytest.approx([0.5, 0.5], 3e-5)
    assert r(y.t_min) == pytest.approx([-1, 1])
    assert r(y.t_max) == pytest.approx([0.0, 0.0], abs=3e-5)

    assert y.t_min == t[0]
    assert y.t_max == t[-1]
    assert np.allclose(y(t)[0] + y(t)[1], np.sum(y0))
    assert np.allclose(r(t)[0] + r(t)[1], 0.0)
