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
    assert dydt(0.0, [1.0, 0.0]) == pytest.approx([-2.0, 2.0])
    assert dydt(5.0, [1.0, 0.0]) == pytest.approx([-2.0, 2.0])
    assert dydt(0.0, [1.0, 1.0]) == pytest.approx([-2.0, 2.0])
    assert dydt(0.0, [10.0, 0.0]) == pytest.approx([-20.0, 20.0])


def test_get_y_propagates_reaction():
    """Ensure get_y properly propagates reactions."""
    scheme = core.Scheme(
        compounds=["A", "B", "AB4"],
        reactions=["A + 4 B -> AB4", "AB4 -> A + 4 B"],
        is_half_equilibrium=np.array([True, True]),
        A=np.array([[-1.0, 1.0], [-4.0, 4.0], [1.0, -1.0]]),
        B=np.array([[-1.0, 0.0], [-4.0, 0.0], [1.0, 0.0]]),
    )
    t, y = simulate.get_y(
        simulate.get_dydt(scheme, [1.0, 1.0]),
        y0=[2.00, 2.00, 0.01],
        t_span=[0.0, 200.0],
    )
    assert t[0] == 0.0
    assert t[-1] == 200.0
    assert y[:, -1] == pytest.approx(
        [1.668212890625, 0.6728515625, 0.341787109375], 1e-3
    )
