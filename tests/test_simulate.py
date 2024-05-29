"""Tests for simulate module."""

from __future__ import annotations

import numpy as np
import pytest

import overreact as rx
from overreact import simulate


def test_get_dydt_calculates_reaction_rate() -> None:
    """Ensure get_dydt gives correct reaction rates."""
    scheme = rx.Scheme(
        compounds=["A", "B"],
        reactions=["A -> B"],
        is_half_equilibrium=np.array([False]),
        A=np.array([[-1.0], [1.0]]),
        B=np.array([[-1.0], [1.0]]),
    )

    # with jitted dydt, we need to use np.ndarray
    k = np.array([2.0])
    dydt = simulate.get_dydt(scheme, k)

    # if JAX is used, dydt won't accept lists, only np.ndarray
    assert dydt(0.0, np.array([1.0, 0.0])) == pytest.approx([-2.0, 2.0])
    assert dydt(5.0, np.array([1.0, 0.0])) == pytest.approx([-2.0, 2.0])
    assert dydt(0.0, np.array([1.0, 1.0])) == pytest.approx([-2.0, 2.0])
    assert dydt(0.0, np.array([10.0, 0.0])) == pytest.approx([-20.0, 20.0])


def test_get_y_propagates_reaction_automatically() -> None:
    """Ensure get_y properly propagates reactions with automatic time span."""
    scheme = rx.Scheme(
        compounds=["A", "B", "AB4"],
        reactions=["A + 4 B -> AB4", "AB4 -> A + 4 B"],
        is_half_equilibrium=np.array([True, True]),
        A=np.array([[-1.0, 1.0], [-4.0, 4.0], [1.0, -1.0]]),
        B=np.array([[-1.0, 0.0], [-4.0, 0.0], [1.0, 0.0]]),
    )
    y0 = [2.00, 2.00, 0.01]

    # with jitted dydt, we need to use np.ndarray
    k = np.array([1.0, 1.0])
    y, r = simulate.get_y(simulate.get_dydt(scheme, k), y0=y0)

    assert y.t_min == 0.0
    assert y.t_max >= 300.0
    assert y(y.t_min) == pytest.approx(y0)
    assert y(y.t_max) == pytest.approx(
        [1.668212890625, 0.6728515625, 0.341787109375],
        4e-4,
    )
    assert r(y.t_min) == pytest.approx([-31.99, -127.96, 31.99])
    assert r(y.t_max) == pytest.approx([0.0, 0.0, 0.0], abs=3e-3)


def test_get_y_propagates_reaction_with_fixed_time() -> None:
    """Ensure get_y properly propagates reactions when given time span."""
    scheme = rx.Scheme(
        compounds=["A", "B", "AB4"],
        reactions=["A + 4 B -> AB4", "AB4 -> A + 4 B"],
        is_half_equilibrium=np.array([True, True]),
        A=np.array([[-1.0, 1.0], [-4.0, 4.0], [1.0, -1.0]]),
        B=np.array([[-1.0, 0.0], [-4.0, 0.0], [1.0, 0.0]]),
    )
    y0 = [2.00, 2.00, 0.01]
    t_span = [0.0, 200.0]

    # with jitted dydt, we need to use np.ndarray
    k = np.array([1.0, 1.0])
    y, r = simulate.get_y(
        simulate.get_dydt(scheme, k),
        y0=y0,
        t_span=t_span,
    )

    assert y.t_min == t_span[0]
    assert y.t_max == t_span[-1]
    assert y(y.t_min) == pytest.approx(y0)
    assert y(y.t_max) == pytest.approx(
        [1.668212890625, 0.6728515625, 0.341787109375],
        3e-3,
    )
    assert r(y.t_min) == pytest.approx([-31.99, -127.96, 31.99])
    assert r(y.t_max) == pytest.approx([0.0, 0.0, 0.0], abs=1.3e-2)


def test_get_y_conservation_in_equilibria() -> None:
    """Ensure get_y properly conserves matter in a toy equilibrium."""
    scheme = rx.parse_reactions("A <=> B")
    y0 = [1, 0]

    # with jitted dydt, we need to use np.ndarray
    k = np.array([1, 1])
    y, r = simulate.get_y(simulate.get_dydt(scheme, k), y0=y0)
    t = np.linspace(y.t_min, y.t_max, num=100)

    assert y.t_min == 0.0
    assert y.t_max >= 3.0
    assert y(y.t_min) == pytest.approx(y0)
    assert y(y.t_max) == pytest.approx([0.5, 0.5], 3e-3)
    assert r(y.t_min) == pytest.approx([-1, 1])
    assert r(y.t_max) == pytest.approx([0.0, 0.0], abs=3e-3)

    assert y.t_min == t[0]
    assert y.t_max == t[-1]
    assert np.allclose(y(t)[0] + y(t)[1], np.sum(y0))
    assert np.allclose(r(t)[0] + r(t)[1], 0.0)


@pytest.mark.parametrize(
    ("cat0", "sub0", "keq", "kcat", "method"),
    [
        (cat0, sub0, keq, kcat, method)
        for cat0 in [0.3, 0.4]
        for sub0 in [0.01, 0.02]
        for keq in [1.0, 10.0, 100.0]
        for kcat in [1e1, 1e10, 1e11, 1e13]
        for method in ("RK23", "LSODA", "Radau", "BDF")
    ],
)
def test_simple_michaelis_menten(
    cat0: float,
    sub0: float,
    keq: float,
    kcat: float,
    method: str,
) -> None:
    """
    Test a simple Michaelis-Menten model.

    See <https://en.wikipedia.org/wiki/Michaelis%E2%80%93Menten_kinetics#Model>.
    """
    scheme = rx.parse_reactions(
        """
C + S <=> CS        // Pre-equilibrium
CS -> TS‡ -> C + P  // Catalyst is released
""",
    )
    y0 = np.array([cat0, sub0, 0.0, 0.0, 0.0])

    # with jitted dydt, we need to use np.ndarray
    k = np.array([keq, 1.0, kcat])
    dydt = simulate.get_dydt(scheme, k)

    # equilibrium constant is kept
    assert np.allclose(dydt.k[0] / dydt.k[1], k[0] / k[1])

    # actual reaction does not change
    assert np.allclose(dydt.k[2], k[2])

    y, r = simulate.get_y(dydt, y0=y0, method=method)

    # catalyst is completely regenerated in the end
    # the substrate is completely consumed in the end
    assert np.allclose(y(y.t_max), [cat0, 0.0, 0.0, 0.0, sub0], atol=2e-5)


@pytest.mark.parametrize(
    ("cat0", "sub0", "keq", "kcat", "method"),
    [
        (cat0, sub0, keq, kcat, method)
        for cat0 in [0.3, 0.4]
        for sub0 in [0.01, 0.02]
        for keq in [10.0, 100.0]
        for kcat in [1e10, 1e11]
        for method in ("RK23", "LSODA")
    ],
)
def test_consuming_michaelis_menten(
    cat0: float,
    sub0: float,
    keq: float,
    kcat: float,
    method: str,
) -> None:
    """Test a faulty system as suggested by @bmounssefjr."""
    scheme = rx.parse_reactions(
        """
C + S <=> CS    // Pre-equilibrium
CS -> TS‡ -> P  // Catalyst is consumed instead of being released
""",
    )
    y0 = np.array([cat0, sub0, 0.0, 0.0, 0.0])

    # with jitted dydt, we need to use np.ndarray
    k = np.array([keq, 1.0, kcat])
    dydt = simulate.get_dydt(scheme, k)

    # equilibrium constant is kept
    assert np.allclose(dydt.k[0] / dydt.k[1], k[0] / k[1])

    # actual reaction does not change
    assert np.allclose(dydt.k[2], k[2])

    y, r = simulate.get_y(dydt, y0=y0, method=method)

    # catalyst is completely consumed in the end
    # the substrate is completely consumed in the end
    assert np.allclose(y(y.t_max), [cat0 - sub0, 0.0, 0.0, 0.0, sub0], atol=2e-5)
