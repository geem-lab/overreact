"""Tests for simulate module."""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

import overreact as rx
from overreact import simulate
from tests.strategies import reactions


@settings(deadline=None, max_examples=50)
@given(reactions(), st.data())
def test_get_dydt_follows_mass_action(reaction, data) -> None:
    """Ensure rates are k times the product of concentrations to their coefficients."""
    reactants, products, _, text = reaction
    scheme = rx.parse_reactions(text)
    concentrations = st.floats(min_value=0.1, max_value=10.0)
    y = np.array([data.draw(concentrations) for _ in scheme.compounds])
    k = data.draw(concentrations)

    rate = k * np.prod(
        [y[scheme.compounds.index(c)] ** n for c, n in reactants.items()],
    )
    net = [products.get(c, 0) - reactants.get(c, 0) for c in scheme.compounds]
    dydt = simulate.get_dydt(scheme, np.array([k]))
    assert dydt(0.0, y) == pytest.approx(rate * np.array(net))


def test_get_y_propagates_reaction_automatically() -> None:
    """Ensure get_y properly propagates reactions with automatic time span."""
    scheme = rx.parse_reactions("A + 4 B <=> AB4")
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
    scheme = rx.parse_reactions("A + 4 B <=> AB4")
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
        for kcat in [1e-1, 1e1, 1e10, 1e11, 1e13]
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
        for keq in [1.0, 10.0, 100.0]
        for kcat in [1e-1, 1e1, 1e10, 1e11, 1e13]
        for method in ("RK23", "LSODA", "Radau", "BDF")
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
