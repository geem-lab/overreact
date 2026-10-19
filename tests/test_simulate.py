"""Tests for simulate module."""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

import overreact as rx
from overreact import simulate
from tests.strategies import COMPOUNDS, reactions

CONCENTRATIONS = st.floats(min_value=0.1, max_value=10.0)


def _rates(scheme, data, k=1.0):
    """Draw concentrations and return them with the rate function of a scheme."""
    y = np.array([data.draw(CONCENTRATIONS) for _ in scheme.compounds])
    return y, simulate.get_dydt(scheme, np.array([k]))


@settings(deadline=None, max_examples=30)
@given(reactions(), st.data())
def test_get_dydt_follows_net_stoichiometry(reaction, data) -> None:
    """Ensure concentrations change along products minus reactants."""
    reactants, products, _, text = reaction
    scheme = rx.Scheme.parse(text)
    net = np.array([products.get(c, 0) - reactants.get(c, 0) for c in scheme.compounds])
    y, dydt = _rates(scheme, data)
    rates = np.asarray(dydt(0.0, y))
    rate = rates @ net / max(net @ net, 1)
    assert np.append(rates, rate) == pytest.approx(np.append(rate * net, abs(rate)))


@settings(deadline=None, max_examples=30)
@given(reactions(), st.data())
def test_get_dydt_broadcasts_over_columns(reaction, data) -> None:
    """Ensure rates of states given as columns are those of each state."""
    scheme = rx.Scheme.parse(reaction[-1])
    states = np.column_stack([_rates(scheme, data)[0] for _ in range(3)])
    dydt = simulate.get_dydt(scheme, np.array([1.0]))
    assert np.asarray(dydt(0.0, states)) == pytest.approx(
        np.column_stack([dydt(0.0, state) for state in states.T]),
    )


@settings(deadline=None, max_examples=30)
@given(reactions(), CONCENTRATIONS, st.data())
def test_get_dydt_orders_are_reactant_coefficients(reaction, factor, data) -> None:
    """Ensure scaling a reactant by f scales rates by f to its coefficient (#838)."""
    reactants, _, _, text = reaction
    scheme = rx.Scheme.parse(text)
    compound = data.draw(st.sampled_from(sorted(reactants) or scheme.compounds))
    y, dydt = _rates(scheme, data)
    scaled = y.copy()
    scaled[scheme.compounds.index(compound)] *= factor
    assert dydt(0.0, scaled) == pytest.approx(
        factor ** reactants.get(compound, 0) * dydt(0.0, y),
    )


@settings(deadline=None, max_examples=30)
@given(reactions(), CONCENTRATIONS, st.data())
def test_get_dydt_overall_order_is_molecularity(reaction, factor, data) -> None:
    """Ensure scaling all concentrations by f scales rates by f to the molecularity."""
    scheme = rx.Scheme.parse(reaction[-1])
    y, dydt = _rates(scheme, data)
    assert dydt(0.0, factor * y) == pytest.approx(
        factor ** scheme.molecularity[0] * dydt(0.0, y),
    )


@settings(deadline=None, max_examples=30)
@given(
    reactions(),
    st.dictionaries(st.sampled_from(COMPOUNDS), CONCENTRATIONS),
    st.data(),
)
def test_fix_preserves_rates(reaction, concentrations, data) -> None:
    """Ensure fixed concentrations and the returned factors keep rates unchanged."""
    scheme = rx.Scheme.parse(reaction[-1])
    concentrations = {c: v for c, v in concentrations.items() if c in scheme.compounds}
    y, dydt = _rates(scheme, data)
    for compound, concentration in concentrations.items():
        y[scheme.compounds.index(compound)] = concentration
    fixed, factors = scheme.fix(concentrations)
    free = [scheme.compounds.index(c) for c in fixed.compounds]
    assert simulate.get_dydt(fixed, factors)(0.0, y[free]) == pytest.approx(
        np.asarray(dydt(0.0, y))[free],
    )


@settings(deadline=None, max_examples=30)
@given(reactions(), st.data())
def test_get_dydt_jacobian_is_directional_derivative(reaction, data) -> None:
    """Ensure the Jacobian maps a direction to the derivative of rates along it."""
    scheme = rx.Scheme.parse(reaction[-1])
    y, dydt = _rates(scheme, data)
    direction = np.array(
        [data.draw(st.floats(min_value=-0.01, max_value=0.01)) for _ in y],
    )
    # rates are polynomials of degree at most 9, where 11-point differences are exact
    assert dydt.jac(0.0, y) @ direction == pytest.approx(
        rx._misc._derivative(lambda h: dydt(0.0, y + h * direction), x0=0.0, order=11),
        abs=2e-15 * np.abs(dydt(0.0, y)).max(),
    )


def test_get_y_propagates_reaction_automatically() -> None:
    """Ensure get_y properly propagates reactions with automatic time span."""
    scheme = rx.Scheme.parse("A + 4 B <=> AB4")
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
    scheme = rx.Scheme.parse("A + 4 B <=> AB4")
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
    scheme = rx.Scheme.parse("A <=> B")
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
    scheme = rx.Scheme.parse(
        """
C + S <=> CS        // Pre-equilibrium
CS -> TS‡ -> C + P  // Catalyst is released
""",
    )
    y0 = np.array([cat0, sub0, 0.0, 0.0])

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
    assert np.allclose(y(y.t_max), [cat0, 0.0, 0.0, sub0], atol=2e-5)


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
    scheme = rx.Scheme.parse(
        """
C + S <=> CS    // Pre-equilibrium
CS -> TS‡ -> P  // Catalyst is consumed instead of being released
""",
    )
    y0 = np.array([cat0, sub0, 0.0, 0.0])

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
    assert np.allclose(y(y.t_max), [cat0 - sub0, 0.0, 0.0, sub0], atol=2e-5)
