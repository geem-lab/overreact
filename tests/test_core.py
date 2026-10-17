"""Tests for core module."""

from __future__ import annotations

import math

import pytest
from hypothesis import given
from hypothesis import strategies as st

import overreact as rx
from overreact import _constants as constants
from tests.strategies import COMPOUNDS, equilibria, reactions, schemes_texts


@given(reactions())
def test_parse_counts_coefficients(reaction) -> None:
    """Ensure parsed coefficients add up over repeated compounds on each side."""
    reactants, products, transition, text = reaction
    scheme = rx.Scheme.parse(text)
    assert scheme.reactants == tuple(
        (reactants.get(compound, 0),) for compound in scheme.compounds
    )
    assert scheme.products == tuple(
        (products.get(compound, 0),) for compound in scheme.compounds
    )
    assert scheme.transition_states == (
        (None,) if transition is None else (scheme.compounds.index(transition),)
    )


@given(schemes_texts())
def test_str_is_inverse_of_parse(text) -> None:
    """Ensure unparsing and parsing again gives the same scheme."""
    scheme = rx.Scheme.parse(text)
    assert rx.Scheme.parse(str(scheme)) == scheme


def test_parse_rejects_transition_states_in_equilibria() -> None:
    """Ensure equilibria cannot go through transition states."""
    with pytest.raises(ValueError, match="equilibri"):
        rx.Scheme.parse("A <=> A‡ <=> B")


@pytest.mark.parametrize(
    "text",
    ["A -> B\nA <=> B", "B -> A\nA <=> B", "A <=> B\nA -> B", "A <=> B\nB -> A"],
)
def test_equilibria_absorb_plain_reactions(text) -> None:
    """Ensure an equilibrium absorbs plain reactions between the same sides."""
    assert rx.Scheme.parse(text) == rx.Scheme.parse("A <=> B")


@pytest.mark.parametrize(
    ("changes", "match"),
    [
        ({"reactants": ((1,),)}, "shape"),
        ({"reactants": ((-1,), (0,))}, "negative"),
        ({"reactants": ((0.5,), (0,))}, "integer"),
        ({"transition_states": (2,)}, "transition state"),
        ({"transition_states": (0,)}, "transition state"),
        ({"equilibria": ((0, 1),)}, "equilibri"),
        ({"equilibria": ((0, 0),)}, "equilibri"),
    ],
)
def test_scheme_rejects_invalid_fields(changes, match) -> None:
    """Ensure Scheme cannot hold inconsistent fields."""
    fields = {
        "compounds": ("A", "B"),
        "reactants": ((1,), (0,)),
        "products": ((0,), (1,)),
        "transition_states": (None,),
        "equilibria": (),
    }
    with pytest.raises(ValueError, match=match):
        rx.Scheme(**(fields | changes))


@given(reactions())
def test_molecularity_counts_reactant_molecules(reaction) -> None:
    """Ensure molecularity is the sum of reactant coefficients."""
    reactants, _, _, text = reaction
    assert rx.Scheme.parse(text).molecularity.tolist() == [sum(reactants.values())]


def _total(side, values):
    """Sum values over the molecules of a side."""
    return sum(n * values[c] for c, n in side.items())


def _symmetry(side):
    """Sum ln n! over the compounds of a side."""
    return sum(math.lgamma(n + 1) for n in side.values())


def _values(scheme):
    """Draw one value per compound of a scheme."""
    return st.fixed_dictionaries({c: st.floats(-1e3, 1e3) for c in scheme.compounds})


@given(reactions(), st.data())
def test_reaction_deltas_are_products_minus_reactants(reaction, data) -> None:
    """Ensure reaction deltas are product values minus reactant values."""
    reactants, products, _, text = reaction
    scheme = rx.Scheme.parse(text)
    values = data.draw(_values(scheme))
    assert scheme.reaction_deltas(
        [values[c] for c in scheme.compounds]
    ) == pytest.approx(
        [_total(products, values) - _total(reactants, values)],
    )


@given(reactions(), st.data())
def test_activation_deltas_reach_transition_states(reaction, data) -> None:
    """Ensure activation deltas end at the transition state, if any."""
    reactants, products, transition, text = reaction
    scheme = rx.Scheme.parse(text)
    values = data.draw(_values(scheme))
    final = _total(products, values) if transition is None else values[transition]
    assert scheme.activation_deltas(
        [values[c] for c in scheme.compounds]
    ) == pytest.approx(
        [final - _total(reactants, values)],
    )


@given(equilibria(), st.data())
def test_reverse_half_equilibria_have_no_activation(text, data) -> None:
    """Ensure only the forward half of an equilibrium has an activation delta."""
    scheme = rx.Scheme.parse(text)
    values = [data.draw(_values(scheme))[c] for c in scheme.compounds]
    forward, reverse = scheme.activation_deltas(values)
    assert (forward, reverse) == pytest.approx((scheme.reaction_deltas(values)[0], 0.0))


@given(reactions())
def test_reaction_entropies_count_indistinguishable_molecules(reaction) -> None:
    """Ensure each compound contributes R ln n! for its n molecules on each side."""
    reactants, products, _, text = reaction
    assert rx.Scheme.parse(text).reaction_entropies == pytest.approx(
        [constants.R * (_symmetry(products) - _symmetry(reactants))],
    )


@given(reactions())
def test_activation_entropies_end_at_transition_states(reaction) -> None:
    """Ensure activation entropies count a single transition state molecule."""
    reactants, products, transition, text = reaction
    final = _symmetry(products) if transition is None else 0.0
    assert rx.Scheme.parse(text).activation_entropies == pytest.approx(
        [constants.R * (final - _symmetry(reactants))],
    )


@given(
    reactions(),
    st.dictionaries(
        st.sampled_from(COMPOUNDS),
        st.floats(0.1, 10.0) | st.integers(1, 10**10),
    ),
)
def test_fix_removes_compounds_and_scales_rates(reaction, concentrations) -> None:
    """Ensure fixing drops compounds and scales rates by their concentrations."""
    reactants, products, transition, text = reaction
    original = rx.Scheme.parse(text)
    scheme, factors = original.fix(concentrations)
    compounds = tuple(c for c in original.compounds if c not in concentrations)
    assert scheme == rx.Scheme(
        compounds=compounds,
        reactants=tuple((reactants.get(c, 0),) for c in compounds),
        products=tuple((products.get(c, 0),) for c in compounds),
        transition_states=(
            None if transition is None else compounds.index(transition),
        ),
        equilibria=(),
    )
    assert factors == pytest.approx(
        [math.prod(concentrations.get(c, 1.0) ** n for c, n in reactants.items())],
    )


def test_fix_rejects_transition_states() -> None:
    """Ensure transition states cannot have fixed concentrations."""
    with pytest.raises(ValueError, match="transition state"):
        rx.Scheme.parse("A -> A‡ -> B").fix({"A‡": 1.0})
