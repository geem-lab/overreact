"""Tests for core module."""

from __future__ import annotations

import pytest
from hypothesis import assume, given

import overreact as rx
from tests.strategies import reactions, schemes_texts


@given(reactions())
def test_parse_reactions_counts_coefficients(reaction) -> None:
    """Ensure parsed coefficients add up over repeated compounds on each side."""
    reactants, products, transition, text = reaction
    scheme = rx.parse_reactions(text)
    assert scheme.reactants == tuple(
        (reactants.get(compound, 0),) for compound in scheme.compounds
    )
    assert scheme.products == tuple(
        (products.get(compound, 0),) for compound in scheme.compounds
    )
    assert scheme.transition_states == (
        (None,) if transition is None else (scheme.compounds.index(transition),)
    )


def _by_name(scheme):
    """Describe a scheme independently of the order of its compounds."""

    def side(matrix, j):
        return {
            c: row[j] for c, row in zip(scheme.compounds, matrix, strict=True) if row[j]
        }

    return set(scheme.compounds), [
        (
            side(scheme.reactants, j),
            side(scheme.products, j),
            None if ts is None else scheme.compounds[ts],
            scheme.is_half_equilibrium[j],
        )
        for j, ts in enumerate(scheme.transition_states)
    ]


@given(schemes_texts())
def test_unparse_reactions_is_inverse_of_parse_reactions(text) -> None:
    """Ensure unparsing and parsing again gives the same scheme."""
    try:
        scheme = rx.parse_reactions(text)
    except ValueError:
        assume(False)
    assert _by_name(rx.parse_reactions(rx.unparse_reactions(scheme))) == _by_name(
        scheme
    )


@pytest.mark.parametrize(
    "text",
    [
        "A -> B\nA <=> B",  # half-equilibrium without its reverse
        "A <=> A‡ <=> B",  # transition state in an equilibrium
    ],
)
def test_parse_reactions_rejects_invalid_equilibria(text) -> None:
    """Ensure ill-defined equilibria raise."""
    with pytest.raises(ValueError, match="equilibri"):
        rx.parse_reactions(text)


@pytest.mark.parametrize(
    ("changes", "match"),
    [
        ({"reactants": ((1,),)}, "shape"),
        ({"reactants": ((-1,), (0,))}, "negative"),
        ({"reactants": ((0,), (0,)), "products": ((0,), (0,))}, "neither"),
        ({"transition_states": (2,)}, "transition state"),
        ({"transition_states": (0,)}, "transition state"),
        ({"is_half_equilibrium": (True,)}, "equilibri"),
    ],
)
def test_scheme_rejects_invalid_fields(changes, match) -> None:
    """Ensure Scheme cannot hold inconsistent fields."""
    fields = {
        "compounds": ("A", "B"),
        "reactants": ((1,), (0,)),
        "products": ((0,), (1,)),
        "transition_states": (None,),
        "is_half_equilibrium": (False,),
    }
    with pytest.raises(ValueError, match=match):
        rx.Scheme(**(fields | changes))
