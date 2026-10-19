"""Tests for core module."""

from __future__ import annotations

import math

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st

import overreact as rx
from overreact import _constants as constants
from tests.strategies import (
    COMPOUNDS,
    equilibria,
    reactions,
    reactions_sides,
    schemes_texts,
    side_texts,
)


@given(reactions())
def test_parse_counts_coefficients(reaction) -> None:
    """Ensure parsed coefficients add up over repeated compounds on each side."""
    reactants, products, transition, text = reaction
    scheme = rx.Scheme.parse(text)
    assert scheme.reactants.tolist() == [
        [reactants.get(compound, 0)] for compound in scheme.compounds
    ]
    assert scheme.products.tolist() == [
        [products.get(compound, 0)] for compound in scheme.compounds
    ]
    assert scheme.transition_states == (transition,)


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
    [
        "A -> A‡",
        "A‡ -> B",
        "A -> A‡ -> A'‡ -> B",
        "A -> T‡ -> B\nC + T‡ -> D",
        "A -> 2 T‡ -> B",
    ],
)
def test_parse_rejects_misplaced_transition_states(text) -> None:
    """Ensure every transition state stands alone between reactants and products."""
    with pytest.raises(ValueError, match="transition state"):
        rx.Scheme.parse(text)


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
        ({"transition_states": ("B",)}, "transition state"),
        ({"compounds": ("A", "B‡")}, "transition state"),
        ({"compounds": ("A", "A")}, "repeated"),
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


VALUES = st.fixed_dictionaries(
    {c: st.floats(-1e3, 1e3) for c in (*COMPOUNDS, "S", "T‡")},
)


def _quantities(scheme, values, kind):
    """Return deltas of values and symmetry entropies of a kind of transformation."""
    deltas = getattr(scheme, f"{kind}_deltas")([values[c] for c in scheme.names])
    return np.concatenate([deltas, getattr(scheme, f"{kind}_entropies")])


@given(equilibria(), VALUES)
def test_reverse_reactions_negate_reaction_quantities(text, values) -> None:
    """Ensure reversing a reaction negates its deltas and symmetry entropies."""
    forward_delta, reverse_delta, forward_entropy, reverse_entropy = _quantities(
        rx.Scheme.parse(text), values, "reaction"
    )
    assert [forward_delta, forward_entropy] == pytest.approx(
        [-reverse_delta, -reverse_entropy],
    )


@given(reactions_sides, VALUES, st.data())
def test_reaction_quantities_are_state_functions(sides, values, data) -> None:
    """Ensure going from reactants to products equals going through nothing."""
    lhs, rhs = (data.draw(side_texts(side)) for side in sides)
    direct, consumption, production = (
        _quantities(rx.Scheme.parse(text), values, "reaction")
        for text in (f"{lhs} -> {rhs}", f"{lhs} ->", f"-> {rhs}")
    )
    assert direct == pytest.approx(consumption + production)


@given(reactions_sides, reactions_sides, VALUES, st.data())
def test_activation_stops_at_transition_states(
    sides, other_sides, values, data
) -> None:
    """Ensure activation through a transition state ignores the products."""
    reactants = data.draw(side_texts(sides[0]))
    texts = [
        f"{reactants} -> T‡ -> {data.draw(side_texts(rhs))}"
        for rhs in (sides[1], other_sides[1])
    ]
    first, second = (
        _quantities(rx.Scheme.parse(text), values, "activation") for text in texts
    )
    assert first == pytest.approx(second)


@given(reactions_sides, VALUES, st.data())
def test_activation_without_transition_states_is_reaction(sides, values, data) -> None:
    """Ensure activation without a transition state goes to the products."""
    text = f"{data.draw(side_texts(sides[0]))} -> {data.draw(side_texts(sides[1]))}"
    scheme = rx.Scheme.parse(text)
    assert _quantities(scheme, values, "activation") == pytest.approx(
        _quantities(scheme, values, "reaction"),
    )


@given(
    schemes_texts(),
    st.lists(st.floats(100.0, 1000.0), min_size=1, max_size=3),
    st.sampled_from(["reaction", "activation"]),
    st.data(),
)
def test_free_energies_broadcast_over_leading_axes(
    text, temperatures, kind, data
) -> None:
    """Ensure free energies of a batch of conditions are those of each condition."""
    scheme = rx.Scheme.parse(text)
    method = getattr(scheme, f"{kind}_freeenergies")
    size = len(scheme.names)
    values = data.draw(
        st.lists(
            st.lists(st.floats(-1e3, 1e3), min_size=size, max_size=size),
            min_size=len(temperatures),
            max_size=len(temperatures),
        ),
    )
    assert method(values, temperature=np.array(temperatures)[:, np.newaxis]) == (
        pytest.approx(
            np.array(
                [
                    method(v, temperature=t)
                    for v, t in zip(values, temperatures, strict=True)
                ],
            ),
        )
    )


@given(
    st.integers(1, 4),
    st.integers(1, 4),
    st.floats(-1e3, 1e3),
    st.floats(100.0, 1000.0),
    st.sampled_from(["reaction", "activation"]),
)
def test_free_energies_count_indistinguishable_molecules(
    reactants, products, value, temperature, kind
) -> None:
    """Ensure n identical molecules weigh as n! distinct ones, on either side."""

    def freeenergies(lhs, rhs):
        scheme = rx.Scheme.parse(f"{lhs} -> T‡ -> {rhs}")
        method = getattr(scheme, f"{kind}_freeenergies")
        return method([value] * len(scheme.names), temperature=temperature)

    identical = freeenergies(f"{reactants} A", f"{products} B")
    distinct = freeenergies(
        " + ".join(f"A{i}" for i in range(reactants)),
        " + ".join(f"B{i}" for i in range(products)),
    )
    ratio = math.factorial(reactants)
    if kind == "reaction":
        ratio /= math.factorial(products)
    assert np.exp(
        (identical - distinct) / (constants.R * temperature)
    ) == pytest.approx(
        [ratio],
    )


@given(
    reactions(),
    st.dictionaries(
        st.sampled_from(COMPOUNDS),
        st.floats(0.1, 10.0) | st.integers(1, 10**10),
    ),
)
def test_fix_removes_compounds(reaction, concentrations) -> None:
    """Ensure fixing drops compounds and keeps reactions otherwise."""
    reactants, products, transition, text = reaction
    original = rx.Scheme.parse(text)
    concentrations = {
        c: v for c, v in concentrations.items() if c in original.compounds
    }
    scheme, _ = original.fix(concentrations)
    compounds = tuple(c for c in original.compounds if c not in concentrations)
    assert scheme == rx.Scheme(
        compounds=compounds,
        reactants=tuple((reactants.get(c, 0),) for c in compounds),
        products=tuple((products.get(c, 0),) for c in compounds),
        transition_states=(transition,),
        equilibria=(),
    )


@given(schemes_texts(), st.data())
def test_fix_broadcasts_over_leading_axes(text, data) -> None:
    """Ensure factors of a batch of concentrations are those of each one."""
    scheme = rx.Scheme.parse(text)
    names = data.draw(st.sets(st.sampled_from(scheme.compounds), min_size=1))
    batch = [{name: data.draw(st.floats(0.1, 10.0)) for name in names} for _ in "ab"]
    _, factors = scheme.fix({name: [c[name] for c in batch] for name in names})
    assert factors == pytest.approx(np.array([scheme.fix(c)[1] for c in batch]))


@given(schemes_texts(), st.data())
def test_concentrations_place_values_by_name(text, data) -> None:
    """Ensure concentrations go to their compounds, and others are zero."""
    scheme = rx.Scheme.parse(text)
    given_names = data.draw(st.sets(st.sampled_from(scheme.compounds)))
    values = {name: data.draw(st.floats(0.1, 10.0)) for name in given_names}
    assert dict(
        zip(scheme.compounds, scheme.concentrations(values), strict=True),
    ) == {name: values.get(name, 0.0) for name in scheme.compounds}


@pytest.mark.parametrize("method", ["fix", "concentrations"])
def test_concentrations_reject_transition_states(method) -> None:
    """Ensure concentrations are only given to compounds of the scheme."""
    with pytest.raises(ValueError, match="not a compound"):
        getattr(rx.Scheme.parse("A -> A‡ -> B"), method)({"A‡": 1.0})


def test_parse_ignores_zero_coefficients() -> None:
    """Ensure compounds with zero coefficients take no part in reactions."""
    assert rx.Scheme.parse("0 A -> B\n-> B") == rx.Scheme.parse("-> B")
