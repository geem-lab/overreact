"""Hypothesis strategies shared by tests."""

from __future__ import annotations

from hypothesis import strategies as st

COMPOUNDS = ("A", "B", "C", "D")

sides = st.dictionaries(
    st.sampled_from(COMPOUNDS),
    st.integers(min_value=1, max_value=3),
    max_size=3,
)
reactions_sides = st.tuples(sides, sides).filter(any)


@st.composite
def side_texts(draw, side: dict[str, int]) -> str:
    """Render a side in any accepted spelling, e.g. `2 B`, `2B`, `1 B + B`."""
    terms = []
    for compound, coefficient in side.items():
        while coefficient > 0:
            part = draw(st.integers(min_value=1, max_value=coefficient))
            prefix = draw(st.sampled_from(["", f"{part} ", f"{part}"]))
            if part > 1 and not prefix:
                prefix = f"{part} "
            terms.append(f"{prefix}{compound}")
            coefficient -= part
    terms = draw(st.permutations(terms))
    return " + ".join(terms)


@st.composite
def reactions(draw) -> tuple[dict[str, int], dict[str, int], str | None, str]:
    """Draw a non-equilibrium reaction as reactants, products, transition state and text."""
    reactants, products = draw(reactions_sides)
    transition = draw(st.sampled_from([None, "T‡", "T#"]))
    lhs, rhs = draw(side_texts(reactants)), draw(side_texts(products))
    middle = " -> " if transition is None else f" -> {transition} -> "
    text = f"{lhs}{middle}{rhs}"
    if draw(st.booleans()):
        middle = " <- " if transition is None else f" <- {transition} <- "
        text = f"{rhs}{middle}{lhs}"
    return reactants, products, transition, text


@st.composite
def schemes_texts(draw) -> str:
    """Draw a model description with reactions, equilibria and transition states."""
    lines = []
    for _ in range(draw(st.integers(min_value=1, max_value=4))):
        if draw(st.booleans()):
            *_, text = draw(reactions())
        else:
            reactants, products = draw(reactions_sides)
            lhs, rhs = draw(side_texts(reactants)), draw(side_texts(products))
            text = f"{lhs} <=> {rhs}"
        lines.append(text)
    return "\n".join(lines)
