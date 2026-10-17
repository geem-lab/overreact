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
            spellings = [f"{part} ", f"{part}"] + ([""] if part == 1 else [])
            terms.append(f"{draw(st.sampled_from(spellings))}{compound}")
            coefficient -= part
    return " + ".join(draw(st.permutations(terms)))


@st.composite
def reactions(draw) -> tuple[dict[str, int], dict[str, int], str | None, str]:
    """Draw a non-equilibrium reaction as reactants, products, transition state and text."""
    reactants, products = draw(reactions_sides)
    transition = draw(st.sampled_from([None, "T‡", "T#"]))
    arrow = draw(st.sampled_from(["->", "<-"]))
    texts = [draw(side_texts(reactants)), draw(side_texts(products))]
    if arrow == "<-":
        texts.reverse()
    middle = f" {arrow} " if transition is None else f" {arrow} {transition} {arrow} "
    return reactants, products, transition, middle.join(texts)


@st.composite
def equilibria(draw) -> str:
    """Draw an equilibrium between two different sides as text."""
    reactants, products = draw(reactions_sides.filter(lambda pair: pair[0] != pair[1]))
    return f"{draw(side_texts(reactants))} <=> {draw(side_texts(products))}"


@st.composite
def schemes_texts(draw) -> str:
    """Draw a model description with reactions, equilibria and transition states."""
    lines = [
        draw(reactions())[-1] if draw(st.booleans()) else draw(equilibria())
        for _ in range(draw(st.integers(min_value=1, max_value=4)))
    ]
    return "\n".join(lines)
