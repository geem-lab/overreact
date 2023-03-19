#!/usr/bin/env python3  # noqa: INP001, EXE001

"""Tests for core module."""

import numpy as np

import overreact as rx


def test_parse_works():  # noqa: PLR0915
    """Test parsing of reactions."""
    scheme = rx.parse_reactions("A -> B  // a direct reaction")
    assert scheme[0] == ("A", "B")
    assert scheme[1] == ("A -> B",)
    assert scheme[2] == (False,)
    assert np.all(scheme[3] == scheme[4])
    assert np.all(np.array([[-1], [1]]) == scheme[3])

    scheme = rx.parse_reactions("B <- A  // reverse reaction of the above")
    assert scheme[0] == ("A", "B")
    assert scheme[1] == ("A -> B",)
    assert scheme[2] == (False,)
    assert np.all(scheme[3] == scheme[4])
    assert np.all(np.array([[-1], [1]]) == scheme[3])

    scheme = rx.parse_reactions("A <=> B  // an equilibrium")
    assert scheme[0] == ("A", "B")
    assert scheme[1] == ("A -> B", "B -> A")
    assert np.all(np.array([True, True]) == scheme[2])
    assert np.all(np.array([[-1.0, 1.0], [1.0, -1.0]]) == scheme[3])
    assert np.all(np.array([[-1.0, 0.0], [1.0, 0.0]]) == scheme[4])

    scheme = rx.parse_reactions(
        """A <=> B  -> A  // a lot of
        A  -> B <=> A     // repeated
        A  -> B <-  A     // reactions
        B <-  A  -> B""",
    )
    assert scheme[0] == ("A", "B")
    assert scheme[1] == ("A -> B", "B -> A")
    assert np.all(np.array([True, True]) == scheme[2])
    assert np.all(np.array([[-1.0, 1.0], [1.0, -1.0]]) == scheme[3])
    assert np.all(np.array([[-1.0, 0.0], [1.0, 0.0]]) == scheme[4])

    scheme = rx.parse_reactions("A -> A‡ -> B  // a transition state")
    assert scheme[0] == ("A", "A‡", "B")
    assert scheme[1] == ("A -> B",)
    assert scheme[2] == (False,)
    assert np.all(np.array([[-1.0], [0.0], [1.0]]) == scheme[3])
    assert np.all(np.array([[-1.0], [1.0], [0.0]]) == scheme[4])

    scheme = rx.parse_reactions("A -> A‡ -> B <- A‡ <- A  // (should be) same as above")
    assert scheme[0] == ("A", "A‡", "B")
    assert scheme[1] == ("A -> B",)
    assert scheme[2] == (False,)
    assert np.all(np.array([[-1.0], [0.0], [1.0]]) == scheme[3])
    assert np.all(np.array([[-1.0], [1.0], [0.0]]) == scheme[4])

    scheme = rx.parse_reactions(
        """
        B  -> B‡  -> C  // chained reactions and transition states
        B‡ -> D         // this is a bifurcation
        B  -> B'‡ -> E  // this is a classical competitive reaction
        A  -> B‡
    """,
    )
    assert scheme[0] == ("B", "B‡", "C", "D", "B'‡", "E", "A")
    assert scheme[1] == ("B -> C", "B -> D", "B -> E", "A -> C", "A -> D")
    assert np.all(np.array([False, False, False, False, False]) == scheme[2])
    assert np.all(
        np.array(
            [
                [-1.0, -1.0, -1.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0, 1.0, 0.0],
                [0.0, 1.0, 0.0, 0.0, 1.0],
                [0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, -1.0, -1.0],
            ],
        )
        == scheme[3],
    )
    assert np.all(
        np.array(
            [
                [-1.0, -1.0, -1.0, 0.0, 0.0],
                [1.0, 1.0, 0.0, 1.0, 1.0],
                [0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, -1.0, -1.0],
            ],
        )
        == scheme[4],
    )

    scheme = rx.parse_reactions(
        """// when in doubt, reactions should be considered distinct
        A -> A‡ -> B  // this is a tricky example
        A -> B        // but it's better to be explicit""",
    )
    assert scheme[0] == ("A", "A‡", "B")
    assert scheme[1] == ("A -> B", "A -> B")
    assert np.all(np.array([False, False]) == scheme[2])
    assert np.all(np.array([[-1.0, -1.0], [0.0, 0.0], [1.0, 1.0]]) == scheme[3])
    assert np.all(np.array([[-1.0, -1.0], [1.0, 0.0], [0.0, 1.0]]) == scheme[4])

    scheme = rx.parse_reactions(
        """
        // the policy is to not chain transition states
        A -> A‡ -> A'‡ -> B  // this is weird""",
    )
    assert scheme[0] == ("A", "A‡", "A'‡", "B")
    assert scheme[1] == ("A -> A'‡",)
    assert scheme[2] == (False,)
    assert np.all(np.array([[-1.0], [0.0], [1.0], [0.0]]) == scheme[3])
    assert np.all(np.array([[-1.0], [1.0], [0.0], [0.0]]) == scheme[4])


def test_private_functions_work():
    """Ensure private functions work as expected."""
    assert list(rx.core._parse_side("A")) == [(1, "A")]  # noqa: SLF001
    assert list(rx.core._parse_side("A")) == list(
        rx.core._parse_side("1 A"),
    )
    assert list(rx.core._parse_side("A")) == list(
        rx.core._parse_side("1A"),
    )
    assert list(rx.core._parse_side("500 A")) == [(500, "A")]  # noqa: SLF001
    assert list(rx.core._parse_side("A + 2 B + 500 D")) == [  # noqa: SLF001
        (1, "A"),
        (2, "B"),
        (500, "D"),
    ]

    assert rx.core._unparse_side([(1, "A")]) == "A"  # noqa: SLF001
    assert rx.core._unparse_side([(500, "A")]) == "500 A"  # noqa: SLF001
    assert rx.core._unparse_side([(1, "A"), (2, "B"), (500, "D")]) == "A + 2 B + 500 D"

    assert (
        rx.core._unparse_side(  # noqa: SLF001
            rx.core._parse_side(
                " 2  *A*1*   +    40B1     +      chlorophyll",
            ),
        )
        == "2 *A*1* + 40 B1 + chlorophyll"
    )

    assert list(rx.core._parse_reactions("A -> B")) == [  # noqa: SLF001
        (((1, "A"),), ((1, "B"),), False),
    ]
    assert list(rx.core._parse_reactions("A <=> B")) == [  # noqa: SLF001
        (((1, "A"),), ((1, "B"),), True),
        (((1, "B"),), ((1, "A"),), True),
    ]
    assert list(rx.core._parse_reactions("2 A -> B\nA -> 20B")) == [  # noqa: SLF001
        (((2, "A"),), ((1, "B"),), False),
        (((1, "A"),), ((20, "B"),), False),
    ]
    assert list(
        rx.core._parse_reactions("E + S <=> ES -> ES‡ -> E + P"),
    ) == [
        (((1, "E"), (1, "S")), ((1, "ES"),), True),
        (((1, "ES"),), ((1, "E"), (1, "S")), True),
        (((1, "ES"),), ((1, "ES‡"),), False),
        (((1, "ES‡"),), ((1, "E"), (1, "P")), False),
    ]

    assert list(
        rx.core._unparse_reactions([(((1, "A"),), ((1, "B"),), True)]),
    ) == [
        "A -> B",
    ]
    assert list(
        rx.core._unparse_reactions(  # noqa: SLF001
            [
                (((2, "A"),), ((3, "B"),), True),
                (((1, "A"),), ((2, "C"),), True),
                (((50, "A"),), ((1, "D"),), True),
            ],
        ),
    ) == ["2 A -> 3 B", "A -> 2 C", "50 A -> D"]
    assert list(
        rx.core._unparse_reactions(  # noqa: SLF001
            [
                (((1, "E"), (1, "S")), ((1, "ES"),), True),
                (((1, "ES"),), ((1, "E"), (1, "S")), True),
                (((1, "ES"),), ((1, "ES‡"),), False),
                (((1, "ES‡"),), ((1, "E"), (1, "P")), False),
            ],
        ),
    ) == ["E + S -> ES", "ES -> E + S", "ES -> ES‡", "ES‡ -> E + P"]

    assert list(
        rx.core._unparse_reactions(  # noqa: SLF001
            rx.core._parse_reactions(
                "1 A -> 2 B <- C <=> 40 D <- E\nA -> 2 B <=> C",
            ),
        ),
    ) == [
        "A -> 2 B",
        "C -> 2 B",
        "C -> 40 D",
        "40 D -> C",
        "E -> 40 D",
        "A -> 2 B",
        "2 B -> C",
        "C -> 2 B",
    ]
