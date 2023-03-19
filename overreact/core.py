#!/usr/bin/env python3  # noqa: EXE001

"""Module dedicated to parsing and modeling of chemical reaction networks."""


__all__ = ["Scheme", "parse_reactions"]


import itertools
import re
from typing import NamedTuple, Sequence, Union

import numpy as np

import overreact as rx


class Scheme(NamedTuple):
    """
    A descriptor of a chemical reaction network.

    Mostly likely, this comes from a parsed input file.
    See `overreact.io.parse_model`.
    """

    compounds: Sequence[str]
    """A descriptor of compounds."""
    reactions: Sequence[str]
    """A descriptor of reactions."""
    is_half_equilibrium: Sequence[bool]
    """An indicator of whether a reaction is half-equilibrium."""
    A: np.ndarray
    """A matrix of stoichiometric coefficients between reactants and products."""
    B: np.ndarray
    """A matrix of stoichiometric coefficients between reactants and transition states."""  # noqa: E501


_abbr_environment = {
    "dcm": "dichloromethane",
    "dmf": "n,n-dimethylformamide",
    "dmso": "dimethylsulfoxide",
    "ccl4": "carbon tetrachloride",
    "g": "gas",
    "mecn": "acetonitrile",
    "meno2": "nitromethane",
    "phno2": "nitrobenzene",
    "s": "solid",
    "thf": "tetrahydrofuran",
    "w": "water",
}


def _check_scheme(scheme_or_text: Union[Scheme, str]) -> Scheme:
    """Interface transparently between strings and schemes.

    Parameters
    ----------
    scheme_or_text : Scheme or str

    Returns
    -------
    Scheme

    Examples
    --------
    >>> _check_scheme("A -> B")
    Scheme(compounds=('A', 'B'),
           reactions=('A -> B',),
           is_half_equilibrium=(False,),
           A=((-1.,), (1.,)),
           B=((-1.,), (1.,)))
    >>> _check_scheme(_check_scheme("A -> B"))
    Scheme(compounds=('A', 'B'),
           reactions=('A -> B',),
           is_half_equilibrium=(False,),
           A=((-1.,), (1.,)),
           B=((-1.,), (1.,)))
    """
    if isinstance(scheme_or_text, Scheme):
        return scheme_or_text
    return parse_reactions(scheme_or_text)


def get_transition_states(A, B, is_half_equilibrium):  # noqa: N803
    """Return the indices of transition states for each reaction.

    Parameters
    ----------
    A, B : array-like
    is_half_equilibrium : sequence

    Returns
    -------
    sequence

    Examples
    --------
    >>> scheme = parse_reactions("A -> B")
    >>> print(scheme)
    Scheme(compounds=('A', 'B'),
           reactions=('A -> B',),
           is_half_equilibrium=(False,),
           A=((-1.,), (1.,)),
           B=((-1.,), (1.,)))
    >>> get_transition_states(scheme.A, scheme.B, scheme.is_half_equilibrium)
    (None,)

    >>> scheme = parse_reactions("S -> E‡ -> S")
    >>> print(scheme)
    Scheme(compounds=('S', 'E‡'),
           reactions=('S -> S',),
           is_half_equilibrium=(False,),
           A=((0.,), (0.,)),
           B=((-1.,), (1.,)))
    >>> get_transition_states(scheme.A, scheme.B, scheme.is_half_equilibrium)
    (1,)

    >>> scheme = parse_reactions("E + S <=> ES -> ES‡ -> E + P")
    >>> print(scheme)
    Scheme(compounds=('E', 'S', 'ES', 'ES‡', 'P'),
           reactions=('E + S -> ES', 'ES -> E + S', 'ES -> E + P'),
           is_half_equilibrium=(True,  True, False),
           A=((-1.,  1.,  1.),
              (-1.,  1.,  0.),
              (1., -1., -1.),
              (0.,  0.,  0.),
              (0.,  0.,  1.)),
           B=((-1.,  0.,  0.),
              (-1.,  0.,  0.),
              (1.,  0., -1.),
              (0.,  0.,  1.),
              (0.,  0.,  0.)))
    >>> get_transition_states(scheme.A, scheme.B, scheme.is_half_equilibrium)
    (None, None, 3)

    """
    tau = np.asarray(B) - np.asarray(A) > 0  # transition state matrix
    return tuple(
        x if not is_half_equilibrium[i] and tau[:, i].any() else None
        for i, x in enumerate(np.argmax(tau, axis=0))
    )


# TODO(schneiderfelipe): some of the more esoteric doctests should become
# real tests.
def unparse_reactions(scheme: Scheme) -> str:
    """Unparse a kinetic model.

    Parameters
    ----------
    scheme : Scheme
        A descriptor of the reaction scheme.
        Mostly likely, this comes from a parsed input file.
        See `overreact.io.parse_model`.

    Returns
    -------
    text : str

    Notes
    -----
    This function assumes complimentary half equilibria are located one after
    the other in ``scheme.reactions``, which is to be expected from
    `parse_reactions`.

    Examples
    --------
    >>> unparse_reactions(Scheme(compounds=('A', 'B'), reactions=('A -> B',),
    ...                   is_half_equilibrium=(False,),
    ...                   A=((-1.,),
    ...                      ( 1.,)),
    ...                   B=((-1.,),
    ...                      ( 1.,))))
    'A -> B'
    >>> unparse_reactions(Scheme(compounds=('A', 'B'),
    ...                   reactions=('A -> B', 'B -> A'),
    ...                   is_half_equilibrium=(True, True),
    ...                   A=((-1.,  1.),
    ...                      ( 1., -1.)),
    ...                   B=((-1.,  0.),
    ...                      ( 1.,  0.))))
    'A <=> B'
    >>> unparse_reactions(Scheme(compounds=('A', 'A‡', 'B'),
    ...                   reactions=('A -> B',),
    ...                   is_half_equilibrium=(False,),
    ...                   A=((-1.,),
    ...                      ( 0.,),
    ...                      ( 1.,)),
    ...                   B=((-1.,),
    ...                      ( 1.,),
    ...                      ( 0.,))))
    'A -> A‡ -> B'
    >>> print(unparse_reactions(Scheme(compounds=('B', 'B‡', 'C', 'D', "B'‡",
    ...                                           'E', 'A'),
    ...                         reactions=('B -> C', 'B -> D', 'B -> E',
    ...                                    'A -> C', 'A -> D'),
    ...                         is_half_equilibrium=(False, False, False,
    ...                                              False, False),
    ...                         A=((-1., -1., -1.,  0.,  0.),
    ...                            ( 0.,  0.,  0.,  0.,  0.),
    ...                            ( 1.,  0.,  0.,  1.,  0.),
    ...                            ( 0.,  1.,  0.,  0.,  1.),
    ...                            ( 0.,  0.,  0.,  0.,  0.),
    ...                            ( 0.,  0.,  1.,  0.,  0.),
    ...                            ( 0.,  0.,  0., -1., -1.)),
    ...                         B=((-1., -1., -1.,  0.,  0.),
    ...                            ( 1.,  1.,  0.,  1.,  1.),
    ...                            ( 0.,  0.,  0.,  0.,  0.),
    ...                            ( 0.,  0.,  0.,  0.,  0.),
    ...                            ( 0.,  0.,  1.,  0.,  0.),
    ...                            ( 0.,  0.,  0.,  0.,  0.),
    ...                            ( 0.,  0.,  0., -1., -1.)))))
    B -> B‡ -> C
    B -> B‡ -> D
    B -> B'‡ -> E
    A -> B‡ -> C
    A -> B‡ -> D
    >>> print(unparse_reactions(Scheme(compounds=('A', 'A‡', 'B'),
    ...                         reactions=('A -> B', 'A -> B'),
    ...                         is_half_equilibrium=(False, False),
    ...                         A=((-1., -1.),
    ...                            ( 0.,  0.),
    ...                            ( 1.,  1.)),
    ...                         B=((-1., -1.),
    ...                            ( 1.,  0.),
    ...                            ( 0.,  1.)))))
    A -> A‡ -> B
    A -> B
    >>> unparse_reactions(Scheme(compounds=('A', 'A‡', "A'‡", 'B'),
    ...                   reactions=("A -> A'‡",),
    ...                   is_half_equilibrium=(False,),
    ...                   A=((-1.,),
    ...                      ( 0.,),
    ...                      ( 1.,),
    ...                      ( 0.,)),
    ...                   B=((-1.,),
    ...                      ( 1.,),
    ...                      ( 0.,),
    ...                      ( 0.,))))
    "A -> A‡ -> A'‡"
    >>> unparse_reactions(Scheme(compounds=('S', 'E‡'),
    ...                   reactions=('S -> S',),
    ...                   is_half_equilibrium=(False,),
    ...                   A=((0.0,),
    ...                      (0.0,)),
    ...                   B=((-1.0,),
    ...                      (1.0,))))
    'S -> E‡ -> S'
    """
    scheme = _check_scheme(scheme)
    transition_states = get_transition_states(
        scheme.A,
        scheme.B,
        scheme.is_half_equilibrium,
    )
    lines = []
    i = 0
    while i < len(scheme.reactions):
        if transition_states[i] is not None:
            lines.append(
                scheme.reactions[i].replace(
                    "->",
                    f"-> {scheme.compounds[transition_states[i]]} ->",
                ),
            )
        elif scheme.is_half_equilibrium[i]:
            lines.append(scheme.reactions[i].replace("->", "<=>"))
            i += 1  # avoid backward reaction, which comes next
        else:
            lines.append(scheme.reactions[i])
        i += 1
    return "\n".join(lines)


def _get_environment(name):
    """Retrieve a compound's environment by its name.

    Parameters
    ----------
    name : str

    Returns
    -------
    str

    Examples
    --------
    >>> _get_environment("pyrrole")
    'gas'

    By default, compounds are assumed to be in gas phase, but you can give a
    special tag to specify the solvent:

    >>> _get_environment("pyrrole(water)")
    'water'
    >>> _get_environment("pyrrole(dichloromethane)")
    'dichloromethane'

    Some abbreviations are accepted, such as "w" for water:

    >>> _get_environment("pyrrole(w)")
    'water'
    >>> _get_environment("pyrrole(dcm)")
    'dichloromethane'

    You can indicate a phase as usual, although solids are currently not
    supported in overreact:

    >>> _get_environment("pyrrole(solid)")
    'solid'
    >>> _get_environment("pyrrole(s)")
    'solid'
    >>> _get_environment("pyrrole(gas)")
    'gas'
    >>> _get_environment("pyrrole(g)")
    'gas'

    For the case of liquids, the returned environment is the name of the
    compound (abbreviations are applied as usual):

    >>> _get_environment("water(l)")
    'water'
    >>> _get_environment("water(liquid)")
    'water'
    >>> _get_environment("dcm(l)")
    'dichloromethane'

    This function also works for names specifying transition states:

    >>> _get_environment("A‡(w)")
    'water'
    >>> _get_environment("TS#(dmf)")
    'n,n-dimethylformamide'
    """
    token = re.match(
        r"\s*(?P<compound>[^\s\(\)]+)\s*(?P<environment>\([^\s\(\)]+\))?\s*",
        name,
    ).groupdict("(gas)")
    name = token["compound"]

    environment = token["environment"][1:-1].lower()
    if environment in {"l", "liquid"}:
        environment = name

    if environment in _abbr_environment:
        environment = _abbr_environment[environment]
    return environment


def is_transition_state(name):
    """Check whether a name specifies a transition state.

    Parameters
    ----------
    name : str

    Returns
    -------
    bool

    Examples
    --------
    >>> is_transition_state("A#")
    True
    >>> is_transition_state("A‡")
    True
    >>> is_transition_state("pyrrole#")
    True
    >>> is_transition_state("pyrrole‡")
    True
    >>> is_transition_state("A")
    False
    >>> is_transition_state("A~")
    False
    >>> is_transition_state("pyrrole")
    False
    >>> is_transition_state("pyrrole~")
    False

    This function also works for names that specify environment:

    >>> is_transition_state("A#(w)")
    True
    >>> is_transition_state("A‡(w)")
    True
    >>> is_transition_state("TS#(w)")
    True
    >>> is_transition_state("TS‡(w)")
    True
    >>> is_transition_state("A(w)")
    False
    >>> is_transition_state("A~(w)")
    False
    >>> is_transition_state("TS(w)")
    False
    >>> is_transition_state("TS~(w)")
    False
    """
    return any(marker in name for marker in {"‡", "#"})


def parse_reactions(text: Union[str, Sequence[str]]) -> Scheme:  # noqa: C901
    """
    Parse a kinetic model as a chemical reaction scheme.

    This is an essential part of the parsing process.
    See `overreact.io.parse_model` other details.

    Parameters
    ----------
    text : str or sequence of str
        Model description or sequence of lines of it.

    Returns
    -------
    scheme : Scheme
        A descriptor of the reaction scheme.

    Notes
    -----
    The model description should comply with the mini-language for systems of
    reactions. A semi-formal definition of the grammar in
    [Backus-Naur form](https://en.wikipedia.org/wiki/Backus%E2%80%93Naur_form)
    is given below:

             equation ::= equation_side arrow equation_side
        equation_side ::= coefficient compound ['+' coefficient compound]*
          coefficient ::= [integers] (defaults to 1)
             compound ::= mix of printable characters
                arrow ::= '->' | '<=>' | '<-'

    Blank lines and comments (starting with `//`) are ignored. Repeated
    reactions are ignored. Furthermore, reactions can be chained one after
    another and, if a single compound (with either a `‡` or a `#` at the end)
    appears alone on one side of a reaction, it's considered a transition
    state. Transition states have zero lifetime during the simulation.

    Examples
    --------
    What follows is a rather long tour over the parsing process and its
    output in general. You can skip it if you are not interested in the
    details.

    >>> scheme = parse_reactions("A -> B  // a direct reaction")

    The reaction above is a direct one (observe that comments are ignored). The
    returned object has the following attributes:

    >>> scheme.compounds
    ('A', 'B')
    >>> scheme.reactions
    ('A -> B',)
    >>> scheme.is_half_equilibrium
    (False,)
    >>> scheme.A
    ((-1.,), (1.,))
    >>> scheme.B
    ((-1.,), (1.,))

    The same reaction can be specified in reverse order:

    >>> parse_reactions("B <- A  // reverse reaction of the above")
    Scheme(compounds=('A', 'B'),
           reactions=('A -> B',),
           is_half_equilibrium=(False,),
           A=((-1.,), (1.,)),
           B=((-1.,), (1.,)))

    Equilibria produce twice as many direct reactions, while the $B$ matrix
    defines an energy relationship for only one of each pair:

    >>> parse_reactions("A <=> B  // an equilibrium")
    Scheme(compounds=('A', 'B'),
           reactions=('A -> B', 'B -> A'),
           is_half_equilibrium=(True, True),
           A=((-1.,  1.),
              (1., -1.)),
           B=((-1.,  0.),
              (1.,  0.)))

    Adding twice the same reaction results in a single reaction being added.
    This of course also works with equilibria (extra whitespaces are ignored):

    >>> parse_reactions('''
    ...     A <=> B  -> A
    ...     A  -> B <=> A
    ...     A  -> B <-  A
    ...     B <-  A  -> B
    ... ''')
    Scheme(compounds=('A', 'B'),
           reactions=('A -> B', 'B -> A'),
           is_half_equilibrium=(True, True),
           A=((-1.,  1.),
              (1., -1.)),
           B=((-1.,  0.),
              (1.,  0.)))

    Transition states are specified with a special symbol at the end (either
    `‡` or `#`). They are shown among compounds, but the matrix $A$ ensures
    they'll never have a non-zero rate of formation/consumption. On the other
    hand, they are needed in the $B$ matrix:

    >>> parse_reactions("A -> A‡ -> B")
    Scheme(compounds=('A', 'A‡', 'B'),
           reactions=('A -> B',),
           is_half_equilibrium=(False,),
           A=((-1.,), (0.,), (1.,)),
           B=((-1.,), (1.,), (0.,)))

    This gives the same result as above:

    >>> parse_reactions("A -> A‡ -> B <- A‡ <- A")
    Scheme(compounds=('A', 'A‡', 'B'),
           reactions=('A -> B',),
           is_half_equilibrium=(False,),
           A=((-1.,), (0.,), (1.,)),
           B=((-1.,), (1.,), (0.,)))

    It is possible to define a reaction whose product is the same as the
    reactant. This is found in isomerization processes (e.g., ammonia
    inversion or the methyl rotation in ethane):

    >>> parse_reactions("S -> E‡ -> S")
    Scheme(compounds=('S', 'E‡'),
           reactions=('S -> S',),
           is_half_equilibrium=(False,),
           A=((0.,), (0.,)),
           B=((-1.,), (1.,)))

    As such, a column full of zeros in the $A$ matrix corresponds to a reaction
    with zero net change. As can be seen, overreact allows for very general
    models. An interesting feature is that a single transition state can link
    many different compounds (whether it is useful is a matter of debate):

    >>> parse_reactions('''
    ...     B  -> B‡  -> C  // chained reactions and transition states
    ...     B‡ -> D         // this is a bifurcation
    ...     B  -> B'‡ -> E  // this is a classical competitive reaction
    ...     A  -> B‡
    ... ''')
    Scheme(compounds=('B', 'B‡', 'C', 'D', "B'‡", 'E', 'A'),
           reactions=('B -> C', 'B -> D', 'B -> E', 'A -> C', 'A -> D'),
           is_half_equilibrium=(False, False, False, False, False),
           A=((-1., -1., -1.,  0.,  0.),
              (0.,  0.,  0.,  0.,  0.),
              (1.,  0.,  0.,  1.,  0.),
              (0.,  1.,  0.,  0.,  1.),
              (0.,  0.,  0.,  0.,  0.),
              (0.,  0.,  1.,  0.,  0.),
              (0.,  0.,  0., -1., -1.)),
           B=((-1., -1., -1.,  0.,  0.),
              (1.,  1.,  0.,  1.,  1.),
              (0.,  0.,  0.,  0.,  0.),
              (0.,  0.,  0.,  0.,  0.),
              (0.,  0.,  1.,  0.,  0.),
              (0.,  0.,  0.,  0.,  0.),
              (0.,  0.,  0., -1., -1.)))

    The following is a borderline case but both reactions should be considered
    different since they define different processes:

    >>> parse_reactions('''
    ...     A -> A‡ -> B
    ...     A -> B
    ... ''')
    Scheme(compounds=('A', 'A‡', 'B'),
           reactions=('A -> B', 'A -> B'),
           is_half_equilibrium=(False, False),
           A=((-1., -1.),
              (0.,  0.),
              (1.,  1.)),
           B=((-1., -1.),
              (1.,  0.),
              (0.,  1.)))

    The following is correct behavior. In fact, the reactions are badly
    defined: if more than one transition state are chained, the following
    happens, which is correct since it's the most physically plausible model
    that can be extracted. It can be seen as a feature that the product B is
    ignored and not the reactant A, since the user would easily see the mistake
    in graphs of concentration over time (the alternative would be no
    reaction happening at all, which is rather cryptic to debug).

    >>> parse_reactions("A -> A‡ -> A'‡ -> B")
    Scheme(compounds=('A', 'A‡', "A'‡", 'B'),
           reactions=("A -> A'‡",),
           is_half_equilibrium=(False,),
           A=((-1.,), (0.,), (1.,), (0.,)),
           B=((-1.,), (1.,), (0.,), (0.,)))

    In any case, it's not clear how a reaction barrier be defined in such a
    case. If you have a use case, don't hesitate to
    [open an issue](https://github.com/geem-lab/overreact/issues/), we'll be
    happy to hear from you.
    """
    compounds: dict[str, int] = {}
    reactions: dict[
        tuple[str, str, bool, str],
        tuple[tuple[tuple[int, str], ...], bool],
    ] = {}
    A = []  # coefficients between reactants and products  # noqa: N806
    B = []  # coefficients between reactants and transition states  # noqa: N806

    def _add_reaction(reactants, products, is_half_equilibrium, transition):
        """Local helper function with side-effects."""
        # TODO(schneiderfelipe): what if reaction is defined, then redefined
        # as equilibrium, or vice-versa?
        if (
            transition is not None
            and (reactants, products, is_half_equilibrium, transition) in reactions
        ):
            return

        # found new reaction
        reactions[(reactants, products, is_half_equilibrium, transition)] = None

        A_vector = np.zeros(len(compounds))  # noqa: N806
        for coefficient, reactant in reactants:
            A_vector[compounds[reactant]] = -coefficient
        B_vector = A_vector  # noqa: N806

        if transition is not None:
            B_vector = A_vector.copy()  # noqa: N806

            # it's assumed that
            #   1. there's a singe transition compound, and
            #   2. its coefficient equals one
            B_vector[compounds[transition[-1][-1]]] = 1

        for coefficient, product in products:
            A_vector[compounds[product]] += coefficient

        if (
            is_half_equilibrium
            and (products, reactants, is_half_equilibrium, transition) in reactions
        ):
            B_vector = np.zeros(len(compounds))  # noqa: N806

        A.append(A_vector)
        B.append(B_vector)

    after_transitions: dict[tuple[tuple[int, str], ...], list[tuple[int, str]]] = {}
    before_transitions: dict[tuple[tuple[int, str], ...], list[tuple[int, str]]] = {}

    for reactants, products, is_half_equilibrium in _parse_reactions(text):
        if (reactants, products, False, None) in reactions or (
            reactants,
            products,
            True,
            None,
        ) in reactions:
            continue

        for _, compound in itertools.chain(reactants, products):
            if compound not in compounds:
                # found new compound
                compounds[compound] = len(compounds)

        # TODO(schneiderfelipe): what if a transition state is used in an
        # equilibrium?

        # it's assumed that if a transition shows up,
        #   1. it's the only compound in its side of the reaction, and
        #   2. its coefficient equals one
        if is_transition_state(reactants[-1][-1]):
            for before_reactants in before_transitions.get(reactants, []):
                _add_reaction(
                    before_reactants,
                    products,
                    is_half_equilibrium,
                    reactants,
                )

            if reactants in after_transitions:
                after_transitions[reactants].append(products)
            else:
                after_transitions[reactants] = [products]
            continue
        elif is_transition_state(products[-1][-1]):  # noqa: RET507
            for after_products in after_transitions.get(products, []):
                _add_reaction(reactants, after_products, is_half_equilibrium, products)

            if products in before_transitions:
                before_transitions[products].append(reactants)
            else:
                before_transitions[products] = [reactants]
            continue

        _add_reaction(reactants, products, is_half_equilibrium, None)

    return Scheme(
        compounds=tuple(compounds),
        reactions=tuple(_unparse_reactions(reactions)),
        is_half_equilibrium=rx._misc.totuple(  # noqa: SLF001
            [reaction[2] for reaction in reactions],
        ),
        A=rx._misc.totuple(  # noqa: SLF001
            np.block(
                [[vector, np.zeros(len(compounds) - len(vector))] for vector in A],
            ).T,
        ),
        B=rx._misc.totuple(  # noqa: SLF001
            np.block(
                [[vector, np.zeros(len(compounds) - len(vector))] for vector in B],
            ).T,
        ),
    )


def _parse_reactions(text):
    r"""Parse reactions.

    Parameters
    ----------
    text : str or sequence of str
        Model description or sequence of lines of it.

    Yields
    ------
    reactants, products : sequence of tuple
    is_half_equilibrium : bool

    Examples
    --------
    >>> r = "E + S <=> ES -> ES‡ -> E + P"
    >>> for reactants, products, is_half_equilibrium in _parse_reactions(r):
    ...     print(reactants, products, is_half_equilibrium)
    ((1, 'E'), (1, 'S')) ((1, 'ES'),) True
    ((1, 'ES'),) ((1, 'E'), (1, 'S')) True
    ((1, 'ES'),) ((1, 'ES‡'),) False
    ((1, 'ES‡'),) ((1, 'E'), (1, 'P')) False

    `_parse_reactions` and `_unparse_reactions` are, in some sense, inverses of
    each other:

    >>> print('\n'.join(_unparse_reactions(_parse_reactions('''
    ...     1 A -> 2 B <- C <=> 40 D <- E
    ...     A -> 2 B <=> C
    ... '''))))
    A -> 2 B
    C -> 2 B
    C -> 40 D
    40 D -> C
    E -> 40 D
    A -> 2 B
    2 B -> C
    C -> 2 B

    """
    try:
        lines = text.split("\n")
    except AttributeError:
        lines = text
    for line in lines:
        line = line.split("//")[0].strip()  # noqa: PLW2901
        if not line:
            continue

        pieces = re.split(r"\s*(->|<=>|<-)\s*", line)
        for reactants, arrow, products in zip(
            pieces[:-2:2],
            pieces[1:-1:2],
            pieces[2::2],
        ):
            if arrow == "<-":
                reactants, products, arrow = products, reactants, "->"  # noqa: PLW2901
            reactants = tuple(_parse_side(reactants))  # noqa: PLW2901
            products = tuple(_parse_side(products))  # noqa: PLW2901

            if arrow == "<=>":
                yield reactants, products, True
                yield products, reactants, True
            else:
                yield reactants, products, False


def _unparse_reactions(reactions):
    """Return string representations of reactions.

    Parameters
    ----------
    reactions : sequence of tuple

    Yields
    ------
    text : str
        Line of model description.

    Examples
    --------
    >>> for text in _unparse_reactions([(((1, 'E'), (1, 'S')), ((1, 'ES'),),
    ...         True),
    ...     (((1, 'ES'),), ((1, 'E'), (1, 'S')), True),
    ...     (((1, 'ES'),), ((1, 'ES‡'),), False),
    ...     (((1, 'ES‡'),), ((1, 'E'), (1, 'P')), False)]):
    ...     print(text)
    E + S -> ES
    ES -> E + S
    ES -> ES‡
    ES‡ -> E + P

    """
    for reaction in reactions:
        yield f"{_unparse_side(reaction[0])} -> {_unparse_side(reaction[1])}"


def _parse_side(side):
    """Parse a left or right hand side of a reaction.

    Parameters
    ----------
    side : str

    Yields
    ------
    coefficient : int
    compound : str

    Examples
    --------
    >>> for coefficient, compound in _parse_side("A + B + 2 C + 500 D"):
    ...     print(coefficient, compound)
      1 A
      1 B
      2 C
    500 D

    `_parse_side` and `_unparse_side` are, in some sense, inverses of each
    other:

    >>> _unparse_side(_parse_side("   2     *A*1*    +  40B1  +  chlorophyll"))
    '2 *A*1* + 40 B1 + chlorophyll'

    """
    for token in re.split(r"\s+\+\s+", side):
        token = re.match(  # noqa: PLW2901
            r"\s*(?P<coefficient>\d+)?\s*(?P<compound>[^\s]+)\s*",
            token,
        ).groupdict(1)
        yield int(token["coefficient"]), token["compound"]


def _unparse_side(unside):
    """Return string representation of a left/right hand side of a reaction.

    Parameters
    ----------
    unside : sequence of tuple

    Returns
    -------
    side : str

    Examples
    --------
    >>> _unparse_side([(1, 'A'),
    ...     (1, 'B'),
    ...     (2, 'C'),
    ...     (500, 'D')])
    'A + B + 2 C + 500 D'

    """
    return " + ".join(
        f"{compound}" if coefficient == 1 else f"{coefficient} {compound}"
        for coefficient, compound in unside
    )
