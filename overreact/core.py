#!/usr/bin/env python3

"""Module dedicated to parsing and modeling of chemical reaction networks."""

import collections as _collections
import itertools as _itertools
import re as _re

import numpy as _np

Scheme = _collections.namedtuple(
    "Scheme", "compounds reactions is_half_equilibrium A B"
)


def parse(text):
    """Parse a kinetic model.

    Parameters
    ----------
    text : str
        Model description

    Returns
    -------
    scheme : Scheme

    Notes
    -----
    The model description should comply with the mini-language for systems of
    reactions::

             equation ::= equation_side arrow equation_side
        equation_side ::= coefficient compound ['+' coefficient compound]*
          coefficient ::= [integers] (defaults to 1)
             compound ::= mix of printable characters
                arrow ::= '->' | '<=>' | '<-'

    Blank lines and comments (starting with '#') are ignored. Doubled reactions
    are ignored. Furthermore, reactions can be chained one after another and,
    if a single compound (with an "*" at the end) appears alone on one side of
    a reaction, it's considered a transition state, whose lifetime is zero.

    Examples
    --------
    >>> scheme = parse("A -> B  # a direct reaction")

    The reaction above is a direct one (observe that comments are ignored). The
    returned object has the following attributes:

    >>> scheme.compounds
    ['A', 'B']
    >>> scheme.reactions
    ['A -> B']
    >>> scheme.is_half_equilibrium
    array([False])
    >>> scheme.A
    array([[-1.], [ 1.]])
    >>> scheme.B
    array([[-1.], [ 1.]])

    The same reaction can be specified in reverse order:

    >>> parse("B <- A  # reverse reaction of the above")
    Scheme(compounds=['A', 'B'],
           reactions=['A -> B'],
           is_half_equilibrium=array([False]),
           A=array([[-1.], [ 1.]]),
           B=array([[-1.], [ 1.]]))

    Equilibria produce twice as many direct reactions, while the B matrix
    defines an energy relationship for only one of each pair:

    >>> parse("A <=> B  # an equilibrium")
    Scheme(compounds=['A', 'B'],
           reactions=['A -> B', 'B -> A'],
           is_half_equilibrium=array([ True, True]),
           A=array([[-1.,  1.],
                    [ 1., -1.]]),
           B=array([[-1.,  0.],
                    [ 1.,  0.]]))

    Adding twice the same reaction results in a single reaction being added.
    This of course also works with equilibria (extra whitespaces are ignored):

    >>> parse('''
    ...     A <=> B  -> A
    ...     A  -> B <=> A
    ...     A  -> B <-  A
    ...     B <-  A  -> B
    ... ''')
    Scheme(compounds=['A', 'B'],
           reactions=['A -> B', 'B -> A'],
           is_half_equilibrium=array([ True, True]),
           A=array([[-1.,  1.],
                    [ 1., -1.]]),
           B=array([[-1.,  0.],
                    [ 1.,  0.]]))

    Transition states are specified with an asterisk at the end. They are shown
    in the list of compounds, but the matrix A ensures they'll never have a
    non-zero rate of formation/consumption. On the other hand, they might be
    needed in the B matrix:

    >>> parse("A -> A* -> B")
    Scheme(compounds=['A', 'A*', 'B'],
           reactions=['A -> B'],
           is_half_equilibrium=array([False]),
           A=array([[-1.], [ 0.], [ 1.]]),
           B=array([[-1.], [ 1.], [ 0.]]))

    This gives the same result as above:

    >>> parse("A -> A* -> B <- A* <- A")
    Scheme(compounds=['A', 'A*', 'B'],
           reactions=['A -> B'],
           is_half_equilibrium=array([False]),
           A=array([[-1.], [ 0.], [ 1.]]),
           B=array([[-1.], [ 1.], [ 0.]]))

    overreact allows extremely general models. An interesting feature is that a
    single transition state can link many different compounds:

    >>> parse('''
    ...     B  -> B*  -> C  # chained reactions and transition states
    ...     B* -> D          # this is a bifurcation
    ...     B  -> B** -> E  # this is a classical competitive reaction
    ...     A  -> B*
    ... ''')
    Scheme(compounds=['B', 'B*', 'C', 'D', 'B**', 'E', 'A'],
           reactions=['B -> C', 'B -> D', 'B -> E', 'A -> C', 'A -> D'],
           is_half_equilibrium=array([False, False, False, False, False]),
           A=array([[-1., -1., -1.,  0.,  0.],
                    [ 0.,  0.,  0.,  0.,  0.],
                    [ 1.,  0.,  0.,  1.,  0.],
                    [ 0.,  1.,  0.,  0.,  1.],
                    [ 0.,  0.,  0.,  0.,  0.],
                    [ 0.,  0.,  1.,  0.,  0.],
                    [ 0.,  0.,  0., -1., -1.]]),
           B=array([[-1., -1., -1.,  0.,  0.],
                    [ 1.,  1.,  0.,  1.,  1.],
                    [ 0.,  0.,  0.,  0.,  0.],
                    [ 0.,  0.,  0.,  0.,  0.],
                    [ 0.,  0.,  1.,  0.,  0.],
                    [ 0.,  0.,  0.,  0.,  0.],
                    [ 0.,  0.,  0., -1., -1.]]))

    The following is a borderline case but both reactions should be considered
    different since they define different processes:

    >>> parse('''
    ...     A -> A* -> B
    ...     A -> B
    ... ''')
    Scheme(compounds=['A', 'A*', 'B'],
           reactions=['A -> B', 'A -> B'],
           is_half_equilibrium=array([False, False]),
           A=array([[-1., -1.],
                    [ 0.,  0.],
                    [ 1.,  1.]]),
           B=array([[-1., -1.],
                    [ 1.,  0.],
                    [ 0.,  1.]]))

    The following is correct output for bad input. If more than one transition
    state are chained, the following happens, which is correct since it's the
    most physically plausible model that can be extracted from the input. I
    think it's a feature that the product B is ignored and not the reactant A
    since the user will easily see the mistake in graphs of concentration over
    time (the alternative would be no reaction happening at all, which is
    cryptical). Furthermore, it's not clear how a reaction barrier be defined
    in such a weird case:

    >>> parse("A -> A* -> A** -> B")
    Scheme(compounds=['A', 'A*', 'A**', 'B'],
           reactions=['A -> A**'],
           is_half_equilibrium=array([False]),
           A=array([[-1.], [ 0.], [ 1.], [ 0.]]),
           B=array([[-1.], [ 1.], [ 0.], [ 0.]]))

    """
    compounds = _collections.OrderedDict()
    reactions = _collections.OrderedDict()
    A = list()  # coefficients between reactants and products
    B = list()  # coefficients between reactants and transition states

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

        A_vector = _np.zeros(len(compounds))
        for coefficient, reactant in reactants:
            A_vector[compounds[reactant]] = -coefficient
        B_vector = A_vector

        if transition is not None:
            B_vector = A_vector.copy()

            # it's assumed that
            #   1. there's a singe transition compound, and
            #   2. its coefficient equals one
            B_vector[compounds[transition[-1][-1]]] = 1

        for coefficient, product in products:
            A_vector[compounds[product]] = coefficient

        if (
            is_half_equilibrium
            and (products, reactants, is_half_equilibrium, transition) in reactions
        ):
            B_vector = _np.zeros(len(compounds))

        A.append(A_vector)
        B.append(B_vector)

    after_transitions = _collections.OrderedDict()
    before_transitions = _collections.OrderedDict()

    for reactants, products, is_half_equilibrium in _parse_reactions(text):
        if (reactants, products, False, None) in reactions or (
            reactants,
            products,
            True,
            None,
        ) in reactions:
            continue

        for _, compound in _itertools.chain(reactants, products):
            if compound not in compounds:
                # found new compound
                compounds[compound] = len(compounds)

        # TODO(schneiderfelipe): what if a transition state is used in an
        # equilibrium?

        # it's assumed that if a transition shows up,
        #   1. it's the only compound in its side of the reaction, and
        #   2. its coefficient equals one
        if reactants[-1][-1].endswith("*"):
            for before_reactants in before_transitions.get(reactants, []):
                _add_reaction(
                    before_reactants, products, is_half_equilibrium, reactants
                )

            if reactants in after_transitions:
                after_transitions[reactants].append(products)
            else:
                after_transitions[reactants] = [products]
            continue
        elif products[-1][-1].endswith("*"):
            for after_products in after_transitions.get(products, []):
                _add_reaction(reactants, after_products, is_half_equilibrium, products)

            if products in before_transitions:
                before_transitions[products].append(reactants)
            else:
                before_transitions[products] = [reactants]
            continue

        _add_reaction(reactants, products, is_half_equilibrium, None)

    return Scheme(
        compounds=list(compounds),
        reactions=list(_unparse_reactions(reactions)),
        is_half_equilibrium=_np.array([reaction[2] for reaction in reactions]),
        A=_np.block(
            [[vector, _np.zeros(len(compounds) - len(vector))] for vector in A]
        ).T,
        B=_np.block(
            [[vector, _np.zeros(len(compounds) - len(vector))] for vector in B]
        ).T,
    )


def _parse_reactions(text):
    r"""Parse reactions.

    Parameters
    ----------
    text : str
        Model description

    Yields
    ------
    reactants, products : list of tuple
    is_half_equilibrium : bool

    Examples
    --------
    >>> r = "E + S <=> ES -> ES* -> E + P"
    >>> for reactants, products, is_half_equilibrium in _parse_reactions(r):
    ...     print(reactants, products, is_half_equilibrium)
    ((1, 'E'), (1, 'S')) ((1, 'ES'),) True
    ((1, 'ES'),) ((1, 'E'), (1, 'S')) True
    ((1, 'ES'),) ((1, 'ES*'),) False
    ((1, 'ES*'),) ((1, 'E'), (1, 'P')) False

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
    for line in text.split("\n"):
        line = line.split("#")[0].strip()
        if not line:
            continue

        pieces = _re.split(r"\s*(->|<=>|<-)\s*", line)
        for reactants, arrow, products in zip(
            pieces[:-2:2], pieces[1:-1:2], pieces[2::2]
        ):
            if arrow == "<-":
                reactants, products, arrow = products, reactants, "->"
            reactants = tuple(_parse_side(reactants))
            products = tuple(_parse_side(products))

            if arrow == "<=>":
                yield reactants, products, True
                yield products, reactants, True
            else:
                yield reactants, products, False


def _unparse_reactions(reactions):
    """Return string representations of reactions.

    Parameters
    ----------
    reactions : list of tuple

    Yields
    ------
    text : str

    Examples
    --------
    >>> for text in _unparse_reactions([(((1, 'E'), (1, 'S')), ((1, 'ES'),),
    ...         True),
    ...     (((1, 'ES'),), ((1, 'E'), (1, 'S')), True),
    ...     (((1, 'ES'),), ((1, 'ES*'),), False),
    ...     (((1, 'ES*'),), ((1, 'E'), (1, 'P')), False)]):
    ...     print(text)
    E + S -> ES
    ES -> E + S
    ES -> ES*
    ES* -> E + P

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
    for token in _re.split(r"\s*\+\s*", side):
        token = _re.match(
            r"\s*(?P<coefficient>\d+)?\s*(?P<compound>[^\s]+)\s*", token
        ).groupdict(1)
        # TODO(schneiderfelipe): should coefficient be float?
        yield int(token["coefficient"]), token["compound"]


def _unparse_side(unside):
    """Return string representation of a lefti/right hand side of a reaction.

    Parameters
    ----------
    unside : list of tuple

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
