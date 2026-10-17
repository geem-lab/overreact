"""Module dedicated to parsing and modeling of chemical reaction networks."""

from __future__ import annotations

__all__ = ["Scheme"]


import collections
import itertools
import re
from dataclasses import dataclass
from functools import cached_property
from typing import TYPE_CHECKING

import numpy as np
from scipy.special import gammaln

import overreact as rx
from overreact import _constants as constants

if TYPE_CHECKING:
    from collections.abc import Iterator, Sequence


@dataclass(frozen=True)
class Scheme:
    """
    A descriptor of a chemical reaction network.

    Mostly likely, this comes from `Scheme.parse` or a parsed input file (see
    `overreact.io.parse_model`).

    Coefficient matrices have one row per compound and one column per reaction.
    Half-equilibria come in consecutive pairs of mutually reverse reactions.
    """

    compounds: tuple[str, ...]
    """Compound names."""
    reactants: tuple[tuple[int, ...], ...]
    """Stoichiometric coefficients of reactants."""
    products: tuple[tuple[int, ...], ...]
    """Stoichiometric coefficients of products."""
    transition_states: tuple[int | None, ...]
    """Index of the transition state compound of each reaction, if any."""
    is_half_equilibrium: tuple[bool, ...]
    """An indicator of whether a reaction is half-equilibrium."""

    def __post_init__(self) -> None:
        """Normalize fields to tuples and validate them."""
        shape = (len(self.compounds), len(self.is_half_equilibrium))
        object.__setattr__(self, "compounds", tuple(self.compounds))
        for name in ("reactants", "products"):
            matrix = np.asarray(getattr(self, name), dtype=int)
            if matrix.size == 0:
                matrix = matrix.reshape(shape)
            if matrix.shape != shape:
                msg = f"{name} has shape {matrix.shape}, expected {shape}"
                raise ValueError(msg)
            if (matrix < 0).any():
                msg = f"{name} has negative coefficients"
                raise ValueError(msg)
            object.__setattr__(self, name, rx._misc.totuple(matrix))
        object.__setattr__(
            self,
            "transition_states",
            tuple(None if ts is None else int(ts) for ts in self.transition_states),
        )
        object.__setattr__(
            self,
            "is_half_equilibrium",
            tuple(bool(x) for x in self.is_half_equilibrium),
        )

        if len(self.transition_states) != shape[1]:
            msg = f"expected {shape[1]} transition states, got {len(self.transition_states)}"
            raise ValueError(msg)
        for ts in self.transition_states:
            if ts is not None and not (
                0 <= ts < shape[0] and is_transition_state(self.compounds[ts])
            ):
                msg = f"invalid transition state index {ts}"
                raise ValueError(msg)

        reactants, products = self._reactants, self._products
        for j in np.flatnonzero(~(reactants.any(axis=0) | products.any(axis=0))):
            msg = f"reaction {j} has neither reactants nor products"
            raise ValueError(msg)
        for i, j in self.equilibria():
            if not (
                j < shape[1]
                and self.is_half_equilibrium[j]
                and self.transition_states[i] is None
                and self.transition_states[j] is None
                and np.array_equal(reactants[:, i], products[:, j])
                and np.array_equal(products[:, i], reactants[:, j])
            ):
                msg = f"half-equilibrium '{self.reactions[i]}' lacks a reverse half-equilibrium without transition states"
                raise ValueError(msg)

    @classmethod
    def parse(cls, text: str | Sequence[str]) -> Scheme:
        """
        Parse a kinetic model.

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

        Raises
        ------
        ValueError
            If a half-equilibrium lacks its reverse or involves a transition state.

        Notes
        -----
        The model description should comply with the mini-language for systems of
        reactions. A semi-formal definition of the grammar in
        [Backus-Naur form](https://en.wikipedia.org/wiki/Backus%E2%80%93Naur_form)
        is given below:

                 equation ::= equation_side arrow equation_side
            equation_side ::= [coefficient compound ['+' coefficient compound]*]
              coefficient ::= [integers] (defaults to 1)
                 compound ::= mix of printable characters
                    arrow ::= '->' | '<=>' | '<-'

        Blank lines and comments (starting with `//`) are ignored. Repeated
        reactions are ignored, and coefficients of a compound repeated on one side
        add up. An empty side denotes a source or a sink. Furthermore, reactions
        can be chained one after another and, if a single compound (with either a
        `‡` or a `#` at the end) appears alone on one side of a reaction, it's
        considered a transition state. Transition states have zero lifetime during
        the simulation.

        Examples
        --------
        What follows is a rather long tour over the parsing process and its
        output in general. You can skip it if you are not interested in the
        details.

        >>> scheme = Scheme.parse("A -> B  // a direct reaction")

        The reaction above is a direct one (observe that comments are ignored). The
        returned object stores coefficients of reactants and products, with one row
        per compound and one column per reaction:

        >>> scheme
        Scheme(compounds=('A', 'B'),
               reactants=((1,), (0,)),
               products=((0,), (1,)),
               transition_states=(None,),
               is_half_equilibrium=(False,))

        The text of each reaction and the matrix $A$ of net coefficients are
        derived from those:

        >>> scheme.reactions
        ('A -> B',)
        >>> scheme.A
        array([[-1],
               [ 1]])

        The same reaction can be specified in reverse order:

        >>> Scheme.parse("B <- A  // reverse reaction of the above")
        Scheme(compounds=('A', 'B'),
               reactants=((1,), (0,)),
               products=((0,), (1,)),
               transition_states=(None,),
               is_half_equilibrium=(False,))

        Equilibria produce twice as many direct reactions, while activation
        deltas (e.g., of energies) are defined for only one of each pair:

        >>> scheme = Scheme.parse("A <=> B  // an equilibrium")
        >>> scheme.reactions
        ('A -> B', 'B -> A')
        >>> list(scheme.equilibria())
        [(0, 1)]
        >>> scheme.activation_deltas([0.0, 10.0])
        array([10.,  0.])

        Adding twice the same reaction results in a single reaction being added.
        This of course also works with equilibria (extra whitespaces are ignored):

        >>> Scheme.parse(\'\'\'
        ...     A <=> B  -> A
        ...     A  -> B <=> A
        ...     A  -> B <-  A
        ...     B <-  A  -> B
        ... \'\'\') == scheme
        True

        Coefficients of a compound add up, and spectators are kept apart from
        reactants and products, so the following is second order in B:

        >>> scheme = Scheme.parse("B + B -> B + C")
        >>> scheme.reactions
        ('2 B -> B + C',)
        >>> scheme.A
        array([[-1],
               [ 1]])

        Transition states are specified with a special symbol at the end (either
        `‡` or `#`). They are shown among compounds, but the matrix $A$ ensures
        they'll never have a non-zero rate of formation/consumption. On the other
        hand, activation deltas go through them:

        >>> scheme = Scheme.parse("A -> A‡ -> B")
        >>> scheme
        Scheme(compounds=('A', 'A‡', 'B'),
               reactants=((1,), (0,), (0,)),
               products=((0,), (0,), (1,)),
               transition_states=(1,),
               is_half_equilibrium=(False,))
        >>> scheme.reactions
        ('A -> B',)
        >>> scheme.A
        array([[-1],
               [ 0],
               [ 1]])
        >>> scheme.activation_deltas([0.0, 25.0, -10.0])
        array([25.])

        This gives the same result as above:

        >>> Scheme.parse("A -> A‡ -> B <- A‡ <- A") == scheme
        True

        It is possible to define a reaction whose product is the same as the
        reactant. This is found in isomerization processes (e.g., ammonia
        inversion or the methyl rotation in ethane):

        >>> scheme = Scheme.parse("S -> E‡ -> S")
        >>> scheme.reactions
        ('S -> S',)
        >>> scheme.A
        array([[0],
               [0]])
        >>> scheme.activation_deltas([0.0, 3.0])
        array([3.])

        As such, a column full of zeros in the $A$ matrix corresponds to a reaction
        with zero net change. As can be seen, overreact allows for very general
        models. An interesting feature is that a single transition state can link
        many different compounds (whether it is useful is a matter of debate):

        >>> scheme = Scheme.parse(\'\'\'
        ...     B  -> B‡  -> C  // chained reactions and transition states
        ...     B‡ -> D         // this is a bifurcation
        ...     B  -> B'‡ -> E  // this is a classical competitive reaction
        ...     A  -> B‡
        ... \'\'\')
        >>> scheme.compounds
        ('B', 'B‡', 'C', 'D', "B'‡", 'E', 'A')
        >>> scheme.reactions
        ('B -> C', 'B -> D', 'B -> E', 'A -> C', 'A -> D')
        >>> scheme.transition_states
        (1, 1, 4, 1, 1)

        The following is a borderline case but both reactions should be considered
        different since they define different processes:

        >>> scheme = Scheme.parse(\'\'\'
        ...     A -> A‡ -> B
        ...     A -> B
        ... \'\'\')
        >>> scheme.reactions
        ('A -> B', 'A -> B')
        >>> scheme.transition_states
        (1, None)

        The following is correct behavior. In fact, the reactions are badly
        defined: if more than one transition state are chained, the following
        happens, which is correct since it's the most physically plausible model
        that can be extracted. It can be seen as a feature that the product B is
        ignored and not the reactant A, since the user would easily see the mistake
        in graphs of concentration over time (the alternative would be no
        reaction happening at all, which is rather cryptic to debug).

        >>> scheme = Scheme.parse("A -> A‡ -> A'‡ -> B")
        >>> scheme.compounds
        ('A', 'A‡', "A'‡", 'B')
        >>> scheme.reactions
        ("A -> A'‡",)
        >>> scheme.transition_states
        (1,)

        In any case, it's not clear how a reaction barrier be defined in such a
        case. If you have a use case, don't hesitate to
        [open an issue](https://github.com/geem-lab/overreact/issues/), we'll be
        happy to hear from you.
        """
        compounds: dict[str, int] = {}
        reactions: dict[
            tuple[
                tuple[tuple[int, str], ...],
                tuple[tuple[int, str], ...],
                bool,
                tuple[tuple[int, str], ...] | None,
            ],
            None,
        ] = {}

        def _add_reaction(reactants, products, is_half_equilibrium, transition):
            """Local helper function with side-effects."""
            reactions.setdefault((reactants, products, is_half_equilibrium, transition))

        def _canonical(side):
            """Add up coefficients of repeated compounds, in order of compounds."""
            coefficients = collections.Counter()
            for coefficient, compound in side:
                coefficients[compound] += coefficient
            return tuple(
                (coefficients[compound], compound)
                for compound in sorted(coefficients, key=compounds.__getitem__)
            )

        after_transitions: dict[tuple[tuple[int, str], ...], list[tuple[int, str]]] = {}
        before_transitions: dict[
            tuple[tuple[int, str], ...], list[tuple[int, str]]
        ] = {}

        for reactants, products, is_half_equilibrium in _parse_reactions(text):
            for _, compound in itertools.chain(reactants, products):
                if compound not in compounds:
                    # found new compound
                    compounds[compound] = len(compounds)
            reactants, products = _canonical(reactants), _canonical(products)

            if (reactants, products, False, None) in reactions or (
                reactants,
                products,
                True,
                None,
            ) in reactions:
                continue

            # it's assumed that if a transition shows up,
            #   1. it's the only compound in its side of the reaction, and
            #   2. its coefficient equals one
            if reactants and is_transition_state(reactants[-1][-1]):
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
            elif products and is_transition_state(products[-1][-1]):
                for after_products in after_transitions.get(products, []):
                    _add_reaction(
                        reactants, after_products, is_half_equilibrium, products
                    )

                if products in before_transitions:
                    before_transitions[products].append(reactants)
                else:
                    before_transitions[products] = [reactants]
                continue

            _add_reaction(reactants, products, is_half_equilibrium, None)

        shape = (len(compounds), len(reactions))
        reactants_matrix, products_matrix = np.zeros(shape, int), np.zeros(shape, int)
        for j, (reactants, products, _, _) in enumerate(reactions):
            for coefficient, compound in reactants:
                reactants_matrix[compounds[compound], j] = coefficient
            for coefficient, compound in products:
                products_matrix[compounds[compound], j] = coefficient
        return cls(
            compounds=tuple(compounds),
            reactants=reactants_matrix,
            products=products_matrix,
            transition_states=tuple(
                None if transition is None else compounds[transition[-1][-1]]
                for _, _, _, transition in reactions
            ),
            is_half_equilibrium=tuple(reaction[2] for reaction in reactions),
        )

    @cached_property
    def reactions(self) -> tuple[str, ...]:
        """Reactions as text, without transition states."""
        return tuple(
            _unparse_reactions(
                (
                    self._side(self._reactants[:, i]),
                    self._side(self._products[:, i]),
                )
                for i in range(len(self.is_half_equilibrium))
            ),
        )

    def equilibria(self) -> Iterator[tuple[int, int]]:
        """Yield indices of the forward and reverse halves of each equilibrium.

        Examples
        --------
        >>> list(Scheme.parse("A <=> B -> C <=> D").equilibria())
        [(0, 1), (3, 4)]
        """
        flags = enumerate(self.is_half_equilibrium)
        for i, is_half_equilibrium in flags:
            if is_half_equilibrium:
                next(flags, None)
                yield i, i + 1

    @cached_property
    def A(self) -> np.ndarray:  # noqa: N802
        """Net stoichiometric coefficients, from reactants to products."""
        return _read_only(self._products - self._reactants)

    @cached_property
    def molecularity(self) -> np.ndarray:
        """Number of reactant molecules of each reaction."""
        return _read_only(self._reactants.sum(axis=0))

    def reaction_deltas(self, values) -> np.ndarray:
        """Change of per-compound values from reactants to products.

        Examples
        --------
        >>> Scheme.parse("2 A -> B").reaction_deltas([1.0, 5.0])
        array([3.])
        """
        return self.A.T @ np.asarray(values)

    def activation_deltas(self, values) -> np.ndarray:
        """Change of per-compound values from reactants to the activated state.

        The activated state is the transition state if any, and the products
        otherwise. Reverse halves of equilibria have no activation.

        Examples
        --------
        >>> Scheme.parse("2 A -> A‡ -> B").activation_deltas([1.0, 5.0, 0.0])
        array([3.])
        """
        return self._activation.T @ np.asarray(values)

    @cached_property
    def reaction_entropies(self) -> np.ndarray:
        r"""Reaction symmetry entropies.

        This is the reaction translational entropy, a result of the
        indistinguishability of molecules of each compound, i.e., a difference in
        entropy of $R \ln 2!$ between $\ce{A + B -> C}$ and $\ce{2A -> C}$.

        Examples
        --------
        >>> Scheme.parse("A + B <=> C").reaction_entropies
        array([0., 0.])
        >>> Scheme.parse("2A <=> C").reaction_entropies
        array([-5.763,  5.763])

        Molecules on both sides of a reaction count on each side:

        >>> Scheme.parse("2B -> B + C").reaction_entropies
        array([-5.763])
        """
        return _read_only(
            _symmetry_entropies(self._products) - _symmetry_entropies(self._reactants),
        )

    @cached_property
    def activation_entropies(self) -> np.ndarray:
        """Activation symmetry entropies, analogous to `reaction_entropies`.

        Examples
        --------
        >>> Scheme.parse("2A -> E# -> C").activation_entropies
        array([-5.763])
        """
        return _read_only(
            _symmetry_entropies(self._activated) - _symmetry_entropies(self._reactants),
        )

    def fix(self, concentrations) -> tuple[Scheme, np.ndarray]:
        """Remove compounds held at fixed concentrations.

        Returns the remaining scheme and the factors to multiply reaction rate
        constants by, so that rates are unchanged.

        Examples
        --------
        Equilibria under a specific pH can be modeled:

        >>> scheme, factors = Scheme.parse("AH <=> A- + H+").fix({"H+": 1e-7})
        >>> scheme.reactions
        ('AH -> A-', 'A- -> AH')
        >>> factors
        array([1.e+00, 1.e-07])

        Fixing all reactants of a reaction turns it into a source:

        >>> scheme, factors = Scheme.parse("2 H2O -> P").fix({"H2O": 55.6})
        >>> scheme.reactions
        ('-> P',)
        >>> factors
        array([3091.36])
        """
        keep = [i for i, c in enumerate(self.compounds) if c not in concentrations]
        fixed = [i for i, c in enumerate(self.compounds) if c in concentrations]
        values = np.array([concentrations[self.compounds[i]] for i in fixed])
        return (
            Scheme(
                compounds=tuple(self.compounds[i] for i in keep),
                reactants=self._reactants[keep],
                products=self._products[keep],
                transition_states=tuple(
                    None if ts is None else keep.index(ts)
                    for ts in self.transition_states
                ),
                is_half_equilibrium=self.is_half_equilibrium,
            ),
            np.prod(values[:, np.newaxis] ** self._reactants[fixed], axis=0),
        )

    def __str__(self) -> str:
        r"""Return the model description.

        Examples
        --------
        >>> str(Scheme(compounds=('A', 'A‡', 'B', 'C'),
        ...            reactants=((1, 0, 0),
        ...                       (0, 0, 0),
        ...                       (0, 0, 2),
        ...                       (0, 2, 0)),
        ...            products=((0, 0, 0),
        ...                      (0, 0, 0),
        ...                      (1, 2, 0),
        ...                      (0, 0, 2)),
        ...            transition_states=(1, None, None),
        ...            is_half_equilibrium=(False, True, True)))
        'A -> A‡ -> B\n2 C <=> 2 B'

        Parsing it gives back the same scheme, up to formatting:

        >>> print(Scheme.parse('''
        ...     B  -> B‡  -> C  // chained reactions and transition states
        ...     B‡ -> D         // this is a bifurcation
        ...     B  -> B'‡ -> E  // this is a classical competitive reaction
        ...     A  -> B‡
        ... '''))
        B -> B‡ -> C
        B -> B‡ -> D
        B -> B'‡ -> E
        A -> B‡ -> C
        A -> B‡ -> D
        >>> str(Scheme.parse("S -> E‡ -> S"))
        'S -> E‡ -> S'
        >>> str(Scheme.parse("-> A <=> B + B ->"))
        '-> A\nA <=> 2 B\n2 B ->'
        """
        return "\n".join(self._lines())

    def _lines(self) -> Iterator[str]:
        """Yield the model description line by line."""
        reactions = enumerate(
            zip(self.reactions, self.transition_states, strict=True),
        )
        for i, (reaction, transition_state) in reactions:
            if transition_state is not None:
                yield reaction.replace(
                    "->",
                    f"-> {self.compounds[transition_state]} ->",
                )
            elif self.is_half_equilibrium[i]:
                next(reactions, None)  # skip the reverse half-equilibrium
                yield reaction.replace("->", "<=>")
            else:
                yield reaction

    @cached_property
    def _reactants(self) -> np.ndarray:
        return _read_only(self._array(self.reactants))

    @cached_property
    def _products(self) -> np.ndarray:
        return _read_only(self._array(self.products))

    @cached_property
    def _activated(self) -> np.ndarray:
        """Coefficients of the activated state of each reaction."""
        activated = self._products.copy()
        for _, j in self.equilibria():
            activated[:, j] = self._reactants[:, j]
        for i, ts in enumerate(self.transition_states):
            if ts is not None:
                activated[:, i] = 0
                activated[ts, i] = 1
        return _read_only(activated)

    @cached_property
    def _activation(self) -> np.ndarray:
        """Coefficients from reactants to the activated state."""
        return _read_only(self._activated - self._reactants)

    def _array(self, matrix) -> np.ndarray:
        return np.reshape(
            np.array(matrix, dtype=int),
            (len(self.compounds), len(self.is_half_equilibrium)),
        )

    def _side(self, coefficients) -> Iterator[tuple[int, str]]:
        return (
            (int(coefficient), compound)
            for coefficient, compound in zip(coefficients, self.compounds, strict=True)
            if coefficient
        )


def _read_only(array: np.ndarray) -> np.ndarray:
    """Return an array after making it read-only."""
    array.setflags(write=False)
    return array


def _symmetry_entropies(coefficients) -> np.ndarray:
    """Return R ln n! summed over compounds, for each column of coefficients."""
    return constants.R * gammaln(coefficients + 1).sum(axis=0)


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


def _check_scheme(scheme_or_text: Scheme | str) -> Scheme:
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
           reactants=((1,), (0,)),
           products=((0,), (1,)),
           transition_states=(None,),
           is_half_equilibrium=(False,))
    >>> _check_scheme(_check_scheme("A -> B")) == _check_scheme("A -> B")
    True
    """
    if isinstance(scheme_or_text, Scheme):
        return scheme_or_text
    return Scheme.parse(scheme_or_text)


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
        line = line.split("//")[0].strip()
        if not line:
            continue

        pieces = re.split(r"\s*(->|<=>|<-)\s*", line)
        for reactants, arrow, products in zip(
            pieces[:-2:2],
            pieces[1:-1:2],
            pieces[2::2],
            strict=False,
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
        yield f"{_unparse_side(reaction[0])} -> {_unparse_side(reaction[1])}".strip()


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
    if not side.strip():
        return
    for token in re.split(r"\s+\+\s+", side):
        token = re.match(
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
