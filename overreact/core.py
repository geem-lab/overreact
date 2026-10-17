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
    """

    compounds: tuple[str, ...]
    """Compound names."""
    reactants: tuple[tuple[int, ...], ...]
    """Stoichiometric coefficients of reactants."""
    products: tuple[tuple[int, ...], ...]
    """Stoichiometric coefficients of products."""
    transition_states: tuple[int | None, ...]
    """Index of the transition state compound of each reaction, if any."""
    equilibria: tuple[tuple[int, int], ...]
    """Indices of the forward and reverse halves of each equilibrium."""

    def __post_init__(self) -> None:
        """Normalize fields to tuples and validate them."""
        object.__setattr__(self, "compounds", tuple(self.compounds))
        object.__setattr__(
            self,
            "transition_states",
            tuple(None if ts is None else int(ts) for ts in self.transition_states),
        )
        object.__setattr__(
            self,
            "equilibria",
            tuple((int(i), int(j)) for i, j in self.equilibria),
        )
        shape = (len(self.compounds), len(self.transition_states))
        for name in ("reactants", "products"):
            given = np.asarray(getattr(self, name))
            matrix = given.astype(int)
            if matrix.size == 0:
                matrix = matrix.reshape(shape)
            if matrix.shape != shape:
                msg = f"{name} has shape {matrix.shape}, expected {shape}"
                raise ValueError(msg)
            if not np.array_equal(given.reshape(shape), matrix):
                msg = f"{name} has non-integer coefficients"
                raise ValueError(msg)
            if (matrix < 0).any():
                msg = f"{name} has negative coefficients"
                raise ValueError(msg)
            object.__setattr__(self, name, rx._misc.totuple(matrix))
            object.__setattr__(self, f"_{name}", _read_only(matrix))

        for ts in self.transition_states:
            if ts is not None and not (
                0 <= ts < shape[0] and is_transition_state(self.compounds[ts])
            ):
                msg = f"invalid transition state index {ts}"
                raise ValueError(msg)

        halves = [k for pair in self.equilibria for k in pair]
        for i, j in self.equilibria:
            if not (
                0 <= i < shape[1]
                and 0 <= j < shape[1]
                and halves.count(i) == halves.count(j) == 1
                and self.transition_states[i] is None
                and self.transition_states[j] is None
                and np.array_equal(self._reactants[:, i], self._products[:, j])
                and np.array_equal(self._products[:, i], self._reactants[:, j])
            ):
                msg = f"invalid equilibrium {(i, j)}: halves must be distinct mutually reverse reactions without transition states"
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
            If an equilibrium involves a transition state.

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
        reactions are ignored, an equilibrium absorbs reactions between its two
        sides, and coefficients of a compound repeated on one side add up. An empty
        side denotes a source or a sink. Compounds are numbered in order of
        appearance in the resulting reactions. Furthermore, reactions
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
               equilibria=())

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
               equilibria=())

        Equilibria produce twice as many direct reactions, paired by `equilibria`:

        >>> scheme = Scheme.parse("A <=> B  // an equilibrium")
        >>> scheme.reactions
        ('A -> B', 'B -> A')
        >>> scheme.equilibria
        ((0, 1),)

        Adding twice the same reaction results in a single reaction being added,
        and an equilibrium absorbs reactions between its two sides (extra
        whitespaces are ignored):

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
               equilibria=())
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
        ('A', 'A‡', "A'‡")
        >>> scheme.reactions
        ("A -> A'‡",)
        >>> scheme.transition_states
        (1,)

        In any case, it's not clear how a reaction barrier be defined in such a
        case. If you have a use case, don't hesitate to
        [open an issue](https://github.com/geem-lab/overreact/issues/), we'll be
        happy to hear from you.
        """

        def _coefficients(terms) -> dict[str, int]:
            """Add up coefficients of repeated compounds."""
            coefficients: dict[str, int] = {}
            for coefficient, compound in terms:
                coefficients[compound] = coefficients.get(compound, 0) + coefficient
            return coefficients

        records: dict[tuple, tuple[dict, dict, str | None, bool]] = {}

        def _add(reactants, products, transition) -> None:
            sides = frozenset(reactants.items()), frozenset(products.items())
            if transition is None and frozenset(sides) in records:
                return  # absorbed by an equilibrium
            records.setdefault(
                (*sides, transition), (reactants, products, transition, False)
            )

        after_transitions = collections.defaultdict(list)
        before_transitions = collections.defaultdict(list)
        for terms_before, terms_after, is_equilibrium in _parse_reactions(text):
            reactants, products = (
                _coefficients(terms_before),
                _coefficients(terms_after),
            )
            if is_equilibrium:
                if any(map(is_transition_state, itertools.chain(reactants, products))):
                    msg = "transition states cannot take part in equilibria"
                    raise ValueError(msg)
                sides = frozenset(reactants.items()), frozenset(products.items())
                records.pop((*sides, None), None)
                records.pop((*reversed(sides), None), None)
                records.setdefault(frozenset(sides), (reactants, products, None, True))
                continue

            # it's assumed that if a transition shows up,
            #   1. it's the only compound in its side of the reaction, and
            #   2. its coefficient equals one
            last_reactant = next(reversed(reactants), None)
            last_product = next(reversed(products), None)
            if last_reactant is not None and is_transition_state(last_reactant):
                for before in before_transitions[last_reactant]:
                    _add(before, products, last_reactant)
                after_transitions[last_reactant].append(products)
            elif last_product is not None and is_transition_state(last_product):
                for after in after_transitions[last_product]:
                    _add(reactants, after, last_product)
                before_transitions[last_product].append(reactants)
            else:
                _add(reactants, products, None)

        compounds: dict[str, int] = {}
        columns = []
        equilibria = []
        for reactants, products, transition, is_equilibrium in records.values():
            for compound in itertools.chain(
                reactants, (transition,) if transition else (), products
            ):
                compounds.setdefault(compound, len(compounds))
            if is_equilibrium:
                equilibria.append((len(columns), len(columns) + 1))
                columns += [(reactants, products, None), (products, reactants, None)]
            else:
                columns.append((reactants, products, transition))

        shape = (len(compounds), len(columns))
        reactants_matrix, products_matrix = np.zeros(shape, int), np.zeros(shape, int)
        for j, (reactants, products, _) in enumerate(columns):
            for compound, coefficient in reactants.items():
                reactants_matrix[compounds[compound], j] = coefficient
            for compound, coefficient in products.items():
                products_matrix[compounds[compound], j] = coefficient
        return cls(
            compounds=tuple(compounds),
            reactants=reactants_matrix,
            products=products_matrix,
            transition_states=tuple(
                None if transition is None else compounds[transition]
                for _, _, transition in columns
            ),
            equilibria=tuple(equilibria),
        )

    @cached_property
    def reactions(self) -> tuple[str, ...]:
        """Reactions as text, without transition states."""
        return tuple(
            f"{_unparse_side(self._side(r))} -> {_unparse_side(self._side(p))}".strip()
            for r, p in zip(self._reactants.T, self._products.T, strict=True)
        )

    @cached_property
    def is_half_equilibrium(self) -> tuple[bool, ...]:
        """An indicator of whether a reaction is half of an equilibrium."""
        halves = {k for pair in self.equilibria for k in pair}
        return tuple(j in halves for j in range(len(self.transition_states)))

    @cached_property
    def A(self) -> np.ndarray:  # noqa: N802
        """Net stoichiometric coefficients, from reactants to products."""
        return _read_only(self._products - self._reactants)

    @cached_property
    def molecularity(self) -> np.ndarray:
        """Number of reactant molecules of each reaction, zero for sources."""
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
        fixed = np.isin(self.compounds, list(concentrations))
        for ts in self.transition_states:
            if ts is not None and fixed[ts]:
                msg = f"cannot fix transition state {self.compounds[ts]}"
                raise ValueError(msg)
        position = np.cumsum(~fixed) - 1
        values = np.array(
            [concentrations[c] for c in itertools.compress(self.compounds, fixed)],
            dtype=float,
        )
        return (
            Scheme(
                compounds=tuple(itertools.compress(self.compounds, ~fixed)),
                reactants=self._reactants[~fixed],
                products=self._products[~fixed],
                transition_states=tuple(
                    None if ts is None else int(position[ts])
                    for ts in self.transition_states
                ),
                equilibria=self.equilibria,
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
        ...            equilibria=((1, 2),)))
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
        reverse = {j for _, j in self.equilibria}
        lines = []
        for i, (reaction, transition_state) in enumerate(
            zip(self.reactions, self.transition_states, strict=True),
        ):
            if i in reverse:
                continue
            if transition_state is not None:
                reaction = reaction.replace(
                    "->",
                    f"-> {self.compounds[transition_state]} ->",
                )
            elif self.is_half_equilibrium[i]:
                reaction = reaction.replace("->", "<=>")
            lines.append(reaction)
        return "\n".join(lines)

    @cached_property
    def _activated(self) -> np.ndarray:
        """Coefficients of the activated state of each reaction."""
        activated = self._products.copy()
        for _, j in self.equilibria:
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
           equilibria=())
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
    is_equilibrium : bool

    Examples
    --------
    >>> r = "E + S <=> ES -> ES‡ -> E + P"
    >>> for reactants, products, is_half_equilibrium in _parse_reactions(r):
    ...     print(reactants, products, is_half_equilibrium)
    ((1, 'E'), (1, 'S')) ((1, 'ES'),) True
    ((1, 'ES'),) ((1, 'ES‡'),) False
    ((1, 'ES‡'),) ((1, 'E'), (1, 'P')) False
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

            yield reactants, products, arrow == "<=>"


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
