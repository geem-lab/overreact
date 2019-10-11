#!/usr/bin/env python3

"""Module dedicated to classifying molecules into point groups."""

import re as _re

import numpy as _np
from scipy.cluster.hierarchy import fcluster as _fcluster
from scipy.cluster.hierarchy import linkage as _linkage
from scipy.spatial import cKDTree as _KDTree
from scipy.spatial.distance import pdist as _pdist
from scipy.spatial.distance import squareform as _squareform
from scipy.spatial.transform import Rotation as _Rotation


def symmetry_number(symbol):
    """Return rotational symmetry number for point group.

    This function has a set of the most common point groups precomputed, but is
    able to calculate the symmetry number if it is not found in known tables.
    Precomputed values are from doi:10.1007/s00214-007-0328-0, ORCA's manual
    (page 279) and Advances in Physical Organic Chemistry (2016), by Ian
    Williams, Nick Williams (page 44).

    Parameters
    ----------
    symbol : str
        Point group symbol.

    Returns
    -------
    int

    Examples
    --------
    >>> symmetry_number("C4")
    4
    >>> symmetry_number("C4") == symmetry_number("C4h")
    True
    >>> symmetry_number("C6")
    6
    >>> symmetry_number("C6") == symmetry_number("C6v")
    True
    >>> symmetry_number("C6") == symmetry_number("C6h")
    True
    >>> symmetry_number("D2") == symmetry_number("Vh")
    True
    >>> symmetry_number("D4")
    8
    >>> symmetry_number("D6")
    12
    >>> symmetry_number("D6") == symmetry_number("D6d")
    True
    >>> symmetry_number("S6")
    3
    >>> symmetry_number("T")
    12
    """
    symbol = symbol.strip().lower()

    if symbol in {"c1", "ci", "cs", "c∞v", "k", "r3"}:
        return 1
    elif symbol in {"c2", "c2v", "c2h", "d∞h", "s4"}:
        return 2
    # elif symbol in {"c3", "c3v", "c3h", "s6"}:
    #     return 3
    elif symbol in {"c4", "c4v", "c4h", "d2", "d2d", "d2h", "s8", "vh"}:
        return 4
    # elif symbol in {"c6", "c6v", "c6h", "d3", "d3d", "d3h"}:
    #     return 6
    # elif symbol in {"d4", "d4d", "d4h"}:
    #     return 8
    elif symbol in {"c12", "c12v", "c12h", "d6", "d6d", "d6h", "s24", "t", "td"}:
        return 12
    elif symbol in {"c24", "c24v", "c24h", "d12", "d12d", "d12h", "s48", "oh"}:
        return 24
    elif symbol in {"c60", "c60v", "c60h", "d30", "d30d", "d30h", "s120", "ih"}:
        return 60

    pieces = _re.match(
        r"(?P<letter>[^\s]+)(?P<number>\d+)(?P<type>[^\s]+)?", symbol
    ).groupdict()

    if pieces["letter"] == "c":
        return int(pieces["number"])
    elif pieces["letter"] == "d":
        return 2 * int(pieces["number"])
    elif pieces["letter"] == "s":
        return int(pieces["number"]) // 2


def find_point_group(atommasses, atomcoords, proper_axes=None, rtol=0.0, atol=1.0e-2):
    """Determine point group of structure.

    Parameters
    ----------
    atommasses : array-like
    atomcoords : array-like
        Atomic coordinates.
    proper_axes : sequence of tuples of int, array-like, optional
        Proper symmetry axes of rotation.
    rtol : float, optional
        The relative tolerance parameter (see `numpy.isclose`).
    atol : float, optional
        The absolute tolerance parameter (see `numpy.isclose`).

    Returns
    -------
    str

    Examples
    --------
    >>> find_point_group([1], [[0, 0, 1]])
    'K'
    >>> find_point_group([1, 1], [[0, 0, 1], [0, 0, 0]])
    'D∞h'
    >>> find_point_group([1.008, 35.45], [[0, 0, 1], [0, 0, 0]])
    'C∞v'
    >>> find_point_group([16, 12, 16], [[1, 0, 1], [1, 0, 0], [1, 0, -1]])
    'D∞h'
    >>> find_point_group([16, 12, 32], [[1, 1, 1], [1, 1, 0], [1, 1, -1]])
    'C∞v'
    >>> find_point_group([12, 12, 12, 12], [[1, 0, 0],
    ...                                     [0, 1, 0],
    ...                                     [0, 0, 0],
    ...                                     [1, 1, 0]])
    'D4h'

    """
    if len(atommasses) == 1:  # atom
        return "K"
    elif len(atommasses) == 2:  # diatomic molecule
        if atommasses[0] == atommasses[1]:
            return "D∞h"
        return "C∞v"

    groups = equivalent_atoms(atommasses, atomcoords)
    moments, axes, atomcoords = inertia(atommasses, atomcoords)
    rotor_class = classify_rotor(moments)

    if rotor_class[1] == "linear":
        return _find_point_group_linear(atomcoords, groups, rtol=rtol, atol=atol)
    elif rotor_class[0] == "spheric":
        if proper_axes is None:
            proper_axes = get_proper_axes(
                atomcoords, groups, axes, rotor_class, rtol=rtol, atol=atol
            )

        return _find_point_group_spheric(
            atomcoords, groups, axes, rotor_class, proper_axes, rtol=rtol, atol=atol
        )
    elif rotor_class[0] == "asymmetric":
        if proper_axes is None:
            proper_axes = get_proper_axes(
                atomcoords, groups, axes, rotor_class, rtol=rtol, atol=atol
            )

        return _find_point_group_asymmetric(
            atomcoords, groups, axes, rotor_class, proper_axes, rtol=rtol, atol=atol
        )
    else:  # symmetric
        if proper_axes is None:
            proper_axes = get_proper_axes(
                atomcoords, groups, axes, rotor_class, rtol=rtol, atol=atol
            )

        return _find_point_group_symmetric(
            atomcoords, groups, axes, rotor_class, proper_axes, rtol=rtol, atol=atol
        )


def _find_point_group_linear(atomcoords, groups, rtol=0.0, atol=1.0e-2):
    """Find point group for linear rotors.

    Point groups searched for are: D∞h, C∞v.

    See find_point_group for information on parameters and return values.
    """
    if has_inversion_center(atomcoords, groups, rtol=rtol, atol=atol):
        return "D∞h"
    else:
        return "C∞v"


def _find_point_group_spheric(
    atomcoords, groups, axes, rotor_class, proper_axes=None, rtol=0.0, atol=1.0e-2
):
    """Find point group for spheric tops.

    Point groups searched for are: Td, Oh, Ih.
    I might eventually search for T, Th, O in the future.

    See find_point_group for information on parameters and return values.
    """
    if not has_inversion_center(atomcoords, groups, rtol=rtol, atol=atol):
        return "Td"

    if proper_axes is None:
        proper_axes = get_proper_axes(
            atomcoords, groups, axes, rotor_class, rtol=rtol, atol=atol
        )

    for n, _ in proper_axes:
        if n == 5:
            return "Ih"
        elif n < 5:
            break
    return "Oh"

    # see https://www.chem.uci.edu/~lawm/9-28.pdf for more about high
    # symmetry groups
    # see too
    # http://web.mit.edu/5.03/www/readings/point_groups/point_groups.pdf

    # the following workflow is loosely inspired by some articles:
    # 1. doi:10.1016/0097-8485(76)80004-6
    #     elif has_3_C4(atomcoords):
    #         if has_center_of_inversion(atomcoords):
    #             return "oh"
    #         else:
    #             return "o"
    #     elif has_3_S4_parallel_to_C2(atomcoords):
    #         return "td"
    #     elif has_center_of_inversion(atomcoords):
    #         return "th"
    #     else:
    #         return "t"


def _find_point_group_asymmetric(
    atomcoords, groups, axes, rotor_class, proper_axes=None, rtol=0.0, atol=1.0e-2
):
    """Find point group for asymmetric tops.

    Point groups searched for here are: C1, Ci, Cs.
    Point groups delegated are: Cn, Cnh, Cnv, Dn, Dnh, Dnd, Sn.

    See find_point_group for information on parameters and return values.
    """
    if proper_axes is None:
        proper_axes = get_proper_axes(
            atomcoords, groups, axes, rotor_class, rtol=rtol, atol=atol
        )

    if proper_axes:
        return _find_point_group_symmetric(
            atomcoords, groups, axes, rotor_class, proper_axes, rtol=rtol, atol=atol
        )
    elif rotor_class[1] in {"regular planar", "irregular planar"} or get_mirror_planes(
        atomcoords, groups, axes, rotor_class, proper_axes, rtol=rtol, atol=atol
    ):
        return "Cs"
    elif has_inversion_center(atomcoords, groups, rtol=rtol, atol=atol):
        return "Ci"
    return "C1"


def _find_point_group_symmetric(
    atomcoords, groups, axes, rotor_class, proper_axes=None, rtol=0.0, atol=1.0e-2
):
    """Find point group for symmetric tops.

    Point groups delegated are: Cn, Cnh, Cnv, Dn, Dnh, Dnd, Sn.

    See find_point_group for information on parameters and return values.
    """
    if proper_axes is None:
        proper_axes = get_proper_axes(
            atomcoords, groups, axes, rotor_class, rtol=rtol, atol=atol
        )
    n_principal = proper_axes[0][0]

    count_twofold = 0
    for n, _ in proper_axes:
        if n == 2:
            count_twofold += 1
        if n_principal == count_twofold:
            return _find_point_group_symmetric_dihedral(
                atomcoords, groups, axes, rotor_class, proper_axes, rtol=rtol, atol=atol
            )
        if n < 2:
            break
    return _find_point_group_symmetric_nondihedral(
        atomcoords, groups, axes, rotor_class, proper_axes, rtol=rtol, atol=atol
    )

    # the following workflow is loosely inspired by some articles:
    # 1. doi:10.1016/0097-8485(76)80004-6
    # if has_proper_ax_of_highest_order(atomcoords):
    #     if has_proper_ax_larger_than_2_or_3_C2_perpendicular(
    #         atomcoords
    #     ):
    #         if has_S2n_parallel_to_Cn(atomcoords):
    #             if has_n_sigma_d(atomcoords):
    #                 return "dnd"
    #             else:
    #                 return "s2n"
    #         elif has_nC2_perpendicular_to_Cn(atomcoords):
    #             if has_sigma_h(atomcoords):
    #                 return "dnh"
    #             else:
    #                 return "dn"
    #         elif has_n_sigma_v(atomcoords):
    #             return "cnv"
    #         elif has_sigma_h(atomcoords):
    #             return "cnh"
    #         else:
    #             return "cn"


def _find_point_group_symmetric_dihedral(
    atomcoords, groups, axes, rotor_class, proper_axes=None, rtol=0.0, atol=1.0e-2
):
    """Find a dihedral point group for symmetric tops.

    Point groups searched for are: Dn, Dnh, Dnd.

    See find_point_group for information on parameters and return values.
    """
    if proper_axes is None:
        proper_axes = get_proper_axes(
            atomcoords, groups, axes, rotor_class, rtol=rtol, atol=atol
        )
    mirror_axes = get_mirror_planes(
        atomcoords, groups, axes, rotor_class, proper_axes, rtol=rtol, atol=atol
    )

    if mirror_axes:
        if mirror_axes[0][0] == "h":
            return f"D{proper_axes[0][0]}h"
        elif len([v for c, v in mirror_axes if c == "v"]) == proper_axes[0][0]:
            # all vertical mirror planes are dihedral for Dnd point groups
            return f"D{proper_axes[0][0]}d"
    return f"D{proper_axes[0][0]}"


def _find_point_group_symmetric_nondihedral(
    atomcoords, groups, axes, rotor_class, proper_axes=None, rtol=0.0, atol=1.0e-2
):
    """Find a nondihedral point group for symmetric tops.

    Point groups searched for are: Cn, Cnh, Cnv, Sn.

    See find_point_group for information on parameters and return values.
    """
    if proper_axes is None:
        proper_axes = get_proper_axes(
            atomcoords, groups, axes, rotor_class, rtol=rtol, atol=atol
        )
    mirror_axes = get_mirror_planes(
        atomcoords, groups, axes, rotor_class, proper_axes, rtol=rtol, atol=atol
    )

    if mirror_axes:
        if mirror_axes[0][0] == "h":
            return f"C{proper_axes[0][0]}h"
        elif len([v for c, v in mirror_axes if c == "v"]) == proper_axes[0][0]:
            return f"C{proper_axes[0][0]}v"

    improper_axes = get_improper_axes(
        atomcoords, groups, axes, rotor_class, proper_axes, rtol=rtol, atol=atol
    )
    if improper_axes:
        return f"S{improper_axes[0][0]}"
    return f"C{proper_axes[0][0]}"


def get_proper_axes(
    atomcoords, groups, axes, rotor_class, rtol=0.0, atol=1.0e-2, slack=0.735
):
    """Get proper symmetry axes and their orders.

    Parameters
    ----------
    atomcoords : array-like
        Atomic coordinates centered at the center of mass.
    groups : sequence of sequence of int
        Groups of symmetry equivalent atoms, in ascending order of size.
    axes : array-like
        Normalized principal axes of inertia.
    rotor_class : tuple of str
    rtol : float, optional
        The relative tolerance parameter (see `numpy.isclose`).
    atol : float, optional
        The absolute tolerance parameter (see `numpy.isclose`).
    slack : float, optional
        Number to multiply rtol and atol prior comparisons.

    Returns
    -------
    sequence of tuples of int, array-like
        Ordered list of tuples in the format ``(order, (x, y, z))``.

    Notes
    -----
    This function has some limitations. First, no C1 axis is never returned.
    Second, an empty list is returned if the structure has a single atom. And
    third, the largest symmetry axis found has order not greater than the
    maximum number of symmetry equivalent atoms, which particularly impacts
    linear rotors (for instance, the largest axis found for the hydrogen
    molecule is twofold, while no axis is found for hydrogen chloride).
    Furthermore, linear rotors are considered symmetric prolate tops, which
    means a single axis is returned (no perpendicular axes). Either way, this
    should be of little impact, as only cases whose groups can easily be
    inferred with no or little knowledge of symmetry axis are affected.

    Examples
    --------
    >>> from overreact.datasets import logfiles
    >>> data = logfiles["diborane"]
    >>> groups = equivalent_atoms(data.atommasses, data.atomcoords[-1])
    >>> moments, axes, atomcoords = inertia(data.atommasses,
    ...                                     data.atomcoords[-1])
    >>> rotor_class = classify_rotor(moments)
    >>> get_proper_axes(atomcoords, groups, axes, rotor_class)
    [(2, (1.0, 0.0, 0.0)),
     (2, (0.0, 0.0, 1.0)),
     (2, (0.0, 1.0, 0.0))]
    """
    rtol, atol = slack * rtol, slack * atol

    if rotor_class[1] == "atomic" or len(atomcoords) == 1:
        return list()

    axes = _np.asanyarray(axes)
    atomcoords = _np.asanyarray(atomcoords)
    orders = _guess_orders(groups, rotor_class)

    def _update_axes(
        ax,
        axes,  # found axes
        nondeg_axes=None,
        atomcoords=atomcoords,
        groups=groups,
        orders=orders,
        rtol=rtol,
        atol=atol,
        normalize=False,
    ):
        """Update axes with ax, and return it with added order (or None)."""
        if nondeg_axes is None:
            nondeg_axes = list()

        if normalize:
            norm = _np.linalg.norm(ax)
            if _np.isclose(norm, 0.0, rtol=rtol, atol=atol):
                return axes, None
            ax = ax / norm

        if not all(
            _np.isclose(ax @ v, 0.0, rtol=rtol, atol=atol) for v in nondeg_axes
        ) or any(
            _np.isclose(_np.abs(ax @ v), 1.0, rtol=rtol, atol=atol) for o, v in axes
        ):
            return axes, None

        for order in orders[::-1]:
            if all(
                is_symmetric(
                    atomcoords[group],
                    operation("c", order=order, axis=ax),
                    rtol=rtol,
                    atol=atol,
                )
                for group in groups[::-1]
            ):
                axes.append((order, tuple(ax)))
                return axes, order

        return axes, None

    found_axes = list()
    nondeg_axes = list()
    if rotor_class[0] == "symmetric prolate":
        nondeg_axes = [axes[:, 0]]
        found_axes, order = _update_axes(axes[:, 0], found_axes)
    elif rotor_class[0] == "symmetric oblate":
        nondeg_axes = [axes[:, 2]]
        found_axes, order = _update_axes(axes[:, 2], found_axes)
    elif rotor_class[0] == "asymmetric":
        for ax in axes.T:
            found_axes, order = _update_axes(ax, found_axes)
        return sorted(found_axes, reverse=True)

    for group in groups:
        for i, a in enumerate(group):
            through_ax = atomcoords[a]
            found_axes, order = _update_axes(
                through_ax, found_axes, nondeg_axes, normalize=True
            )
            if rotor_class[0] == "spheric" and order == 5:
                return sorted(found_axes, reverse=True)

            for b in group[:i]:
                midpoint_ax = atomcoords[a] + atomcoords[b]
                found_axes, order = _update_axes(
                    midpoint_ax, found_axes, nondeg_axes, normalize=True
                )
                if rotor_class[0] == "spheric" and order == 5:
                    return sorted(found_axes, reverse=True)

    if rotor_class[0] == "spheric":
        twofold_axes = [ax for o, ax in found_axes if o == 2]
        for i, ax_a in enumerate(twofold_axes):
            for ax_b in twofold_axes[:i]:
                ax = _np.cross(ax_a, ax_b)
                found_axes, order = _update_axes(
                    ax, found_axes, nondeg_axes, normalize=True
                )
                if order == 5:
                    return sorted(found_axes, reverse=True)

    return sorted(found_axes, reverse=True)


def _guess_orders(groups, rotor_class):
    """Guess possible group orders based on groups.

    The guess consists of the numbers two to n, where n is the number of
    elements in the largest group. For cubic groups, n is taken to be at most
    five. The trivial order one is never guessed.

    Parameters
    ----------
    groups : sequence of sequence of int
        Groups of symmetry equivalent atoms, in ascending order of size.
    rotor_class : tuple of str

    Returns
    -------
    sequence of int

    Examples
    --------
    >>> _guess_orders([[0], [1, 2, 3, 4], [5, 6, 7, 8]],
    ...               ("symmetric prolate", "nonplanar"))
    range(2, 5)
    >>> _guess_orders([[0], [1, 2, 3, 4], [5, 6, 7, 8]],
    ...               ("spheric", "nonplanar"))
    range(2, 5)
    >>> _guess_orders([[0], [1, 2, 3, 4], [5, 6, 7, 8, 9, 10]],
    ...               ("symmetric prolate", "nonplanar"))
    range(2, 7)
    >>> _guess_orders([[0], [1, 2, 3, 4], [5, 6, 7, 8, 9, 10]],
    ...               ("spheric", "nonplanar"))
    range(2, 6)
    """
    max_order = len(groups[-1])
    if rotor_class[0] == "spheric":
        max_order = min(max_order, 5)
    return range(2, max_order + 1)


def get_improper_axes(
    atomcoords,
    groups,
    axes,
    rotor_class,
    proper_axes=None,
    rtol=0.0,
    atol=1.0e-2,
    slack=1.888,
):
    """Get improper symmetry axes and their orders.

    Parameters
    ----------
    atomcoords : array-like
        Atomic coordinates centered at the center of mass.
    groups : sequence of sequence of int
        Groups of symmetry equivalent atoms, in ascending order of size.
    axes : array-like
        Normalized principal axes of inertia.
    rotor_class : tuple of str
    proper_axes : sequence of tuples of int, array-like, optional
        Proper symmetry axes of rotation.
    rtol : float, optional
        The relative tolerance parameter (see `numpy.isclose`).
    atol : float, optional
        The absolute tolerance parameter (see `numpy.isclose`).
    slack : float, optional
        Number to multiply rtol and atol prior comparisons.

    Returns
    -------
    sequence of tuples of int, array-like

    Examples
    --------
    >>> from overreact.datasets import logfiles
    >>> data = logfiles["methane"]
    >>> groups = equivalent_atoms(data.atommasses, data.atomcoords[-1])
    >>> moments, axes, atomcoords = inertia(data.atommasses,
    ...                                     data.atomcoords[-1])
    >>> rotor_class = classify_rotor(moments)
    >>> get_improper_axes(atomcoords, groups, axes, rotor_class)
    [(4, (0.0, 0.0, -1.0)),
     (4, (0.0, -1.0, 0.0)),
     (4, (-1.0, 0.0, 0.0))]
    """
    rtol, atol = slack * rtol, slack * atol

    if rotor_class[1] == "atomic" or len(atomcoords) == 1:
        return list()

    axes = _np.asanyarray(axes)
    atomcoords = _np.asanyarray(atomcoords)

    if proper_axes is None:
        proper_axes = get_proper_axes(
            atomcoords, groups, axes, rotor_class, rtol=rtol, atol=atol
        )

    def _update_axes(
        n,
        ax,
        axes,  # found axes
        atomcoords=atomcoords,
        groups=groups,
        rtol=rtol,
        atol=atol,
        normalize=False,
    ):
        """Update axes with ax and return it."""
        if normalize:
            norm = _np.linalg.norm(ax)
            if _np.isclose(norm, 0.0, rtol=rtol, atol=atol):
                return axes
            ax = ax / norm

        for order in [2 * n, n]:
            if all(
                is_symmetric(
                    atomcoords[group],
                    operation("s", order=order, axis=ax),
                    rtol=rtol,
                    atol=atol,
                )
                for group in groups[::-1]
            ):
                axes.append((order, tuple(ax)))
                break

        return axes

    found_axes = list()
    for n, ax in proper_axes:
        found_axes = _update_axes(n, ax, found_axes)
    return sorted(found_axes, reverse=True)


def get_mirror_planes(
    atomcoords,
    groups,
    axes,
    rotor_class,
    proper_axes=None,
    rtol=0.0,
    atol=1.0e-2,
    slack=2.020,
):
    """Get (and attempt to classify) mirror plane normal axes.

    Parameters
    ----------
    atomcoords : array-like
        Atomic coordinates centered at the center of mass.
    groups : sequence of sequence of int
        Groups of symmetry equivalent atoms, in ascending order of size.
    axes : array-like
        Normalized principal axes of inertia.
    rotor_class : tuple of str
    proper_axes : sequence of tuples of int, array-like, optional
        Proper symmetry axes of rotation.
    rtol : float, optional
        The relative tolerance parameter (see `numpy.isclose`).
    atol : float, optional
        The absolute tolerance parameter (see `numpy.isclose`).
    slack : float, optional
        Number to multiply rtol and atol prior comparisons.

    Returns
    -------
    sequence of tuples of str, array-like

    Notes
    -----
    This function only classifies horizontal and vertical planes. Besides,
    horizontal classification always takes precedence. Unclassified planes are
    associated with an empty string. This is all that is needed for point group
    classification.

    This function has some limitations. First, an empty list is always returned
    if the structure has a single atom. Second, an empty list is returned for
    C∞v as well. And third, since classifications are made against proper
    symmetry axes, no classification can be made for point groups having no
    proper symmetry axes (e.g., the mirror plane in Cs is not classified).

    This should be of little impact, as only cases whose groups can easily be
    inferred with no or little knowledge of mirror planes are affected.

    Examples
    --------
    >>> from overreact.datasets import logfiles
    >>> data = logfiles["1-iodo-2-chloroethylene"]
    >>> groups = equivalent_atoms(data.atommasses, data.atomcoords[-1])
    >>> moments, axes, atomcoords = inertia(data.atommasses,
    ...                                     data.atomcoords[-1])
    >>> rotor_class = classify_rotor(moments)
    >>> get_mirror_planes(atomcoords, groups, axes, rotor_class)
    [('', (0.0, 0.0, 1.0))]
    """
    rtol, atol = slack * rtol, slack * atol

    if rotor_class[1] == "atomic" or len(atomcoords) == 1:
        return list()

    axes = _np.asanyarray(axes)
    atomcoords = _np.asanyarray(atomcoords)

    if proper_axes is None:
        proper_axes = get_proper_axes(
            atomcoords, groups, axes, rotor_class, rtol=rtol, atol=atol
        )

    def _update_axes(
        ax,
        axes,  # found axes
        nondeg_axes=None,
        atomcoords=atomcoords,
        groups=groups,
        rtol=rtol,
        atol=atol,
        normalize=False,
        proper_axes=proper_axes,
    ):
        """Update axes with ax and return it."""
        if nondeg_axes is None:
            nondeg_axes = list()

        if normalize:
            norm = _np.linalg.norm(ax)
            if _np.isclose(norm, 0.0, rtol=rtol, atol=atol):
                return axes
            ax = ax / norm

        if (
            not all(_np.isclose(ax @ v, 0.0, rtol=rtol, atol=atol) for v in nondeg_axes)
            # or not all(  # TODO(schneiderfelipe): improve this filter
            #     _np.isclose(ax @ v, 0.0, rtol=rtol, atol=atol)
            #     or _np.isclose(_np.abs(ax @ v), 1.0, rtol=rtol, atol=atol)
            #     for n, v in proper_axes
            # )
            or any(
                _np.isclose(_np.abs(ax @ v), 1.0, rtol=rtol, atol=atol) for c, v in axes
            )
        ):
            return axes

        if all(
            is_symmetric(
                atomcoords[group], operation("σ", axis=ax), rtol=rtol, atol=atol
            )
            for group in groups[::-1]
        ):
            class_ = ""
            if any(
                _np.isclose(_np.abs(ax @ v), 1.0, rtol=rtol, atol=atol)
                for n, v in proper_axes
                if proper_axes[0][0] == n
            ):
                class_ = "h"
            elif any(
                _np.isclose(ax @ v, 0.0, rtol=rtol, atol=atol)
                for n, v in proper_axes
                if proper_axes[0][0] == n
            ):
                class_ = "v"
            axes.append((class_, tuple(ax)))

        return axes

    def _kf(x):
        """Order function for returned list."""
        c, v = x
        if c:
            return -ord(c), v
        else:
            return 0, v

    found_axes = list()
    nondeg_axes = list()
    if rotor_class[0] == "symmetric prolate":
        nondeg_axes = [axes[:, 0]]
        found_axes = _update_axes(axes[:, 0], found_axes)
    elif rotor_class[0] == "symmetric oblate":
        nondeg_axes = [axes[:, 2]]
        found_axes = _update_axes(axes[:, 2], found_axes)
    elif rotor_class[0] == "asymmetric":
        for ax in axes.T:
            found_axes = _update_axes(ax, found_axes)
        return sorted(found_axes, reverse=True, key=_kf)

    for group in groups:
        for i, a in enumerate(group):
            for b in group[:i]:
                ab_ax = atomcoords[b] - atomcoords[a]
                found_axes = _update_axes(
                    ab_ax, found_axes, nondeg_axes, normalize=True
                )

    return sorted(found_axes, reverse=True, key=_kf)


def has_inversion_center(atomcoords, groups, rtol=0.0, atol=1.0e-2, slack=1.888):
    """Check whether the molecule has an inversion center.

    Parameters
    ----------
    atomcoords : array-like
        Atomic coordinates centered at the center of mass.
    groups : sequence of sequence of int
        Groups of symmetry equivalent atoms, in ascending order of size.
    rtol : float, optional
        The relative tolerance parameter (see `numpy.isclose`).
    atol : float, optional
        The absolute tolerance parameter (see `numpy.isclose`).
    slack : float, optional
        Number to multiply rtol and atol prior comparisons.

    Returns
    -------
    bool

    Examples
    --------
    >>> has_inversion_center([[0, 0, -1], [0, 0, 1]], [[0, 1]])
    True
    >>> has_inversion_center([[0, 0, -1], [0, 0, 1], [1, 1, 1]], [[0, 1], [2]])
    False
    """
    rtol, atol = slack * rtol, slack * atol

    atomcoords = _np.asanyarray(atomcoords)
    return all(
        is_symmetric(atomcoords[group], operation("i"), rtol=rtol, atol=atol)
        for group in groups[::-1]
    )


def is_symmetric(atomcoords, op, rtol=0.0, atol=1.0e-2, slack=10.256):
    """Check if structure satisfies symmetry.

    Parameters
    ----------
    atomcoords : array-like
        Atomic coordinates centered at the center of mass.
    op : array-like
    rtol : float, optional
        The relative tolerance parameter (see `numpy.isclose`).
    atol : float, optional
        The absolute tolerance parameter (see `numpy.isclose`).
    slack : float, optional
        Number to multiply rtol and atol prior comparisons.

    Returns
    -------
    bool

    Examples
    --------
    >>> is_symmetric([[1, 0, 0],
    ...               [0, 1, 0],
    ...               [0, 0, 0]], operation("c", order=4, axis=[0, 0, 1]))
    False
    >>> is_symmetric([[1, 0, 0],
    ...               [0, 1, 0],
    ...               [0, 0, 0]], operation("c", order=2, axis=[1, 1, 0]))
    True
    >>> is_symmetric([[1, 0, 0],
    ...               [0, 1, 0],
    ...               [0, 0, 0]], operation("sigma", axis=[0, 0, 1]))
    True
    >>> is_symmetric([[1, 0, 0],
    ...               [0, 1, 0],
    ...               [0, 0, 0],
    ...               [-1, 0, 0]], operation("c", order=4, axis=[0, 0, 1]))
    False
    >>> is_symmetric([[1, 0, 0],
    ...               [0, 1, 0],
    ...               [0, 0, 0],
    ...               [-1, 0, 0]], operation("sigma", axis=[1, 0, 0]))
    True
    >>> is_symmetric([[1, 0, 0],
    ...               [0, 1, 0],
    ...               [0, 0, 0],
    ...               [-1, 0, 0],
    ...               [0, -1, 0]], operation("c", order=4, axis=[0, 0, 1]))
    True
    >>> is_symmetric([[1, 0, 0],
    ...               [0, 1, 0],
    ...               [0, 0, 0],
    ...               [-1, 0, 0],
    ...               [0, -1, 0]], operation("i"))
    True
    """
    rtol, atol = slack * rtol, slack * atol
    inner_slack = 1.055

    atomcoords, tree = atomcoords, _KDTree(atomcoords)
    d, i = tree.query(atomcoords @ op.T)

    return (
        set(i) == set(range(len(atomcoords)))
        and _np.allclose(d.mean(), 0.0, rtol=rtol, atol=atol)
        and _np.allclose(d.max(), 0.0, rtol=inner_slack * rtol, atol=inner_slack * atol)
    )


def operation(name, order=2, axis=None):
    """Calculate a symmetry operation.

    Parameters
    ----------
    name : str
    order : int, optional
    axis : array-like, optional

    Returns
    -------
    array-like

    Examples
    --------
    >>> operation("e")
    array([[1., 0., 0.],
           [0., 1., 0.],
           [0., 0., 1.]])
    >>> operation("i")
    array([[-1., -0., -0.],
           [-0., -1., -0.],
           [-0., -0., -1.]])
    >>> operation("c", order=4, axis=[0, 0, 1])
    array([[ 0., -1., 0.],
           [ 1.,  0., 0.],
           [ 0.,  0., 1.]])
    >>> operation("σ", axis=[0, 0, 1])
    array([[ 1., 0.,  0.],
           [ 0., 1.,  0.],
           [ 0., 0., -1.]])
    >>> operation("σ", axis=[0, 1, 0])
    array([[ 1.,  0., 0.],
           [ 0., -1., 0.],
           [ 0.,  0., 1.]])
    >>> operation("σ", axis=[1, 0, 0])
    array([[-1., 0., 0.],
           [ 0., 1., 0.],
           [ 0., 0., 1.]])
    >>> operation("s", order=4, axis=[0, 0, 1])
    array([[ 0., -1.,  0.],
           [ 1.,  0.,  0.],
           [ 0.,  0., -1.]])
    >>> operation("s", order=4, axis=[0, 1, 0])
    array([[ 0.,  0., 1.],
           [ 0., -1., 0.],
           [-1.,  0., 0.]])
    >>> operation("s", order=4, axis=[1, 0, 0])
    array([[-1., 0.,  0.],
           [ 0., 0., -1.],
           [ 0., 1.,  0.]])
    """
    if axis is None:
        axis = _np.array([0, 0, 1])

    if name == "i":
        return -_np.eye(3)
    elif name == "e":
        return _np.eye(3)
    elif name in {"c", "σ", "sigma", "s"}:  # normalize axis
        axis = _np.asanyarray(axis)
        axis = axis / _np.linalg.norm(axis)

        if name in {"c", "s"}:
            rotation = _Rotation.from_rotvec(2.0 * _np.pi * axis / order).as_matrix()
        if name in {"σ", "sigma", "s"}:
            reflection = _np.eye(3) - 2.0 * _np.outer(axis, axis)
        if name == "c":
            return rotation
        elif name in {"σ", "sigma"}:
            return reflection
        elif name == "s":
            return rotation @ reflection
    raise ValueError(f"unknown operation '{name}'")


def classify_rotor(moments, rtol=0.0, atol=1.0e-2, slack=0.870):
    """Classify rotors based on moments of inertia.

    See doi:10.1002/jcc.23493.

    Parameters
    ----------
    moments : array-like
        Moments of inertia in ascending order.
    rtol : float, optional
        The relative tolerance parameter (see `numpy.isclose`).
    atol : float, optional
        The absolute tolerance parameter (see `numpy.isclose`).
    slack : float, optional
        Number to multiply rtol and atol prior comparisons.

    Return
    ------
    top, shape : str

    Notes
    -----
    Moments are actually compared by their ratios such that the "aspect ratio"
    of the molecule is what is compared.

    Examples
    --------
    Do examples from
    <https://www.tau.ac.il/~tsirel/dump/Static/knowino.org/wiki/Classification_of_rigid_rotors.html>.

    The idea behind this function is, among other things, to help in
    classifying point groups. For instance, there are the following
    possibilities for atoms or linear molecules:

    >>> classify_rotor([0, 0, 0])
    ('spheric', 'atomic')
    >>> classify_rotor([0, 1, 1])
    ('symmetric prolate', 'linear')

    Spheric tops can be any cubic group:

    >>> classify_rotor([1, 1, 1])
    ('spheric', 'nonplanar')

    Asymmetric tops can be a lot of diferent groups, the ones lacking proper
    axis of symmetry (C1, Ci, Cs) being exclusively found in asymmetric tops.
    Furthermore, any group found for symmetric tops can be found for asymmetric
    tops as well, which complicates things a bit.

    >>> classify_rotor([1, 2, 4])
    ('asymmetric', 'nonplanar')

    Symmetric tops can be found in a subset of the ones found in asymmetric
    tops and they always have a proper axis of symmetry. I further classify
    them here into smaller groups (I believe that this subclassification and
    its relationship with possible point groups can be further improved). For
    instance, planar symmetric tops are always oblate and have a plane of
    symmetry:

    >>> classify_rotor([1, 1, 2])
    ('symmetric oblate', 'regular planar')

    >>> classify_rotor([1, 3, 4])
    ('asymmetric', 'irregular planar')

    >>> classify_rotor([1, 1, 3])
    ('symmetric oblate', 'nonplanar')

    >>> classify_rotor([1, 2, 2])
    ('symmetric prolate', 'nonplanar')
    """
    rtol, atol = slack * rtol, slack * atol
    inner_slack = 2.130

    if _np.isclose(moments[2], 0.0, rtol=inner_slack * rtol, atol=inner_slack * atol):
        return "spheric", "atomic"
    moments = _np.asanyarray(moments) / moments[2]

    # basic tests for tops
    is_oblate = _np.isclose(
        moments[0], moments[1], rtol=inner_slack * rtol, atol=inner_slack * atol
    )
    is_spheric = _np.isclose(
        moments[0], moments[2], rtol=inner_slack * rtol, atol=inner_slack * atol
    )
    is_prolate = _np.isclose(
        moments[1], moments[2], rtol=inner_slack * rtol, atol=inner_slack * atol
    )

    # basic tests for shapes
    fits_line = _np.isclose(
        moments[0], 0.0, rtol=inner_slack * rtol, atol=inner_slack * atol
    )
    fits_plane = _np.isclose(moments[0] + moments[1], moments[2], rtol=rtol, atol=atol)

    is_spheric = is_spheric and is_oblate and is_prolate
    if is_spheric:
        top = "spheric"
    elif is_oblate:
        top = "symmetric oblate"
    elif is_prolate:
        top = "symmetric prolate"
    else:
        top = "asymmetric"

    fits_line = fits_line and is_prolate
    if fits_line:
        shape = "linear"
    elif fits_plane:
        if is_oblate:
            shape = "regular planar"
        else:
            shape = "irregular planar"
    else:
        shape = "nonplanar"

    return top, shape


def inertia(atommasses, atomcoords, align=True):
    """Calculate primary moments and axes from the inertia tensor.

    Parameters
    ----------
    atommasses : array-like
    atomcoords : array-like
        Atomic coordinates.
    align : bool, optional
        If true, the returned coordinates are aligned to the primary axes of
        inertia. The returned axes correspond to the aligned coordinates as
        well.

    Returns
    -------
    moments, axes : array-like
        Primary moments of inertia and associated normalized axes. Axes always
        correspond to returned atomic coordinates.
    atomcoords : array-like
        Coordinates centered at the center of mass and, if align was set to
        True, rotated to the primary axes of inertia.

    Examples
    --------
    >>> import numpy as np
    >>> atommasses = np.array([12.011,  1.008,  1.008,  1.008])  # CH3·
    >>> atomcoords = np.array([[-0.      , -0.      , -1.      ],
    ...                        [ 1.07883 ,  0.      , -1.      ],
    ...                        [-0.539415,  0.934294, -1.      ],
    ...                        [-0.539415, -0.934294, -1.      ]])
    >>> moments, axes, atomcoords = inertia(atommasses, atomcoords)
    >>> moments
    array([1.75977704, 1.75977774, 3.51955478])
    >>> axes
    array([[1., 0., 0.],
           [0., 1., 0.],
           [0., 0., 1.]])
    >>> atomcoords
    array([[ 0.      ,  0.      ,  0.      ],
           [ 1.07883 ,  0.      ,  0.      ],
           [-0.539415,  0.934294,  0.      ],
           [-0.539415, -0.934294,  0.      ]])

    This allows one to calculate the rotational constants in cm-1:

    >>> from scipy.constants import angstrom, atomic_mass, c, centi, h
    >>> h * centi / (8 * np.pi**2 * c * moments * atomic_mass * angstrom**2)
    array([9.5794, 9.5794, 4.7897])
    """
    com = _np.average(atomcoords, axis=0, weights=atommasses)
    atomcoords = atomcoords - com

    w_coords = _np.sqrt(atommasses)[:, _np.newaxis] * atomcoords
    squared_w_coords = w_coords ** 2

    i_xx = _np.sum(squared_w_coords[:, 1] + squared_w_coords[:, 2])
    i_yy = _np.sum(squared_w_coords[:, 0] + squared_w_coords[:, 2])
    i_zz = _np.sum(squared_w_coords[:, 0] + squared_w_coords[:, 1])

    i_xy = -_np.sum(w_coords[:, 0] * w_coords[:, 1])
    i_xz = -_np.sum(w_coords[:, 0] * w_coords[:, 2])
    i_yz = -_np.sum(w_coords[:, 1] * w_coords[:, 2])

    inertia_tensor = _np.array(
        [[i_xx, i_xy, i_xz], [i_xy, i_yy, i_yz], [i_xz, i_yz, i_zz]]
    )
    moments, axes = _np.linalg.eigh(inertia_tensor)
    if align:
        return inertia(atommasses, atomcoords @ axes, align=False)
    return moments, axes, atomcoords


# thresh >= 0.106
def equivalent_atoms(
    atommasses, atomcoords, method="cluster", thresh=0.106, plot=False
):
    """Generate groups of symmetry equivalent atoms.

    Parameters
    ----------
    atommasses : array-like
    atomcoords : array-like
        Atomic coordinates.
    method : str, optioanl
        Method of partitioning: "atommass" (same atoms same groups), "cluster".
    thresh : int, optional
        Threshold to consider atom clusters.

    Returns
    -------
    groups : sequence of sequence of int
        Groups of symmetry equivalent atoms, in ascending order of size. Each
        element in the list is a sequence of indices, one list for each group
        of equivalent atoms. See examples below.

    Notes
    -----
    This function works for up to ten thousand randomly placed atoms and finds
    the equivalent groups in less than a second. As such, the function performs
    sufficiently well for the current use.

    Examples
    --------
    >>> import numpy as np
    >>> atommasses = [12.011,  1.008,  1.008,  1.008]  # CH3·
    >>> atomcoords = np.array([[-0.      , -0.      , -1.      ],
    ...                        [ 1.07883 ,  0.      , -1.      ],
    ...                        [-0.539415,  0.934294, -1.      ],
    ...                        [-0.539415, -0.934294, -1.      ]])
    >>> for indices in equivalent_atoms(atommasses, atomcoords):
    ...     indices
    [0]
    [1, 2, 3]

    >>> atommasses = np.array([14.007, 1.008, 1.008, 1.008])  # ammonia
    >>> atomcoords = np.array(
    ...     [
    ...         [0.0, 0.0, 0.07878],
    ...         [0.0, 0.98569, -0.18381],
    ...         [0.85363, -0.49284, -0.18381],
    ...         [-0.85363, -0.49284, -0.18381],
    ...     ]
    ... )
    >>> for indices in equivalent_atoms(atommasses, atomcoords):
    ...     indices
    [0]
    [1, 2, 3]

    """
    if len(atommasses) == 1:  # atom
        return [[0]]
    elif len(atommasses) == 2:  # diatomic molecule
        if atommasses[0] == atommasses[1]:
            return [[0, 1]]
        return [[0], [1]]

    groups = list()

    def _update_groups_with_condition(condition, groups):
        groups.append(sorted(_np.nonzero(condition)[0]))
        return groups

    if method == "cluster":
        D = _squareform(_pdist(atomcoords))
        # mu = _np.outer(atommasses, atommasses) / _np.add.outer(
        #     atommasses, atommasses
        # )  # reduced masses
        # D = mu * D  # does this help?

        omega = _np.mean(D, axis=0)
        sigma = _np.std(D, axis=0)
        delta = _np.sqrt(_np.sum(D ** 2, axis=0))

        criteria = _np.block(
            [[omega], [sigma], [delta]]
        ).T  # TODO(schneiderfelipe): use xy plane?
        Z = _linkage(_pdist(criteria), method="single")
        clusters = _fcluster(Z, thresh, criterion="distance")

        # TODO(schneiderfelipe): this is for debug and should evtl. be removed.
        if plot:
            import matplotlib.pyplot as plt

            plt.clf()
            for cluster in _np.unique(clusters):
                plt.scatter(
                    criteria[clusters == cluster, 0], criteria[clusters == cluster, 1]
                )
            for i, (atommass, _) in enumerate(zip(atommasses, clusters)):
                plt.annotate(atommass, (criteria[i, 0], criteria[i, 1]))
            plt.xlabel("omega")
            plt.ylabel("sigma")
            plt.show()

        for mass in _np.unique(atommasses):
            mass_condition = atommasses == mass
            for cluster in _np.unique(clusters[mass_condition]):
                groups = _update_groups_with_condition(
                    mass_condition & (clusters == cluster), groups
                )
    elif method == "atommass":
        for mass in _np.unique(atommasses):
            groups = _update_groups_with_condition(atommasses == mass, groups)
    else:
        raise ValueError(f"unavailable method: '{method}'")

    return sorted(groups, key=len)
