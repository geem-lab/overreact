#!/usr/bin/env python3  # noqa: EXE001

"""Module for storing constant data such as physical fundamental constants.

Most of this data comes from `scipy.constants`.
"""

import numpy as np
from scipy.constants import (
    N_A,
    R,
    angstrom,
    atm,
    atomic_mass,
    bar,
    c,
    calorie,
    centi,
    eV,
    giga,
    h,
    hbar,
    k,
    kilo,
    liter,
    physical_constants,
    torr,
)

__all__ = [
    "N_A",
    "R",
    "angstrom",
    "atm",
    "atomic_mass",
    "bar",
    "c",
    "eV",
    "giga",
    "h",
    "hbar",
    "k",
    "liter",
    "torr",
]

hartree, _, _ = physical_constants["Hartree energy"]
bohr, _, _ = physical_constants["Bohr radius"]
kcal = kilo * calorie

# W. Haynes. CRC Handbook of Chemistry and Physics. 100 Key Points.
# CRC Press, London, 95th edition, 2014. ISBN 9781482208689.
_vdw_radius_crc = [
    110.0,
    140.0,
    182.0,
    153.0,
    192.0,
    170.0,
    155.0,
    152.0,
    147.0,
    154.0,
    227.0,
    173.0,
    184.0,
    210.0,
    180.0,
    180.0,
    175.0,
    188.0,
    275.0,
    231.0,
    215.0,
    211.0,
    207.0,
    206.0,
    205.0,
    204.0,
    200.0,
    197.0,
    196.0,
    201.0,
    187.0,
    211.0,
    185.0,
    190.0,
    185.0,
    202.0,
    303.0,
    249.0,
    232.0,
    223.0,
    218.0,
    217.0,
    216.0,
    213.0,
    210.0,
    210.0,
    211.0,
    218.0,
    193.0,
    217.0,
    206.0,
    206.0,
    198.0,
    216.0,
    343.0,
    268.0,
    243.0,
    242.0,
    240.0,
    239.0,
    238.0,
    236.0,
    235.0,
    234.0,
    233.0,
    231.0,
    230.0,
    229.0,
    227.0,
    226.0,
    224.0,
    223.0,
    222.0,
    218.0,
    216.0,
    216.0,
    213.0,
    213.0,
    214.0,
    223.0,
    196.0,
    202.0,
    207.0,
    197.0,
    202.0,
    220.0,
    348.0,
    283.0,
    247.0,
    245.0,
    243.0,
    241.0,
    239.0,
    243.0,
    244.0,
    245.0,
    244.0,
    245.0,
    245.0,
    245.0,
    246.0,
    246.0,
    246.0,
    None,
    None,
    None,
    None,
    None,
    None,
    None,
    None,
    None,
    None,
    None,
    None,
    None,
    None,
    None,
]


@np.vectorize
def _vdw_radius(atomno):
    """Select reasonable estimates for the van der Waals radii.

    This is a helper for `vdw_radius`.
    """
    radius = _vdw_radius_crc[atomno - 1]
    if radius is None:
        return 2.0
    return radius * centi


def vdw_radius(atomno):
    """Select reasonable estimates for the van der Waals radii.

    This function returns van der Waals radii as recommended in the 95th
    edition of the "CRC Handbook of Chemistry and Physics" (2014). This
    consists of Bondi radii (A. Bondi. "van der Waals Volumes and Radii". J.
    Phys. Chem. 1964, 68, 3, 441-451. doi:10.1021/j100785a001) together with
    the values recommended by Truhlar (M. Mantina et al. "Consistent van der
    Waals Radii for the Whole Main Group". J. Phys. Chem. A 2009, 113, 19,
    5806-5812. doi:10.1021/jp8111556). For hydrogen, the value recommended by
    Taylor is employed (R. Rowland et al. "Intermolecular Nonbonded Contact
    Distances in Organic Crystal Structures: Comparison with Distances Expected
    from van der Waals Radii". J. Phys. Chem. 1996, 100, 18, 7384-7391.
    doi:10.1021/jp953141+). Other elements receive values recommended by either
    Hu (S.-Z., Hu. Kristallogr. 224, 375, 2009) or Guzei (Guzei, I. A. and
    Wendt, M., Dalton Trans., 2006, 3991, 2006). If neither are defined, we use
    2.0 Å as default.

    Parameters
    ----------
    atomno : array-like

    Returns
    -------
    array-like

    Examples
    --------
    >>> vdw_radius(1)  # H
    array(1.1)
    >>> vdw_radius(35)  # Br
    array(1.85)
    >>> vdw_radius([1, 2, 3, 4, 5])  # H, He, Li, Be, B
    array([1.1 , 1.4 , 1.82, 1.53, 1.92])
    >>> vdw_radius(range(1, 11))  # H, He, Li, Be, B, C, N, O, F, Ne
    array([1.1 , 1.4 , 1.82, 1.53, 1.92, 1.7 , 1.55, 1.52, 1.47, 1.54])
    >>> vdw_radius(range(21, 31))  # Sc, Ti, V, ..., Ni, Cu, Zn
    array([2.15, 2.11, 2.07, 2.06, 2.05, 2.04, 2.  , 1.97, 1.96, 2.01])
    """
    return _vdw_radius(atomno)
