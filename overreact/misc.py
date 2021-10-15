#!/usr/bin/env python3

"""Miscellaneous functions that do not currently fit in other modules.

Ideally, the functions here will be transferred to other modules in the future.
"""

from functools import lru_cache as cache

import numpy as np
from scipy.stats import cauchy
from scipy.stats import norm

import overreact as rx
from overreact import constants


def _find_package(package):
    """Check if a package exists without importing it.

    Inspired by
    <https://github.com/cclib/cclib/blob/87abf82c6a06836a2e5fb95a64cdf376c5ef8d4f/cclib/parser/utils.py#L35-L46>.

    Parameters
    ----------
    package : str

    Returns
    -------
    bool

    Examples
    --------
    >>> _find_package("overreact")
    True
    >>> _find_package("a_package_that_does_not_exist")
    False
    """
    import importlib

    module_spec = importlib.util.find_spec(package)
    return module_spec is not None and module_spec.loader is not None


_found_jax = _find_package("jax")
_found_seaborn = _find_package("seaborn")
_found_thermo = _find_package("thermo")
if _found_thermo:
    from thermo.chemical import Chemical

# Inspired by
# https://github.com/cclib/cclib/blob/master/cclib/parser/utils.py#L159
element = [
    None,
    "H",
    "He",
    "Li",
    "Be",
    "B",
    "C",
    "N",
    "O",
    "F",
    "Ne",
    "Na",
    "Mg",
    "Al",
    "Si",
    "P",
    "S",
    "Cl",
    "Ar",
    "K",
    "Ca",
    "Sc",
    "Ti",
    "V",
    "Cr",
    "Mn",
    "Fe",
    "Co",
    "Ni",
    "Cu",
    "Zn",
    "Ga",
    "Ge",
    "As",
    "Se",
    "Br",
    "Kr",
    "Rb",
    "Sr",
    "Y",
    "Zr",
    "Nb",
    "Mo",
    "Tc",
    "Ru",
    "Rh",
    "Pd",
    "Ag",
    "Cd",
    "In",
    "Sn",
    "Sb",
    "Te",
    "I",
    "Xe",
    "Cs",
    "Ba",
    "La",
    "Ce",
    "Pr",
    "Nd",
    "Pm",
    "Sm",
    "Eu",
    "Gd",
    "Tb",
    "Dy",
    "Ho",
    "Er",
    "Tm",
    "Yb",
    "Lu",
    "Hf",
    "Ta",
    "W",
    "Re",
    "Os",
    "Ir",
    "Pt",
    "Au",
    "Hg",
    "Tl",
    "Pb",
    "Bi",
    "Po",
    "At",
    "Rn",
    "Fr",
    "Ra",
    "Ac",
    "Th",
    "Pa",
    "U",
    "Np",
    "Pu",
    "Am",
    "Cm",
    "Bk",
    "Cf",
    "Es",
    "Fm",
    "Md",
    "No",
    "Lr",
    "Rf",
    "Db",
    "Sg",
    "Bh",
    "Hs",
    "Mt",
    "Ds",
    "Rg",
    "Cn",
    "Nh",
    "Fl",
    "Mc",
    "Lv",
    "Ts",
    "Og",
]

# Inspired by
# https://github.com/cclib/cclib/blob/master/cclib/parser/utils.py#L159
atomic_mass = [
    None,
    1.008,
    4.002602,
    6.94,
    9.0121831,
    10.81,
    12.011,
    14.007,
    15.999,
    18.998403163,
    20.1797,
    22.98976928,
    24.305,
    26.9815385,
    28.085,
    30.973761998,
    32.06,
    35.45,
    39.948,
    39.0983,
    40.078,
    44.955908,
    47.867,
    50.9415,
    51.9961,
    54.938044,
    55.845,
    58.933194,
    58.6934,
    63.546,
    65.38,
    69.723,
    72.63,
    74.921595,
    78.971,
    79.904,
    83.798,
    85.4678,
    87.62,
    88.90584,
    91.224,
    92.90637,
    95.95,
    97.90721,
    101.07,
    102.9055,
    106.42,
    107.8682,
    112.414,
    114.818,
    118.71,
    121.76,
    127.6,
    126.90447,
    131.293,
    132.90545196,
    137.327,
    138.90547,
    140.116,
    140.90766,
    144.242,
    144.91276,
    150.36,
    151.964,
    157.25,
    158.92535,
    162.5,
    164.93033,
    167.259,
    168.93422,
    173.045,
    174.9668,
    178.49,
    180.94788,
    183.84,
    186.207,
    190.23,
    192.217,
    195.084,
    196.966569,
    200.592,
    204.38,
    207.2,
    208.9804,
    209.0,
    210.0,
    222.0,
    223.0,
    226.0,
    227.0,
    232.0377,
    231.03588,
    238.02891,
    237.0,
    244.0,
    243.0,
    247.0,
    247.0,
    251.0,
    252.0,
    257.0,
    258.0,
    259.0,
    262.0,
    267.0,
    268.0,
    271.0,
    274.0,
    269.0,
    276.0,
    281.0,
    281.0,
    285.0,
    286.0,
    289.0,
    288.0,
    293.0,
    294.0,
    294.0,
]

# Inspired by
# https://github.com/cclib/cclib/blob/master/cclib/parser/utils.py#L159
atomic_number = {
    "H": 1,
    "He": 2,
    "Li": 3,
    "Be": 4,
    "B": 5,
    "C": 6,
    "N": 7,
    "O": 8,
    "F": 9,
    "Ne": 10,
    "Na": 11,
    "Mg": 12,
    "Al": 13,
    "Si": 14,
    "P": 15,
    "S": 16,
    "Cl": 17,
    "Ar": 18,
    "K": 19,
    "Ca": 20,
    "Sc": 21,
    "Ti": 22,
    "V": 23,
    "Cr": 24,
    "Mn": 25,
    "Fe": 26,
    "Co": 27,
    "Ni": 28,
    "Cu": 29,
    "Zn": 30,
    "Ga": 31,
    "Ge": 32,
    "As": 33,
    "Se": 34,
    "Br": 35,
    "Kr": 36,
    "Rb": 37,
    "Sr": 38,
    "Y": 39,
    "Zr": 40,
    "Nb": 41,
    "Mo": 42,
    "Tc": 43,
    "Ru": 44,
    "Rh": 45,
    "Pd": 46,
    "Ag": 47,
    "Cd": 48,
    "In": 49,
    "Sn": 50,
    "Sb": 51,
    "Te": 52,
    "I": 53,
    "Xe": 54,
    "Cs": 55,
    "Ba": 56,
    "La": 57,
    "Ce": 58,
    "Pr": 59,
    "Nd": 60,
    "Pm": 61,
    "Sm": 62,
    "Eu": 63,
    "Gd": 64,
    "Tb": 65,
    "Dy": 66,
    "Ho": 67,
    "Er": 68,
    "Tm": 69,
    "Yb": 70,
    "Lu": 71,
    "Hf": 72,
    "Ta": 73,
    "W": 74,
    "Re": 75,
    "Os": 76,
    "Ir": 77,
    "Pt": 78,
    "Au": 79,
    "Hg": 80,
    "Tl": 81,
    "Pb": 82,
    "Bi": 83,
    "Po": 84,
    "At": 85,
    "Rn": 86,
    "Fr": 87,
    "Ra": 88,
    "Ac": 89,
    "Th": 90,
    "Pa": 91,
    "U": 92,
    "Np": 93,
    "Pu": 94,
    "Am": 95,
    "Cm": 96,
    "Bk": 97,
    "Cf": 98,
    "Es": 99,
    "Fm": 100,
    "Md": 101,
    "No": 102,
    "Lr": 103,
    "Rf": 104,
    "Db": 105,
    "Sg": 106,
    "Bh": 107,
    "Hs": 108,
    "Mt": 109,
    "Ds": 110,
    "Rg": 111,
    "Cn": 112,
    "Nh": 113,
    "Fl": 114,
    "Mc": 115,
    "Lv": 116,
    "Ts": 117,
    "Og": 118,
}


def _check_package(package, found_package):
    """Raise an issue if a package was not found.

    Parameters
    ----------
    package : str
        Package name.
    found_package : bool
        Whether the package was found or not.

    Raises
    ------
    ImportError
        If the package could not be imported.

    Examples
    --------
    >>> _check_package("i_do_exist", True)
    >>> _check_package("i_dont_exist", False)
    Traceback (most recent call last):
      ...
    ImportError: You must install `i_dont_exist` to use this function
    """
    if not found_package:
        raise ImportError(f"You must install `{package}` to use this function")


# TODO(schneiderfelipe): what does this function returns for identifier="gas"
# or identifier="solvent"?
def _get_chemical(
    identifier, temperature=298.15, pressure=constants.atm, *args, **kwargs
):
    """Wrap `thermo.Chemical`.

    This function is used for obtaining property values and requires the
    `thermo` package.

    All parameters are passed to `thermo.Chemical` and the returned object is
    returned.

    Parameters
    ----------
    identifier : str
    temperature : array-like, optional
        Absolute temperature in Kelvin.
    pressure : array-like, optional
        Reference gas pressure.

    Examples
    --------
    >>> from overreact import constants
    >>> water = _get_chemical("water", pressure=constants.atm)
    >>> water.name
    'water'
    >>> water.Van_der_Waals_volume
    0.0
    >>> water.Vm
    1.806904e-5
    >>> water.permittivity
    78.355308122325
    >>> water.isobaric_expansion
    0.000402256
    >>> water.omega
    0.344
    >>> water.mul
    0.00091272
    """
    _check_package("thermo", _found_thermo)
    # TODO(schneiderfelipe): return a named tuple with only the required data.
    # TODO(schneiderfelipe): support logging the retrieval of data.
    # TODO(schneiderfelipe): test returned parameters.
    return Chemical(identifier, temperature, pressure, *args, **kwargs)


def broaden_spectrum(
    x, x0, y0, distribution="gaussian", scale=1.0, fit_points=True, *args, **kwargs
):
    """Broaden a point spectrum.

    Parameters
    ----------
    x : array-like
        Points where to return the spectrum.
    x0, y0 : array-like
        Spectrum to broaden as x, y points. Must have same shape.
    distribution : scipy.stats continuous distribution or `str`, optional
        An object from scipy stats. Strings "gaussian"/"norm" (default) and
        "cauchy"/"lorentzian" are also accepted.
    scale : float
        Scale parameter of distribution.
    fit_points : bool, optional
        Whether to fit the point spectrum, i.e., match maxima of y.

    Returns
    -------
    array-like
        Discretized continuum spectrum.

    Notes
    -----
    All other values are passed to the pdf method of the distribution.

    Examples
    --------
    >>> vibfreqs = np.array([81.44, 448.3, 573.57, 610.86, 700.53, 905.17,
    ...                      1048.41, 1114.78, 1266.59, 1400.68, 1483.76,
    ...                      1523.79, 1532.97, 1947.39, 3135.34, 3209.8,
    ...                      3259.29, 3863.13])  # infrared for acetic acid
    >>> vibirs = np.array([0.636676, 5.216484, 43.002425, 45.491292, 107.5175,
    ...                    3.292874, 41.673025, 13.081044, 213.36621,
    ...                    41.210458, 107.200119, 14.974489, 11.980532,
    ...                    342.170308, 0.532659, 1.875945, 2.625792,
    ...                    79.794631])  # associated intensities
    >>> x = np.linspace(vibfreqs.min() - 100.,
    ...                 vibfreqs.max() + 100., num=1000)
    >>> broaden_spectrum(x, vibfreqs, vibirs, scale=20.)  # broadened spectrum
    array([2.37570938e-006, 6.30824800e-006, 1.60981742e-005, 3.94817964e-005,
           9.30614047e-005, ..., 1.10015814e+002, ..., 3.42170308e+002, ...,
           4.94825527e-003, 2.01758488e-003, 7.90612998e-004, 2.97747760e-004])
    >>> broaden_spectrum(x, vibfreqs, vibirs, scale=20., fit_points=False)
    array([4.73279317e-008, 1.25670393e-007, 3.20701386e-007, 7.86540552e-007,
           1.85393207e-006, ..., 1.14581680e+000, ..., 6.81657998e+000, ...,
           9.85771618e-005, 4.01935188e-005, 1.57502758e-005, 5.93161175e-006])

    """
    if distribution in {"gaussian", "norm"}:
        distribution = norm
    elif distribution in {"lorentzian", "cauchy"}:
        distribution = cauchy

    s = np.sum(
        [
            yp * distribution.pdf(x, xp, scale=scale, *args, **kwargs)
            for xp, yp in zip(x0, y0)
        ],
        axis=0,
    )

    if fit_points:
        s_max = np.max(s)
        if s_max == 0.0:
            s_max = 1.0
        return s * np.max(y0) / s_max
    return s


# https://stackoverflow.com/a/10016613
def totuple(a):
    """Convert a numpy.array into nested tuples.

    Parameters
    ----------
    a : array-like

    Returns
    -------
    tuple

    Examples
    --------
    >>> array = np.array(((2,2),(2,-2)))
    >>> totuple(array)
    ((2, 2), (2, -2))
    """
    # we don't touch some types, and this includes namedtuples
    if isinstance(a, (int, float, str, rx.Scheme)):
        return a

    try:
        a = a.tolist()
    except AttributeError:
        pass

    try:
        return tuple(totuple(i) for i in a)
    except TypeError:
        return a


def halton(num, dim=None, jump=1, cranley_patterson=True):
    """Calculate Halton low-discrepancy sequences.

    Those sequences are good performers for Quasi-Monte Carlo numerical
    integration for dimensions up to around 6. A Cranley-Patterson rotation is
    applied by default. The origin is also jumped over by default.

    Parameters
    ----------
    num : int
    dim : int, optional
    jump : int, optional
    cranley_patterson : bool, optional

    Returns
    -------
    array-like

    Examples
    --------
    >>> halton(10, 3)  # random  # doctest: +SKIP
    array([[0.82232931, 0.38217312, 0.01170043],
           [0.57232931, 0.71550646, 0.21170043],
           [0.07232931, 0.1599509 , 0.41170043],
           [0.44732931, 0.49328423, 0.61170043],
           [0.94732931, 0.82661757, 0.85170043],
           [0.69732931, 0.27106201, 0.05170043],
           [0.19732931, 0.60439534, 0.25170043],
           [0.38482931, 0.93772868, 0.45170043],
           [0.88482931, 0.08587683, 0.65170043],
           [0.63482931, 0.41921016, 0.89170043]])
    >>> halton(10, 3, cranley_patterson=False)
    array([[0.5       , 0.33333333, 0.2       ],
           [0.25      , 0.66666667, 0.4       ],
           [0.75      , 0.11111111, 0.6       ],
           [0.125     , 0.44444444, 0.8       ],
           [0.625     , 0.77777778, 0.04      ],
           [0.375     , 0.22222222, 0.24      ],
           [0.875     , 0.55555556, 0.44      ],
           [0.0625    , 0.88888889, 0.64      ],
           [0.5625    , 0.03703704, 0.84      ],
           [0.3125    , 0.37037037, 0.08      ]])

    Cranley-Patterson rotations can improve Quasi-Monte Carlo integral
    estimates done with Halton sequences. Compare the following estimates of
    the integral of x between 0 and 1, which is exactly 0.5:

    >>> np.mean(halton(100, cranley_patterson=False))
    0.489921875
    >>> I = [np.mean(halton(100)) for i in range(1000)]
    >>> np.mean(I), np.var(I) < 0.00004
    (0.500, True)

    Now the integral of x**2 between 0 and 1, which is exactly 1/3:

    >>> np.mean(halton(100, cranley_patterson=False)**2)
    0.3222149658203125
    >>> I = [np.mean(halton(100)**2) for i in range(1000)]
    >>> np.mean(I), np.var(I) < 0.00004
    (0.333, True)

    >>> x = halton(1500)
    >>> np.mean(x)  # estimate of the integral of x between 0 and 1
    0.50
    >>> np.mean(x**2)  # estimate of the integral of x**2 between 0 and 1
    0.33
    """
    if dim is None:
        actual_dim = 1
    else:
        actual_dim = dim

    res = np.array(
        [
            [_vdc(i, b) for i in range(jump, jump + num)]
            for b in _first_primes(actual_dim)
        ]
    )

    if cranley_patterson:
        res = (res + np.random.rand(actual_dim, 1)) % 1.0
    if dim is None:
        return res.reshape((num,))
    return res.T


def _first_primes(size):
    """Help haltonspace.

    Examples
    --------
    >>> _first_primes(1)
    [2]
    >>> _first_primes(4)
    [2, 3, 5, 7]
    >>> _first_primes(10)
    [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]
    """

    def _is_prime(num):
        """Check if num is prime."""
        for i in range(2, int(np.sqrt(num)) + 1):
            if (num % i) == 0:
                return False
        return True

    primes = [2]
    p = 3
    while len(primes) < size:
        if _is_prime(p):
            primes.append(p)
        p += 2
    return primes


@cache(maxsize=1000000)
def _vdc(n, b=2):
    """Help haltonspace."""
    res, denom = 0, 1
    while n:
        denom *= b
        n, remainder = divmod(n, b)
        res += remainder / denom
    return res
