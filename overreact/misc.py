#!/usr/bin/env python3

"""Miscelaneous functions that do not currently fit in other modules.

Ideally, the functions here will be transfered to other modules in the future.
"""

import numpy as _np
from scipy.constants import atm as _atm
from scipy.constants import atomic_mass as _atomic_mass
from scipy.constants import h as _h
from scipy.constants import hbar as _hbar
from scipy.constants import k as _k
from scipy.constants import N_A as _N_A
from scipy.constants import R as _R
from scipy.stats import cauchy as _cauchy
from scipy.stats import norm as _norm


def _find_package(package):
    """Check if a package exists without importing it.

    Derived from
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
    ImportError if the package could not be imported.

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


# TODO(schneiderfelipe): I am in the position of testing all cases found in
# Table 5-3 of Statistical Thermodynamics, McQuarrie. All I need is the (spin)
# electronic and translational entropies.


# TODO(schneiderfelipe): in the future, there could be an implementation using
# a distribution of excited states. Currently, only the spin multiplicity of
# the ground state is considered. A starting point is (part of) the last two
# terms in eq. 5-19 of Statistical Thermodynamics, McQuarrie.
#
# Inspired by
# <https://github.com/eljost/thermoanalysis/blob/89b28941520fdeee1c96315b1900e124f094df49/thermoanalysis/thermo.py#L31>
def electronic_entropy(multiplicity):
    """Electronic entropy.

    Only the ground state is considered. See
    <https://en.wikipedia.org/wiki/Electronic_entropy>.

    Parameters
    ----------
    multiplicity : int
        Multiplicity of the molecule.

    Returns
    -------
    float
        Electronic entropy in J/(mol*K).

    Examples
    --------
    >>> electronic_entropy(1)
    0.0
    >>> electronic_entropy(2)
    5.763
    >>> electronic_entropy(3)
    9.134

    """
    return _R * _np.log(multiplicity)


# TODO(schneiderfelipe): this is only valid for diatomic molecules at high
# enough temperatures.
def vibrational_entropy(nu, temperature=298.15):
    r"""Calculate the vibrational entropy of an ideal gas.

    Take a look at <https://socratic.org/questions/5715a4e711ef6b17257e0033>.

    Parameters
    ----------
    nu : array-like
        Frequency magnitude.
    temperature : array-like, optional

    Returns
    -------
    float
        Vibrational entropy in J/(mol*K).

    Examples
    --------
    >>> from scipy.constants import k, h
    >>> vibrational_entropy(3374 * k / h)  # nitrogen molecule
    0.0012463

    """
    vibrational_temperature = _h * nu / _k
    energy_fraction = vibrational_temperature / temperature
    return _R * (
        (energy_fraction / (_np.exp(energy_fraction) - 1.0))
        - _np.log(1.0 - _np.exp(-energy_fraction))
    )


# TODO(schneiderfelipe): this is only valid for diatomic molecules at high
# enough temperatures.
#
# TODO(schneiderfelipe): we have Table 6-3, Statistical Thermodynamics,
# McQuarrie, to validate.
def rotational_entropy(moment_inertia, temperature=298.15, symmetry_number=1):
    r"""Calculate the rotational entropy of an ideal gas.

    Take a look at <https://socratic.org/questions/5715a4e711ef6b17257e0033>.

    Parameters
    ----------
    moment_inertia : float
        Molecular mass in atomic mass units (amu).
    temperature : float, optional
        Absolute temperature in Kelvin.
    symmetry_number : int, optional

    Returns
    -------
    float
        Rotational entropy in J/(mol*K).

    Examples
    --------
    >>> from scipy.constants import hbar, k
    >>> rotational_entropy(hbar**2 / (2.0 * k * 15.2))  # HCl
    33.06
    >>> rotational_entropy(hbar**2 / (2.0 * k * 87.6), symmetry_number=2)  # H2
    12.73

    """
    rotational_temperature = _hbar ** 2 / (2.0 * _k * moment_inertia)
    q_rot = temperature / (symmetry_number * rotational_temperature)
    return _R * (_np.log(q_rot) + 1.0)


# Inspired by
# <https://github.com/eljost/thermoanalysis/blob/89b28941520fdeee1c96315b1900e124f094df49/thermoanalysis/thermo.py#L74>
def sackur_tetrode(molecular_mass, temperature=298.15, pressure=_atm):
    r"""Calculate the translational entropy of an ideal gas.

    Take a look at <https://socratic.org/questions/5715a4e711ef6b17257e0033>.

    Parameters
    ----------
    molecular_mass : float
        Molecular mass in atomic mass units (amu).
    temperature : float
        Absolute temperature in Kelvin.
    pressure : float

    Returns
    -------
    float
        Translational entropy in J/(mol*K).

    Notes
    -----
    It seems that using 1 bar (1e5 Pa) instead of 1 atm (1.01325e5 Pa) better
    agrees with the results of Gaussian and ORCA.

    The formula used here follows
    <https://en.wikipedia.org/wiki/Sackur%E2%80%93Tetrode_equation#Derivation_from_information_theoretic_perspective>.

    Examples
    --------
    >>> sackur_tetrode(35.45)  # Cl-
    153.246

    >>> from scipy.constants import bar
    >>> sackur_tetrode(35.45, pressure=bar)  # Cl-
    153.356

    There is a simplified formula for this, which should (somewhat) agree
    with the one used here. They differ by less than 10 J/(mol*K):

    >>> import numpy as np
    >>> from scipy.constants import R
    >>> R * (3/2 * np.log(35.45) + 5/2 * np.log(298.15)) - 2.315  # check!
    160.617
    >>> R * (3/2 * np.log(35.45) + 5/2 * np.log(298.15)) - 2.315 \
    ... - sackur_tetrode(35.45, temperature=298.15) < 7.4
    True

    """
    molecular_mass = molecular_mass * _atomic_mass
    debroglie_wavelength = _h / _np.sqrt(
        2.0 * _np.pi * molecular_mass * _k * temperature
    )
    q_trans = molar_volume(temperature, pressure) / (_N_A * debroglie_wavelength ** 3)
    return _R * (_np.log(q_trans) + 2.5)


def molar_volume(temperature=298.15, pressure=_atm):
    """Ideal gas molar volume.

    Parameters
    ----------
    temperature : array-like, optional
    pressure : array-like, optional

    Returns
    -------
    float

    Examples
    --------
    >>> from scipy.constants import bar
    >>> molar_volume(temperature=273.15, pressure=bar)
    0.0227110
    """
    return _R * _np.asanyarray(temperature) / _np.asanyarray(pressure)


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
    >>> import numpy as np
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
        distribution = _norm
    elif distribution in {"lorentzian", "cauchy"}:
        distribution = _cauchy

    s = _np.sum(
        [
            yp * distribution.pdf(x, xp, scale=scale, *args, **kwargs)
            for xp, yp in zip(x0, y0)
        ],
        axis=0,
    )

    if fit_points:
        s_max = _np.max(s)
        if s_max == 0.0:
            s_max = 1.0
        return s * _np.max(y0) / s_max
    return s
