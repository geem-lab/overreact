#!/usr/bin/env python3

"""Module dedicated to the calculation of thermodynamic properties."""

import numpy as _np
from scipy.constants import atm as _atm
from scipy.constants import h as _h
from scipy.constants import k as _k
from scipy.constants import liter as _liter
from scipy.constants import R as _R

from overreact.misc import molar_volume


# TODO(schneiderfelipe): this is the only contribution for the atomic inner
# energy normally considered. In the future I would like to implement an
# electronic correction for excited states, which might be useful for atoms
# with low lying excitation states such as fluorine, specially larger
# temperatures (second term of eq. 5-17 in Statistical Thermodynamics,
# McQuarrie).
def translational_energy(temperature=298.15):
    r"""Calculate the translational energy of an ideal gas.

    Take a look at <https://socratic.org/questions/5715a4e711ef6b17257e0033>.

    Parameters
    ----------
    temperature : float
        Absolute temperature in Kelvin.

    Returns
    -------
    float
        Translational energy in J/mol.

    Examples
    --------
    >>> translational_energy()
    3718.
    >>> translational_energy(373.15)
    4653.

    """
    return 1.5 * _R * temperature


# TODO(schneiderfelipe): this rotational contribution encompasses the high
# temperature limit. Higher order terms are possible, although important only
# at higher temperatures. See eq. 6-48 of Statistical Thermodynamics,
# McQuarrie.
#
# TODO(schneiderfelipe): validate values using Table 8-1 of Statistical
# Thermodynamics, McQuarrie.
def rotational_energy(moments_of_inertia=None, temperature=298.15):
    r"""Calculate the rotational energy of an ideal gas.

    Take a look at <https://socratic.org/questions/5715a4e711ef6b17257e0033>.

    Parameters
    ----------
    moments_of_inertia : array-like, optional
    temperature : array-like, optional

    Returns
    -------
    float
        Rotational energy in J/mol.

    Examples
    --------
    >>> rotational_energy()
    2479.

    """
    if moments_of_inertia is not None:
        return _np.count_nonzero(moments_of_inertia) * _R * temperature / 2.0
    return _R * temperature  # assume linear molecule


# TODO(schneiderfelipe): validate values using Table 8-1 of Statistical
# Thermodynamics, McQuarrie.
def vibrational_energy(nu, temperature=298.15):
    r"""Calculate the vibrational energy of an ideal gas.

    Take a look at <https://socratic.org/questions/5715a4e711ef6b17257e0033>.

    This includes the zero point energy (ZPE).

    Parameters
    ----------
    nu : array-like
        Frequency magnitudes.
    temperature : array-like, optional

    Returns
    -------
    float
        Vibrational energy in J/mol.

    Examples
    --------
    >>> import numpy as np
    >>> from scipy.constants import k, h
    >>> vibrational_energy(3374 * k / h)  # nitrogen molecule
    14026.
    >>>
    >>> vibrational_energy(np.array([3360, 1890, 954, 954]) * k / h)  # CO2
    3.045e4

    """
    vibrational_temperature = _h * nu / _k
    return _R * _np.sum(
        vibrational_temperature / 2.0
        + vibrational_temperature
        / (_np.exp(vibrational_temperature / temperature) - 1.0)
    )
    # TODO(schneiderfelipe): expression like the above could benefit from
    # <https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.expit.html>.


def get_delta_freeenergy(transform, freeenergy):
    """Calculate reaction free energies.

    Reaction free energies are calculated from matrices and absolute free
    energies.

    Parameters
    ----------
    transform : array-like
    freeenergy : array-like

    Returns
    -------
    delta_freeenergy : array-like

    Examples
    --------
    >>> get_delta_freeenergy([-1, 1], [-10, 10])
    20

    """
    return _np.asanyarray(transform).T @ _np.asanyarray(freeenergy)


def equilibrium_constant(
    delta_freeenergy,
    delta_n=0,
    temperature=298.15,
    pressure=_atm,
    volume=None,
    volume_factor=1.0,
    R=_R,
):
    r"""Calculate an equilibrium constant from a reaction Gibbs energy.

    This function uses the usual `relationship between reaction Gibbs energy
    and equilibrium constant
    <https://en.wikipedia.org/wiki/Equilibrium_constant>`_:

    .. math::
        K(T) = \exp\left(-\frac{\Delta_\text{r} G^\circ}{R T}\right)

    Parameters
    ----------
    delta_freeenergy : array-like
    delta_n : array-like, optional
        Multiply the end result by
        :math:`\left( \frac{p}{R T} \right)^{\Delta n}` (for one atmosphere and
        chosen temperature), which effectively calculates a solution
        equilibrium constant for gas phase input data.
    temperature : array-like, optional
    pressure : array-like, optional
        Reference gas pressure.
    volume : float, optional
        Molar volume.
    volume_factor : float or callable, optional
        Value that multiplies the molar volume outside the exponential term.
        If, callable the function receives the temperature as parameter (which
        means it should accept an array-like).
    R : float, optional
        Gas constant.

    Returns
    -------
    K : array-like

    Examples
    --------
    This first example is from Wikipedia
    (<https://en.wikipedia.org/wiki/Stability_constants_of_complexes#The_chelate_effect>):

    >>> import numpy as np
    >>> np.log10(equilibrium_constant(-60670.0))
    10.62

    A :math:`K_p` (gas phase), followed by its :math:`K` (solution):

    >>> equilibrium_constant(64187.263215698644, temperature=745.0)
    3.16e-5
    >>> equilibrium_constant(64187.263215698644, delta_n=-2, temperature=745.0)
    0.118e-6

    The following reproduces some data from doi:10.1021/ic202081z:

    >>> from scipy.constants import kilo
    >>> delta_freeenergy = kilo * (np.array([25.2, 15.4, -19.4, -24.3, 40.7])
    ...                       + np.array([27.0, 11.6, -16.5, -32.9, 38.6])) / 2
    >>> equilibrium_constant(delta_freeenergy, temperature=298.0)
    array([2.7e-5, 4.3e-3, 1.4e3, 1.0e5, 1.1e-7])

    THE PAPER COULD NOT WELL APPROXIMATE EXPERIMENTAL DATA. THIS MIGHT BE DUE
    TO LACK OF SYMMETRY OR OTHER FACTORS BEING USED.

    """
    temperature = _np.asanyarray(temperature)

    if volume is None:
        volume = molar_volume(temperature, pressure)
    if callable(volume_factor):
        volume_factor = volume_factor(temperature)

    order = 1 - _np.asanyarray(delta_n)
    return _np.exp(-_np.asanyarray(delta_freeenergy) / (R * temperature)) * (
        volume * volume_factor
    ) ** (order - 1)


def change_reference_state(
    new_reference=1.0 / _liter,
    old_reference=None,
    sign=1.0,
    temperature=298.15,
    pressure=_atm,
    volume=None,
    volume_factor=1.0,
    R=_R,
):
    r"""Calculate an aditive correction to a change in reference states.

    .. math::
        \Delta G_\text{corr} =
            R T \ln \left( \frac{\chi_\text{new}}{\chi_\text{old}} \right)

    The value returned can be directly summed to old reference free energies
    to obtain new reference free energies. See notes below.

    For instance, the concentration correction to Gibbs free energy for gas to
    liquid standard state change is simply
    (:math:`c^\circ = \frac{\text{1 atm}}{R T}`),

    .. math::
        \Delta G_\text{conc} =
            R T \ln \left( \frac{\text{1 M}}{c^\circ} \right)

    Parameters
    ----------
    new_reference : array-like, optional
        New reference state. Default value corresponds to 1 mol/liter.
    old_reference : array-like, optional
        Old reference state. Default value corresponds to the concentration of
        an ideal gas at the given temperature and 1 atm.
    sign : float, optional
    temperature : array-like, optional
    pressure : array-like, optional
        Reference gas pressure.
    volume : float, optional
        Molar volume.
    volume_factor : float or callable, optional
        Value that multiplies the molar volume outside the exponential term.
        If, callable the function receives the temperature as parameter (which
        means it should accept an array-like).
    R : float, optional
        Gas constant.

    Returns
    -------
    correction : array-like

    Notes
    -----
    This function can be used to add any energy correction in the form above.
    The only drawback is that, sometimes, those corrections are written with a
    minus sign in front of them (this implies switching the roles of
    `old_reference` and `new_reference`). The easiest way to accomplish this is
    by using ``sign=-1``.

    Examples
    --------
    By default, the correction returns a change in concentration from the gas
    phase standard concentration to the solvated state standard concentration:

    >>> from scipy.constants import kilo, calorie
    >>> change_reference_state() / (kilo * calorie)
    1.89
    >>> change_reference_state(temperature=273.15) / (kilo * calorie)
    1.69

    This function can also be used to adjust symmetry effects from C1
    calculations (symmetry number equals to one). For D7h, for instance, the
    symmetry number is 14:

    >>> change_reference_state(14, 1, sign=-1) / (kilo * calorie)
    -1.56

    """
    temperature = _np.asanyarray(temperature)

    if old_reference is None:
        if volume is None:
            volume = molar_volume(temperature, pressure)
        if callable(volume_factor):
            volume_factor = volume_factor(temperature)
        old_reference = 1.0 / (volume * volume_factor)
    return sign * R * temperature * _np.log(new_reference / old_reference)
