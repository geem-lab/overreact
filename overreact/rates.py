#!/usr/bin/env python3

"""Module dedicated to the calculation of reaction rate constants."""

import numpy as np

from overreact import constants
from overreact import misc as misc
from overreact import _thermo


@np.vectorize
def liquid_viscosity(id, temperature=298.15, pressure=constants.atm):
    """Dynamic viscosity of a solvent.

    This function requires the `thermo` package for obtaining property values.

    Parameters
    ----------
    id : str,
    temperature : array-like, optional
        Absolute temperature in Kelvin.
    pressure : array-like, optional
        Reference gas pressure.

    Returns
    -------
    float
        Dynamic viscosity in SI units (Pa*s).

    Examples
    --------
    >>> liquid_viscosity("water", temperature=299.26)
    8.90e-4
    """
    # TODO(schneiderfelipe): test viscosities for the following solvents at
    # the following temperatures:
    # - water: 274 K -- 373 K
    # - pentane: 144 K -- 308 K
    # - hexane: 178 K -- 340 K
    # - heptane: 183 K -- 370 K
    # - octane: 217 K -- 398 K
    return misc._get_chemical(id, temperature, pressure).mul


def smoluchowski(
    radii,
    viscosity=None,
    reactive_radius=None,
    temperature=298.15,
    pressure=constants.atm,
    mutual_diff_coef=None,
):
    r"""Calculate irreversible diffusion-controlled reaction rate constant.

    PRETTY MUCH EVERYTHING HERE IS OPTIONAL!

    Parameters
    ----------
    radii : array-like, optional
    viscosity : float, str or callable, optional
    reactive_radius : float, optional
    temperature : array-like, optional
        Absolute temperature in Kelvin.
    pressure : array-like, optional
        Reference gas pressure.
    mutual_diff_coef : array-like, optional

    Returns
    -------
    float

    Notes
    -----
    TODO: THERE ARE DOUBTS ABOUT HOW TO SELECT reactive_radius.
    doi:10.1002/jcc.23409 HELPS CLARIFY SOME ASPECTS BUT THERE'S STILL
    PROBLEMS. I BELIEVE THERE'S A RELATIONSHIP BETWEEN THE IMAGINARY FREQUENCY
    AND HOW FAR ATOMS MOVE CLOSE TO REACT, WHICH MIGHT GIVE SOME LIGHT. IN ANY
    CASE, I BELIEVE THAT THIS VALUE SHOULD BE LARGER THAN A CHARACTERISTIC
    DISTANCE IN THE TRANSITION STATE, SHOULD BE LARGER FOR LIGHTER GROUPS BEING
    TRANSFERED (OR BETTER, ELECTRONS), BUT SHOULD BE SMALLER THAN A
    CHARACTERISTIC DISTANCE IN THE REACTIVE COMPLEX. THIS GIVES A RANGE TO
    START WORKING WITH.

    Below I delineate a temptive algorithm:
    1. superpose reactant A onto TS and RC
    2. superpose reactant B onto TS and RC
    3. identify the closest atoms of A/B in TS
    4. measure the distance of the closest atoms in RC

    Examples
    --------
    >>> radii = np.array([2.59, 2.71]) * constants.angstrom
    >>> smoluchowski(radii, reactive_radius=2.6 * constants.angstrom,
    ...              viscosity=8.91e-4) / constants.liter
    3.6e9
    >>> smoluchowski(radii, "water", reactive_radius=2.6 * constants.angstrom) \
    ...     / constants.liter
    3.6e9
    >>> smoluchowski(radii, viscosity=8.91e-4) / constants.liter  # doctest: +SKIP
    3.6e9
    """
    radii = np.asanyarray(radii)
    temperature = np.asanyarray(temperature)
    pressure = np.asanyarray(pressure)  # TODO(schneiderfelipe): do we need this?

    if mutual_diff_coef is None:
        if callable(viscosity):
            viscosity = viscosity(temperature)
        elif isinstance(viscosity, str):
            viscosity = liquid_viscosity(viscosity, temperature, pressure)
        mutual_diff_coef = (
            constants.k * temperature / (6.0 * np.pi * np.asanyarray(viscosity))
        ) * np.sum(1.0 / radii)

    if reactive_radius is None:
        reactive_radius = np.sum(radii)

    return 4.0 * np.pi * mutual_diff_coef * reactive_radius * constants.N_A


def collins_kimball(k_tst, k_diff):
    """Calculate reaction rate constant inclusing diffusion effects.

    This implementation is based on doi:10.1016/0095-8522(49)90023-9.

    Examples
    --------
    >>> collins_kimball(2.3e7, 3.6e9)
    2.3e7
    """
    return k_tst * k_diff / (k_tst + k_diff)


def convert_rate_constant(
    val,
    new_scale,
    old_scale="atm-1 s-1",
    molecularity=1,
    temperature=298.15,
    pressure=constants.atm,
):
    r"""Convert a reaction rate constant between common units.

    The reference paper used for developing this function is doi:10.1021/ed046p54.

    Parameters
    ----------
    val : array-like
        Rate constant to convert.
    new_scale : str
        New units. Possible values are "cm3 mol-1 s-1", "l mol-1 s-1",
        "m3 mol-1 s-1", "cm3 particle-1 s-1", "mmHg-1 s-1" and "atm-1 s-1".
    old_scale : str, optional
        Old units. Possible values are the same as for `new_scale`.
    molecularity : array-like, optional
        Reaction order, i.e., number of molecules that come together to react.
    temperature : array-like, optional
        Absolute temperature in Kelvin.
    pressure : array-like, optional
        Reference gas pressure.

    Returns
    -------
    array-like

    Notes
    -----
    Some symbols are accepted as alternatives in "new_scale" and "old_scale":
    "M-1", "ml" and "torr-1" are understood as "l mol-1", "cm3" and "mmHg-1",
    respectively.

    Examples
    --------
    >>> convert_rate_constant(1.0, "cm3 particle-1 s-1", "m3 mol-1 s-1", molecularity=2)
    0.1660e-17

    If `old_scale` is not given, it defaults to ``"atm-1 s-1"``:

    >>> convert_rate_constant(1.0, "m3 mol-1 s-1", molecularity=2, temperature=1.0)
    8.21e-5
    >>> convert_rate_constant(1.0, "cm3 particle-1 s-1", molecularity=2,
    ...                       temperature=1.0)
    13.63e-23
    >>> convert_rate_constant(1e3, "l mol-1 s-1", molecularity=2, temperature=273.15)
    22414.

    If `old_scale` is the same as `new_scale`, or if the molecularity is one,
    the received value is returned:

    >>> convert_rate_constant(1e3, "atm-1 s-1", "atm-1 s-1", molecularity=2)
    1e3
    >>> convert_rate_constant(1e3, "l mol-1 s-1", "atm-1 s-1", molecularity=1)
    1e3

    Below are some examples regarding the accepted alternative symbols:

    >>> convert_rate_constant(1.0, "M-1 s-1", molecularity=2) \
    ...     == convert_rate_constant(1.0, "l mol-1 s-1", molecularity=2)
    True
    >>> convert_rate_constant(1.0, "ml mol-1 s-1", molecularity=2) \
    ...     == convert_rate_constant(1.0, "cm3 mol-1 s-1", molecularity=2)
    True
    >>> convert_rate_constant(1.0, "torr-1 s-1", molecularity=2) \
    ...     == convert_rate_constant(1.0, "mmHg-1 s-1", molecularity=2)
    True
    """
    # new_scale, old_scale = new_scale.lower(), old_scale.lower()
    for alt, ref in [("M-1", "l mol-1"), ("ml", "cm3"), ("torr-1", "mmHg-1")]:
        new_scale, old_scale = new_scale.replace(alt, ref), old_scale.replace(alt, ref)

    # no need to convert if same units or if molecularity is one
    if old_scale == new_scale or np.all(molecularity == 1):
        # TODO(schneiderfelipe): probably do something else with array-like
        # molecularity?
        return val

    # we first convert anything to l mol-1 s-1
    if old_scale == "cm3 mol-1 s-1":
        factor = 1.0 / constants.kilo
    elif old_scale == "l mol-1 s-1":
        factor = 1.0
    elif old_scale == "m3 mol-1 s-1":
        factor = constants.kilo
    elif old_scale == "cm3 particle-1 s-1":
        factor = constants.N_A / constants.kilo
    elif old_scale == "mmHg-1 s-1":
        factor = (
            _thermo.molar_volume(temperature, pressure)
            * pressure
            * constants.kilo
            / constants.torr
        )
    elif old_scale == "atm-1 s-1":
        factor = _thermo.molar_volume(temperature, pressure) * constants.kilo
    else:
        raise ValueError(f"unit not recognized: {old_scale}")

    # now we convert l mol-1 s-1 to anything
    if new_scale == "cm3 mol-1 s-1":
        factor *= constants.kilo
    elif new_scale == "l mol-1 s-1":
        factor *= 1.0
    elif new_scale == "m3 mol-1 s-1":
        factor *= 1.0 / constants.kilo
    elif new_scale == "cm3 particle-1 s-1":
        factor *= constants.kilo / constants.N_A
    elif new_scale == "mmHg-1 s-1":
        factor *= constants.torr / (
            _thermo.molar_volume(temperature, pressure) * pressure * constants.kilo
        )
    elif new_scale == "atm-1 s-1":
        factor *= 1.0 / (_thermo.molar_volume(temperature, pressure) * constants.kilo)
    else:
        raise ValueError(f"unit not recognized: {new_scale}")

    return val * factor ** (molecularity - 1)


def eyring(
    delta_freeenergy,
    molecularity=1,
    temperature=298.15,
    pressure=constants.atm,
    volume=None,
):
    r"""Calculate a reaction rate constant.

    This function uses the `Eyring-Evans-Polanyi equation
    <https://en.wikipedia.org/wiki/Eyring_equation>`_ from `transition state
    theory <https://en.wikipedia.org/wiki/Transition_state_theory>`_:

    .. math::
        k(T) = \frac{k_\text{B} T}{h} K^\ddagger
             = \frac{k_\text{B} T}{h}
               \exp\left(-\frac{\Delta^\ddagger G^\circ}{R T}\right)

    where :math:`h` is Planck's constant, :math:`k_\text{B}` is Boltzmann's
    constant and :math:`T` is the absolute temperature.

    Parameters
    ----------
    delta_freeenergy : array-like
    molecularity : array-like, optional
        Reaction order, i.e., number of molecules that come together to react.
    temperature : array-like, optional
        Absolute temperature in Kelvin.
    pressure : array-like, optional
        Reference gas pressure.
    volume : float, optional
        Molar volume.

    Returns
    -------
    k : array-like
        Reaction rate constant(s). Units assume atm as the base for
        concentration units.

    Notes
    -----
    The end result is multiplied by :math:`\left( \frac{p}{R T} \right)^{\Delta n}` (for
    one atmosphere and chosen temperature), where the difference in moles is calculated
    from the reaction molecularity.

    Examples
    --------
    >>> eyring(17.26 * constants.kcal)  # unimolecular, s-1
    1.38
    >>> eyring(18.86 * constants.kcal)  # unimolecular, s-1
    0.093
    """
    temperature = np.asanyarray(temperature)
    delta_freeenergy = np.asanyarray(delta_freeenergy)
    delta_moles = 1 - molecularity
    return (
        _thermo.equilibrium_constant(
            delta_freeenergy, delta_moles, temperature, pressure, volume
        )
        * constants.k
        * temperature
        / constants.h
    )
