#!/usr/bin/env python3  # noqa: EXE001

"""Module dedicated to the calculation of reaction rate constants."""


from __future__ import annotations

__all__ = ["eyring"]


import logging
from typing import Optional, Union

import numpy as np

import overreact as rx
from overreact import _constants as constants

logger = logging.getLogger(__name__)


@np.vectorize
def liquid_viscosity(id, temperature=298.15, pressure=constants.atm):  # noqa: A002
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
    return rx._misc._get_chemical(id, temperature, pressure).mul  # noqa: SLF001


# TODO(schneiderfelipe): log the calculated diffusional reaction rate limit.
def smoluchowski(  # noqa: PLR0913
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
    This is a work in progress!

    TODO(schneiderfelipe): THERE ARE DOUBTS ABOUT HOW TO SELECT
    reactive_radius. doi:10.1002/jcc.23409 HELPS CLARIFY SOME ASPECTS BUT
    THERE'S STILL PROBLEMS. I BELIEVE THERE'S A RELATIONSHIP BETWEEN THE
    IMAGINARY FREQUENCY AND HOW FAR ATOMS MOVE CLOSE TO REACT, WHICH MIGHT
    GIVE SOME LIGHT. IN ANY CASE, I BELIEVE THAT THIS VALUE SHOULD BE LARGER
    THAN A CHARACTERISTIC DISTANCE IN THE TRANSITION STATE, SHOULD BE LARGER
    FOR LIGHTER GROUPS BEING TRANSFERRED (OR BETTER, ELECTRONS), BUT SHOULD BE
    SMALLER THAN A CHARACTERISTIC DISTANCE IN THE REACTIVE COMPLEX. THIS GIVES
    A RANGE TO START WORKING WITH.

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
    >>> smoluchowski(radii, viscosity=8.91e-4) / constants.liter
    3.7e9
    """
    radii = np.asarray(radii)
    temperature = np.asarray(temperature)

    if mutual_diff_coef is None:
        if callable(viscosity):
            viscosity = viscosity(temperature)
        elif isinstance(viscosity, str):
            viscosity = liquid_viscosity(viscosity, temperature, pressure)
        mutual_diff_coef = (
            constants.k * temperature / (6.0 * np.pi * np.asarray(viscosity))
        ) * np.sum(1.0 / radii)

    if reactive_radius is None:
        # NOTE(schneiderfelipe): not sure if I should divide by two here, but
        # it works. My guess is that there is some confusion between contact
        # distances (which are basically sums of two radii) and sums of pairs
        # of radii.
        reactive_radius = np.sum(radii) / 2

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


def convert_rate_constant(  # noqa: C901, PLR0912, PLR0913
    val,
    new_scale,
    old_scale="l mol-1 s-1",
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

    Raises
    ------
    ValueError
        If either `old_scale` or `new_scale` are not recognized.

    Notes
    -----
    Some symbols are accepted as alternatives in "new_scale" and "old_scale":
    "M-1", "ml" and "torr-1" are understood as "l mol-1", "cm3" and "mmHg-1",
    respectively.

    Examples
    --------
    >>> convert_rate_constant(1.0, "cm3 particle-1 s-1", "m3 mol-1 s-1",
    ...                       molecularity=2)
    0.1660e-17

    If `old_scale` is not given, it defaults to ``"l mol-1 s-1"``:

    >>> convert_rate_constant(1.0, "m3 mol-1 s-1", molecularity=2)
    0.001

    There are many options for `old_scale` and `new_scale`:

    >>> convert_rate_constant(1.0, "m3 mol-1 s-1", "atm-1 s-1",
    ...                       molecularity=2, temperature=1.0)
    8.21e-5
    >>> convert_rate_constant(1.0, "cm3 particle-1 s-1", "atm-1 s-1",
    ...                       molecularity=2, temperature=1.0)
    13.63e-23
    >>> convert_rate_constant(1e3, "l mol-1 s-1", "atm-1 s-1",
    ...                       molecularity=2, temperature=273.15)
    22414.

    If `old_scale` is the same as `new_scale`, or if the molecularity is one,
    the received value is returned:

    >>> convert_rate_constant(12345, "atm-1 s-1", "atm-1 s-1", molecularity=2)
    12345
    >>> convert_rate_constant(67890, "l mol-1 s-1", "atm-1 s-1", molecularity=1)
    67890

    Below are some examples regarding some accepted alternative symbols:

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
    for alt, ref in [("M-1", "l mol-1"), ("ml", "cm3"), ("torr-1", "mmHg-1")]:
        new_scale, old_scale = new_scale.replace(alt, ref), old_scale.replace(alt, ref)

    # no need to convert if same units or if molecularity is one
    if old_scale == new_scale or np.all(molecularity == 1):
        return val

    # we first convert to l mol-1 s-1
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
            rx.thermo.molar_volume(temperature, pressure)
            * pressure
            * constants.kilo
            / constants.torr
        )
    elif old_scale == "atm-1 s-1":
        factor = rx.thermo.molar_volume(temperature, pressure) * constants.kilo
    else:
        raise ValueError(f"old unit not recognized: {old_scale}")  # noqa: EM102, TRY003

    # now we convert l mol-1 s-1 to what we need
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
            rx.thermo.molar_volume(temperature, pressure) * pressure * constants.kilo
        )
    elif new_scale == "atm-1 s-1":
        factor *= 1.0 / (rx.thermo.molar_volume(temperature, pressure) * constants.kilo)
    else:
        raise ValueError(f"new unit not recognized: {new_scale}")  # noqa: EM102, TRY003

    factor **= molecularity - 1
    logger.info(
        f"conversion factor ({old_scale} to {new_scale}) = {factor}",  # noqa: G004
    )
    return val * factor


def eyring(
    delta_freeenergy: Union[float, np.ndarray],  # noqa: UP007
    molecularity: Optional[int] = None,  # noqa: UP007
    temperature: Union[float, np.ndarray] = 298.15,  # noqa: UP007
    pressure: float = constants.atm,
    volume: Optional[float] = None,  # noqa: UP007
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
        Delta Gibbs activation free energies. **This assumes values were already
        corrected for a one molar reference state (if applicable).**
    molecularity : array-like, optional
        Reaction order, i.e., number of molecules that come together to react.
        If set, this is used to calculate `delta_moles` for
        `equilibrium_constant`, which effectively calculates a solution
        equilibrium constant between reactants and the transition state for
        gas phase data. **You should set this to `None` if your free energies
        were already adjusted for solution Gibbs free energies.**
    temperature : array-like, optional
        Absolute temperature in Kelvin.
    pressure : array-like, optional
        Reference gas pressure.
    volume : float, optional
        Molar volume. This is passed on to `equilibrium_constant`.

    Returns
    -------
    k : array-like
        Reaction rate constant(s). By giving energies in one molar reference
        state, returned units are then accordingly given, e.g. "l mol-1 s-1"
        if second-order, etc.

    Notes
    -----
    This function uses `equilibrium_constant` internally to calculate the
    equilibrium constant between reactants and the transition state.

    Examples
    --------
    The following are examples from
    [Thermochemistry in Gaussian](https://gaussian.com/thermo/), in which the
    kinetic isotope effect of a bimolecular reaction is analyzed:

    >>> eyring(17.26 * constants.kcal)
    array([1.38])
    >>> eyring(18.86 * constants.kcal)
    array([0.093])

    It is well known that, at room temperature, if you "decrease" a reaction
    barrier by 1.4 kcal/mol, the reaction becomes around ten times faster:

    >>> dG = np.random.uniform(1.0, 100.0) * constants.kcal
    >>> eyring(dG - 1.4 * constants.kcal) / eyring(dG)
    array([10.])

    A similar relationship is found for a twofold increase in speed and a
    0.4 kcal/mol decrease in the reaction barrier (again, at room
    temperature):

    >>> eyring(dG - 0.4 * constants.kcal) / eyring(dG)
    array([2.0])

    """
    temperature = np.asarray(temperature)
    delta_freeenergy = np.asarray(delta_freeenergy)

    delta_moles = None
    if molecularity is not None:
        delta_moles = 1 - np.asarray(molecularity)

    return (
        rx.thermo.equilibrium_constant(
            delta_freeenergy,
            delta_moles,
            temperature,
            pressure,
            volume,
        )
        * constants.k
        * temperature
        / constants.h
    )
