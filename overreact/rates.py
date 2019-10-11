#!/usr/bin/env python3

"""Module dedicated to the calculation of reaction rate constants."""

import numpy as _np
from scipy.constants import atm as _atm
from scipy.constants import h as _h
from scipy.constants import k as _k
from scipy.constants import kilo as _kilo
from scipy.constants import N_A as _N_A
from scipy.constants import R as _R
from scipy.constants import torr as _torr

from overreact import misc as _misc
from overreact import thermo as _thermo

# Solution inspired by
# <https://github.com/cclib/cclib/blob/87abf82c6a06836a2e5fb95a64cdf376c5ef8d4f/cclib/bridge/cclib2psi4.py#L10-L19>
_found_thermo = _misc._find_package("thermo")
if _found_thermo:
    from thermo.chemical import Chemical as _Chemical


@_np.vectorize
def liquid_viscosity(id, temperature=298.15, pressure=_atm):
    """Dynamic viscosity of a solvent.

    This function uses the `thermo` package.

    Parameters
    ----------
    id : str,
    temperature : array-like, optional
    pressure : array-like, optional

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
    _misc._check_package("thermo", _found_thermo)
    return _Chemical(id, temperature, pressure).mul


def smoluchowski(
    radii,
    viscosity=None,
    reactive_radius=None,
    temperature=298.15,
    pressure=_atm,
    mutual_diff_coef=None,
):
    """Calculate irreversible diffusion-controlled reaction rate constant.

    PRETTY MUCH EVERYTHING HERE IS OPTIONAL!

    Parameters
    ----------
    radii : array-like, optional
    viscosity : float, str or callable, optional
    reactive_radius : float, optional
    temperature : array-like, optional
    pressure : array-like, optional
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
    >>> import numpy as np
    >>> from scipy.constants import angstrom, liter
    >>> radii = np.array([2.59, 2.71]) * angstrom
    >>> smoluchowski(radii, reactive_radius=2.6 * angstrom,
    ...              viscosity=8.91e-4) / liter
    3.6e9
    >>> smoluchowski(radii, "water", reactive_radius=2.6 * angstrom) / liter
    3.6e9
    >>> smoluchowski(radii, viscosity=8.91e-4) / liter  # doctest: +SKIP
    3.6e9
    """
    radii = _np.asanyarray(radii)
    temperature = _np.asanyarray(temperature)
    pressure = _np.asanyarray(pressure)  # TODO(schneiderfelipe): do we need?

    if mutual_diff_coef is None:
        if callable(viscosity):
            viscosity = viscosity(temperature)
        elif isinstance(viscosity, str):
            viscosity = liquid_viscosity(viscosity, temperature, pressure)
        mutual_diff_coef = (
            _k * temperature / (6.0 * _np.pi * _np.asanyarray(viscosity))
        ) * _np.sum(1.0 / radii)

    if reactive_radius is None:
        reactive_radius = _np.sum(radii)

    return 4.0 * _np.pi * mutual_diff_coef * reactive_radius * _N_A


def collins_kimball(k_tst, k_diff):
    """Calculate reaction rate constant inclusing diffusion effects.

    This implementation is based on doi:10.1016/0095-8522(49)90023-9.

    Examples
    --------
    >>> collins_kimball(2.3e7, 3.6e9)
    2.3e7
    """
    return k_tst * k_diff / (k_tst + k_diff)


# TODO(schneiderfelipe): invert order between new_scale and old_scale
def convert_rate_constant(
    val, new_scale, old_scale, delta_n=0, temperature=298.15, pressure=_atm
):
    """Convert a reaction rate constant between common units.

    Parameters
    ----------
    val : array-like
        Rate constant to convert.
    new_scale, old_scale : str
        New and old units. Possible values are "cm3 mol-1 s-1", "l mol-1 s-1",
        "m3 mol-1 s-1", "cm3 particle-1 s-1", "mmHg-1 s-1" and "atm-1 s-1".
    delta_n : array-like, optional
        One minus reaction order.
    temperature : array-like, optional
    pressure : array-like, optional

    Returns
    -------
    array-like

    Examples
    --------
    >>> convert_rate_constant(1e3, "l mol-1 s-1", "atm-1 s-1", delta_n=-1,
    ...                       temperature=273.15)
    22414.
    """
    # TODO(schneiderfelipe): accept case-insensitive units
    if old_scale == new_scale:
        return 1.0

    # we first convert anything to l mol-1 s-1
    if old_scale == "cm3 mol-1 s-1":  # TODO(schneiderfelipe): accept ml mol-1 s-1
        factor = 1.0 / _kilo
    elif old_scale == "l mol-1 s-1":  # TODO(schneiderfelipe): accept m-1 s-1
        factor = 1.0
    elif old_scale == "m3 mol-1 s-1":
        factor = _kilo
    elif (
        old_scale == "cm3 particle-1 s-1"
    ):  # TODO(schneiderfelipe): accept ml particle-1 s-1
        factor = _N_A / _kilo
    elif old_scale == "mmHg-1 s-1":  # TODO(schneiderfelipe): accept torr-1 s-1
        factor = _misc.molar_volume(temperature) * _atm * _kilo / _torr
    elif old_scale == "atm-1 s-1":
        factor = _misc.molar_volume(temperature) * _kilo
    else:
        raise ValueError(f"unit not recognized: {old_scale}")

    # now we convert l mol-1 s-1 to anything
    if new_scale == "cm3 mol-1 s-1":
        factor *= _kilo
    elif new_scale == "l mol-1 s-1":
        factor *= 1.0
    elif new_scale == "m3 mol-1 s-1":
        factor *= 1.0 / _kilo
    elif new_scale == "cm3 particle-1 s-1":
        factor *= _kilo / _N_A
    elif new_scale == "mmHg-1 s-1":
        factor *= _torr / (_misc.molar_volume(temperature) * _atm * _kilo)
    elif new_scale == "atm-1 s-1":
        factor *= 1.0 / (_misc.molar_volume(temperature) * _kilo)
    else:
        raise ValueError(f"unit not recognized: {new_scale}")

    order = 1 - _np.asanyarray(delta_n)
    return val * factor ** (order - 1)


def eyring(
    delta_freeenergy,
    delta_n=0,
    temperature=298.15,
    pressure=_atm,
    volume=None,
    volume_factor=1.0,
    R=_R,
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
    delta_n : array-like, optional
        Multiply the end result by
        :math:`\left( \frac{p}{R T} \right)^{\Delta n}` (for one atmosphere and
        chosen temperature). See `equilibrium_constant` for details.
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
    k : array-like

    Notes
    -----
    TODO(schneiderfelipe): the comments below are not current anymore, as we
    now have a "convert_rate_constant" function. So we have now a very powerful
    and controllable way of producing reaction rate constants in any unit. We
    need to use this and make "volume_factor" obsolete.

    You may need some conversion factors to convert rate constants on the fly.
    There is the `volume_factor` parameter that makes it easy to multiply a
    single factor that is independent of molecularity. See doi:10.1021/ed046p54
    for a very useful reference table on conversion factors.

    Three conversion factors are extra useful:

    >>> from scipy.constants import N_A, R, atm, centi
    >>> si2cm3 = 1.0 / (N_A * centi**3)
    >>> si2cm3  # m3 mol-1 s-1 (SI) to cm3 particle-1 s-1
    0.1660e-17
    >>> cc2mol = lambda temperature: temperature * R / atm
    >>> cc2mol(1.0)  # atm-1 s-1 to m3 mol-1 s-1 (SI) depends on temperature
    8.21e-5
    >>> cc2cm3 = lambda temperature: temperature * R / (atm * centi ** 3 * N_A)
    >>> cc2cm3(1.0)  # atm-1 s-1 to cm3 particle-1 s-1 depends on temperature
    13.63e-23

    Examples
    --------
    >>> import numpy as np
    >>> import matplotlib.pyplot as plt
    >>> from scipy.constants import kilo, calorie, N_A
    >>> from overreact import thermo

    >>> eyring(17.26 * kilo * calorie)  # unimolecular, s-1
    1.38
    >>> eyring(18.86 * kilo * calorie)  # unimolecular, s-1
    0.093

    >>> temperatures = np.array([200, 298.15, 300, 400])
    >>> delta_freeenergy = np.array([8.0, 10.3, 10.3, 12.6])
    >>> delta_freeenergy -= thermo.change_reference_state(
    ...     temperature=temperatures) / (kilo * calorie)  # 1 atm to 1 M
    >>> delta_freeenergy
    array([6.9, 8.4, 8.4, 9.9])
    >>> delta_freeenergy += thermo.change_reference_state(4, 1, sign=-1,
    ...     temperature=temperatures) / (kilo * calorie)  # 4-fold symmetry TS
    >>> delta_freeenergy
    array([6.3, 7.6, 7.6, 8.8])
    >>> k = eyring(delta_freeenergy * kilo * calorie, temperature=temperatures,
    ...            volume_factor=cc2cm3, delta_n=-1)  # bimolecular
    >>> np.log10(k)  # doctest: +SKIP
    array([-15.65757732, -14.12493874, -14.10237291, -13.25181197])

    >>> from scipy.constants import c
    >>> from overreact.tunnel import wigner, eckart
    >>> kappa_wigner = wigner(1218 * c / centi, temperature=temperatures)
    >>> kappa_wigner
    array([4.2, 2.4, 2.4, 1.8])
    >>> kappa_eckart = eckart(1218 * c / centi, 13672.624, 24527.729644,
    ...                       temperature=temperatures)
    >>> kappa_eckart
    array([17.1, 4.0, 3.9, 2.3])

    THE PLOT BELOW SHOWS THAT OUR ERRORS ARE LARGER FOR LARGER TEMPERATURES.
    OUR CALCULATIONS SEEM TO GUESS REACTIONS FASTER THAN THEY MIGHT BE.
    I BELIEVE THIS IS DUE TO SOME PRE-FACTOR IN THE CODE THAT MULTIPLIES THE
    TEMPERATURE. THIS CAN EITHER BE THE k T / h TERM OR THE CODE FOR MOLAR
    VOLUME. I NEED TO FURTHER TEST BOTH PIECES.
    IF THIS IS THE k T / h TERM, WE MIGHT BE HAVING A PROBLEM OF SLIGHT
    DIFFERENCES IN CONSTANTS BEING "OVERESTIMATED" AS TEMPERATURES INCREASE.
    IF THIS IS DUE TO THE MOLAR VOLUME CODE, WE MIGHT BE FACING A STANDARD
    PRESSURE DISCREPANCY: EyringPy MIGHT BE WRONGLY USING 1 bar INSTEAD OF
    1 atm AND I NEED TO INVESTIGATE THAT.

    >>> plt.clf()
    >>> plt.plot(1000 / temperatures, np.log10(4.0 * k), "ro--", label="TST")
    [<matplotlib.lines.Line2D object at 0...>]
    >>> plt.plot(1000 / temperatures, np.log10(4.0 * k * kappa_wigner), "bo--",
    ...          label="TST+W")
    [<matplotlib.lines.Line2D object at 0x...>]
    >>> plt.plot(1000 / temperatures, np.log10(4.0 * k * kappa_eckart), "go--",
    ...          label="TST+E")
    [<matplotlib.lines.Line2D object at 0x...>]
    >>> plt.xlim(3.0, 5.5)
    (3.0, 5.5)
    >>> plt.ylim(-15.5, -12.5)
    (-15.5, -12.5)
    >>> plt.legend()
    <matplotlib.legend.Legend object at 0x...>
    >>> # plt.show()

    """
    temperature = _np.asanyarray(temperature)
    delta_freeenergy = _np.asanyarray(delta_freeenergy)
    return (
        _thermo.equilibrium_constant(
            delta_freeenergy, delta_n, temperature, pressure, volume, volume_factor, R
        )
        * _k
        * temperature
        / _h
    )
