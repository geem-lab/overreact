#!/usr/bin/env python3  # noqa: EXE001

"""Module dedicated to quantum tunneling approximations."""


from __future__ import annotations

__all__ = ["eckart", "wigner"]


import logging
from typing import Optional, Union

import numpy as np
from scipy.integrate import fixed_quad
from scipy.special import roots_laguerre

from overreact import _constants as constants

logger = logging.getLogger(__name__)


def _check_nu(vibfreq: float) -> float:
    """Convert vibrational frequencies in cm$^{-1}$ to s-1.

    Parameters
    ----------
    vibfreq : array-like
        Magnitude of the imaginary frequency in cm$^{-1}$. Only the absolute value
        is used.

    Returns
    -------
    nu : array-like

    Raises
    ------
    ValueError
        If `vibfreq` is zero.

    Examples
    --------
    >>> vibfreq = 1000.0
    >>> _check_nu(vibfreq)
    2.99792458e13
    >>> _check_nu(2.0 * vibfreq) / _check_nu(vibfreq)
    2.0
    >>> _check_nu(vibfreq) == _check_nu(-vibfreq)
    True
    """
    if np.isclose(vibfreq, 0.0).any():
        raise ValueError(  # noqa: TRY003
            f"vibfreq should not be zero for tunneling: {vibfreq}",  # noqa: EM102
        )  # noqa: RUF100
    return np.abs(vibfreq) * constants.c / constants.centi


def wigner(
    vibfreq: float,
    temperature: Union[float, np.ndarray] = 298.15,  # noqa: UP007
) -> float:  # noqa: RUF100
    """Calculate the Wigner correction to quantum tunneling.

    Parameters
    ----------
    vibfreq : array-like
        Magnitude of the imaginary frequency in cm$^{-1}$. Only the absolute value
        is used.
    temperature : array-like, optional
        Absolute temperature in Kelvin.

    Returns
    -------
    kappa : array-like
        The quantum tunneling correction.

    Raises
    ------
    ValueError
        If `vibfreq` is zero.

    Examples
    --------
    >>> wigner(1821.0777)
    4.218
    >>> wigner(262.38)
    1.06680
    >>> wigner(190.5927)
    1.03525
    >>> wigner(169.14)
    1.02776
    >>> wigner(113.87)
    1.01258

    """
    temperature = np.asarray(temperature)

    nu = _check_nu(vibfreq)
    u = constants.h * nu / (constants.k * temperature)

    kappa = 1.0 + (u**2) / 24.0
    logger.info(f"Wigner tunneling coefficient: {kappa}")  # noqa: G004
    return kappa


def eckart(
    vibfreq: float,
    delta_forward: float,
    delta_backward: Optional[float] = None,  # noqa: UP007
    temperature: Union[float, np.ndarray] = 298.15,  # noqa: UP007
) -> Union[float, np.ndarray]:  # noqa: UP007
    """Calculate the Eckart correction to quantum tunneling.

    References are
    [*J. Phys. Chem.* **1962**, 66, 3, 532-533](https://doi.org/10.1021/j100809a040)
    and
    [*J. Res. Natl. Inst. Stand. Technol.*, **1981**, 86, 357](https://doi.org/10.6028/jres.086.014).

    Parameters
    ----------
    vibfreq : array-like
        Magnitude of the imaginary frequency in cm$^{-1}$. **Only the absolute value
        is used**.
    delta_forward : array-like
        Activation enthalpy at 0 K for the forward reaction.
    delta_backward : array-like, optional
        Activation enthalpy at 0 K for the reverse reaction. If delta_backward
        is not given, the "symmetrical" Eckart model is used (i.e.,
        ``delta_backward == delta_forward`` is assumed).
    temperature : array-like, optional
        Absolute temperature in Kelvin.

    Returns
    -------
    kappa : array-like
        The quantum tunneling correction.

    Raises
    ------
    ValueError
        If `vibfreq` is zero.

    Examples
    --------
    >>> eckart(1218, 13672.624, 24527.729644, temperature=300)
    array(3.9)
    >>> eckart(1218, 13672.624, 24527.729644, temperature=[200, 298.15])
    array([17.1, 4.0])
    >>> eckart([1218, 200], 13672.624, 24527.729644, temperature=400)
    array([2.3, 1.0])

    If no backward barrier is given, a symmetric Eckart potential is assumed:

    >>> eckart(414.45, 394.54)
    array(1.16)
    >>> eckart(414.45, 789.08)
    array(1.3)
    >>> eckart(3315.6, 3156.31)
    array(3.3)

    And if either the forward or backward barrier is non-positive, we fall back
    to the Wigner correction, but a warning is issued:

    >>> eckart(190.5927, 109920.73434972763, -154.0231580734253)
    1.03525
    >>> eckart(190.5927, -154.0231580734253, 109920.73434972763)
    1.03525
    >>> eckart(190.5927, -154.0231580734253)
    1.03525

    """
    temperature = np.asarray(temperature)

    nu = _check_nu(vibfreq)
    u = constants.h * nu / (constants.k * temperature)

    if delta_backward is None:
        delta_backward = delta_forward

    logger.debug(f"forward  potential barrier: {delta_forward} J/mol")  # noqa: G004
    logger.debug(f"backward potential barrier: {delta_backward} J/mol")  # noqa: G004

    if delta_forward <= 0 or delta_backward <= 0:
        logger.warning(
            "forward or backward barrier is non-positive, falling back to Wigner correction",  # noqa: E501
        )
        return wigner(vibfreq, temperature)

    assert np.all(delta_forward > 0), "forward barrier should be positive"
    assert np.all(delta_backward > 0), "backward barrier should be positive"

    # convert energies in joules per mole to joules
    delta_forward = delta_forward / constants.N_A
    delta_backward = delta_backward / constants.N_A

    assert delta_backward is not None, "delta_backward should be given"
    two_pi = 2.0 * np.pi
    alpha1 = two_pi * delta_forward / (constants.h * nu)
    alpha2 = two_pi * delta_backward / (constants.h * nu)

    kappa = _eckart(u, alpha1, alpha2)
    logger.info(f"Eckart tunneling coefficient: {kappa}")  # noqa: G004
    return kappa


@np.vectorize
def _eckart(
    u: float,
    alpha1: float,
    alpha2: Optional[float] = None,  # noqa: UP007
) -> float:  # noqa: RUF100
    """Implement of the (unsymmetrical) Eckart tunneling approximation.

    This is based on doi:10.1021/j100809a040 and doi:10.6028/jres.086.014.

    Parameters
    ----------
    u : array-like
        u = h * nu / (k * T).
    alpha1 : array-like
        alpha1 = 2 * pi * delta_forward / (h * nu).
    alpha2 : array-like, optional
        alpha2 = 2 * pi * delta_backward / (h * nu). If not set, the
        symmetrical Eckart potential is employed.

    Returns
    -------
    float

    Notes
    -----
    This function integrates the Eckart transmission function over a Boltzmann
    distribution using a mixed set of quadratures (Gauss quadrature for values
    below zero and Laguerre quadrature for values from zero to infinity). The
    orders for both quadratures are fixed and are the smallest numbers that
    allow us to reproduce values from the literature (doi:10.1021/j100809a040).

    Both alpha1 and alpha2 should be non-negative.

    Examples
    --------
    Symmetrical barrier:

    >>> _eckart(10, 20)
    1150.

    Unsymmetrical barrier:

    >>> _eckart(2, 0.5, 1.0)
    1.125
    >>> _eckart(2, 1.0, 0.5)
    1.125

    """
    # minimum orders that pass tests (with same precision as order=100)
    gauss_n = 18
    laguerre_n = 11

    # symmetrical potential?
    if alpha2 is None:
        alpha2 = alpha1

    two_pi = 2.0 * np.pi
    v1 = alpha1 * u / (two_pi)
    v2 = alpha2 * u / (two_pi)

    d = 4.0 * alpha1 * alpha2 - np.pi**2
    if d > 0:  # noqa: SIM108
        D = np.cosh(np.sqrt(d))  # noqa: N806
    else:
        D = np.cos(np.sqrt(np.abs(d)))  # noqa: N806

    sqrt_alpha1 = np.sqrt(alpha1)
    sqrt_alpha2 = np.sqrt(alpha2)
    F = (  # noqa: N806
        np.sqrt(2.0)
        * sqrt_alpha1
        * sqrt_alpha2
        / (np.sqrt(np.pi) * (sqrt_alpha1 + sqrt_alpha2))
    )

    def f(eps, with_exp=True):  # noqa: FBT002
        """Transmission function multiplied or not by the Boltzmann weight."""
        a1 = F * np.sqrt((eps + v1) / u)
        a2 = F * np.sqrt((eps + v2) / u)

        qplus = np.cosh(two_pi * (a1 + a2))
        qminus = np.cosh(two_pi * (a1 - a2))

        p = (qplus - qminus) / (D + qplus)
        if with_exp:
            return p * np.exp(-eps)
        return p

    # integral from -min(v1, v2) to zero using Gauss quadrature
    integ1 = fixed_quad(f, -min(v1, v2), 0, n=gauss_n)[0]

    # integral from 0 to infinity using Laguerre quadrature
    x, w = roots_laguerre(n=laguerre_n)
    integ2 = w @ f(x, with_exp=False)

    return integ1 + integ2
