#!/usr/bin/env python3

"""High-level interface."""

import logging
import warnings

import numpy as _np
from scipy.misc import derivative as _derivative

from overreact import constants
from overreact import coords
from overreact import core as _core
from overreact import io as _io
from overreact import rates
from overreact import _thermo
from overreact import tunnel

from overreact.core import get_transition_states
from overreact.core import is_transition_state
from overreact.core import parse_reactions
from overreact.datasets import data_path
from overreact.io import parse_compounds
from overreact.io import parse_model
from overreact.simulate import get_dydt
from overreact.simulate import get_y
from overreact._thermo import get_delta

logger = logging.getLogger(__name__)


def get_internal_energies(compounds, qrrho=True, temperature=298.15):
    """Obtain internal energies for compounds.

    Parameters
    ----------
    compounds : dict-like
    qrrho : bool, optional
        Apply the quasi-rigid rotor harmonic oscilator (QRRHO) approximation of
        M. Head-Gordon (see doi:10.1021/jp509921r) on top of the classical
        RRHO.
    temperature : array-like, optional
        Absolute temperature in Kelvin.

    Returns
    -------
    array-like

    Examples
    --------
    >>> model = parse_model("data/ethane/B97-3c/model.k")
    >>> internal_energies = get_internal_energies(model.compounds)
    >>> internal_energies - internal_energies.min()
    array([   0.        , 9207.05855504])

    Beware that the values below have not been validated yet:

    >>> model = parse_model("data/tanaka1996/UMP2/6-311G(2df,2pd)/model.jk")
    >>> internal_energies = get_internal_energies(model.compounds)
    >>> internal_energies - internal_energies.min()
    array([1.20690241e+09, 1.06016411e+08, 0.00000000e+00, 1.20864175e+09,
           1.04280309e+08])
    """
    compounds = _io._check_compounds(compounds)
    internal_energes = []
    for name in compounds:
        logger.info(f"calculate internal energy: {name}")

        # TODO(schneiderfelipe): inertia might benefit from caching
        moments, _, _ = coords.inertia(
            compounds[name].atommasses, compounds[name].atomcoords
        )

        internal_energy = _thermo.calc_internal_energy(
            energy=compounds[name].energy,
            degeneracy=compounds[name].mult,
            moments=moments,
            vibfreqs=compounds[name].vibfreqs,
            qrrho=qrrho,
            temperature=temperature,
        )
        internal_energes.append(internal_energy)
    return _np.array(internal_energes)


def get_enthalpies(compounds, qrrho=True, temperature=298.15):
    """Obtain enthalpies for compounds.

    Parameters
    ----------
    compounds : dict-like
    qrrho : bool, optional
        Apply the quasi-rigid rotor harmonic oscilator (QRRHO) approximation of
        M. Head-Gordon (see doi:10.1021/jp509921r) on top of the classical
        RRHO.
    temperature : array-like, optional
        Absolute temperature in Kelvin.

    Returns
    -------
    array-like

    Examples
    --------
    >>> model = parse_model("data/ethane/B97-3c/model.k")
    >>> enthalpies = get_enthalpies(model.compounds)
    >>> enthalpies - enthalpies.min()
    array([   0.        , 9207.05855504])

    Beware that the values below have not been validated yet:

    >>> model = parse_model("data/tanaka1996/UMP2/6-311G(2df,2pd)/model.jk")
    >>> enthalpies = get_enthalpies(model.compounds)
    >>> enthalpies - enthalpies.min()
    array([1.20690241e+09, 1.06016411e+08, 0.00000000e+00, 1.20864175e+09,
           1.04280309e+08])
    """
    compounds = _io._check_compounds(compounds)
    enthalpies = []
    for name in compounds:
        logger.info(f"calculate enthalpy: {name}")

        # TODO(schneiderfelipe): inertia might benefit from caching
        moments, _, _ = coords.inertia(
            compounds[name].atommasses, compounds[name].atomcoords
        )

        enthalpy = _thermo.calc_enthalpy(
            energy=compounds[name].energy,
            degeneracy=compounds[name].mult,
            moments=moments,
            vibfreqs=compounds[name].vibfreqs,
            qrrho=qrrho,
            temperature=temperature,
        )
        enthalpies.append(enthalpy)
    return _np.array(enthalpies)


def get_entropies(compounds, qrrho=True, temperature=298.15, pressure=constants.atm):
    """Obtain entropies for compounds.

    Parameters
    ----------
    compounds : dict-like
    qrrho : bool, optional
        Apply the quasi-rigid rotor harmonic oscilator (QRRHO) approximation of
        S. Grimme (see doi:10.1002/chem.201200497) on top of the classical
        RRHO.
    temperature : array-like, optional
        Absolute temperature in Kelvin.
    pressure : array-like, optional
        Reference gas pressure.

    Returns
    -------
    array-like

    Examples
    --------
    >>> model = parse_model("data/ethane/B97-3c/model.k")
    >>> entropies = get_entropies(model.compounds)
    >>> entropies - entropies.min()
    array([6.00745174, 0.        ])

    Beware that the values below have not been validated yet:

    >>> model = parse_model("data/tanaka1996/UMP2/6-311G(2df,2pd)/model.jk")
    >>> entropies = get_entropies(model.compounds)
    >>> entropies - entropies.min()
    array([ 26.89918418,   0.        , 109.67559002,  35.85552122,
            27.55907539])
    """
    compounds = _io._check_compounds(compounds)
    entropies = []
    for name in compounds:
        logger.info(f"calculate entropy: {name}")

        if "point_group" in compounds[name]:
            point_group = compounds[name].point_group
        else:
            point_group = coords.find_point_group(
                compounds[name].atommasses, compounds[name].atomcoords
            )
        symmetry_number = coords.symmetry_number(point_group)

        # TODO(schneiderfelipe): inertia might benefit from caching
        moments, _, _ = coords.inertia(
            compounds[name].atommasses, compounds[name].atomcoords
        )

        environment = _core._get_environment(name)
        entropy = _thermo.calc_entropy(
            atommasses=compounds[name].atommasses,
            energy=compounds[name].energy,
            degeneracy=compounds[name].mult,
            moments=moments,
            symmetry_number=symmetry_number,
            vibfreqs=compounds[name].vibfreqs,
            environment=environment,
            qrrho=qrrho,
            temperature=temperature,
            pressure=pressure,
        )

        if compounds[name].symmetry is not None:
            entropy += _thermo.change_reference_state(compounds[name].symmetry, 1)
        entropies.append(entropy)
    return _np.array(entropies)


def get_freeenergies(
    compounds, bias=0.0, qrrho=True, temperature=298.15, pressure=constants.atm
):
    """Obtain free energies for compounds.

    Parameters
    ----------
    compounds : dict-like
    bias : array-like, optional
        Energy to be added to free energies.
    qrrho : bool, optional
        Apply both the quasi-rigid rotor harmonic oscilator (QRRHO)
        approximations of M. Head-Gordon (enthalpy correction, see
        doi:10.1021/jp509921r) and S. Grimme (entropy correction, see
        doi:10.1002/chem.201200497) on top of the classical RRHO.
    temperature : array-like, optional
        Absolute temperature in Kelvin.
    pressure : array-like, optional
        Reference gas pressure.

    Returns
    -------
    array-like

    Examples
    --------
    >>> model = parse_model("data/ethane/B97-3c/model.k")
    >>> freeenergies = get_freeenergies(model.compounds)
    >>> freeenergies - freeenergies.min()
    array([    0.        , 10998.18028986])

    You can set a simple energy bias, either as a constant or compound-wise:

    >>> get_freeenergies(model.compounds, bias=1.0) - freeenergies
    array([1.0, 1.0])
    >>> get_freeenergies(model.compounds, bias=-1.0) - freeenergies
    array([-1.0, -1.0])
    >>> get_freeenergies(model.compounds, bias=[1.0, -1.0]) - freeenergies
    array([ 1., -1.])

    Beware that the values below have not been validated yet:

    >>> model = parse_model("data/tanaka1996/UMP2/6-311G(2df,2pd)/model.jk")
    >>> freeenergies = get_freeenergies(model.compounds)
    >>> freeenergies - freeenergies.min()
    array([1.20692709e+09, 1.06049111e+08, 0.00000000e+00, 1.20866376e+09,
           1.04304792e+08])
    """
    enthalpies = get_enthalpies(compounds, qrrho=qrrho, temperature=temperature)
    entropies = get_entropies(compounds, qrrho=qrrho, temperature=temperature)
    return enthalpies - temperature * entropies + _np.asanyarray(bias)


# TODO(schneiderfelipe): this should probably be deprecated but it is very
# good as a starting point for sensitivity analyses in general.
def get_drc(
    scheme,
    compounds,
    y0,
    t_span=None,
    method="Radau",
    qrrho=True,
    scale="l mol-1 s-1",
    temperature=298.15,
    dx=1.5e3,  # joules
    order=3,
):
    """Calculate the degree of rate control for a single compound.

    Examples
    --------
    >>> model = parse_model("data/tanaka1996/UMP2/6-311G(2df,2pd)/model.jk")
    """
    x0 = _np.zeros(len(scheme.compounds))

    def func(t, x=0.0, i=-1):
        bias = _np.copy(x0)
        bias[i] += x

        k = get_k(
            scheme,
            compounds=compounds,
            bias=bias,
            # tunneling=tunneling,
            qrrho=qrrho,
            scale=scale,
            temperature=temperature,
            # pressure=pressure,
            # delta_freeenergies=delta_freeenergies,
            # molecularity=molecularity,
            # volume=volume,
        )
        _, r = get_y(get_dydt(scheme, k), y0=y0, t_span=t_span, method=method)

        return _np.log(r(t))

    def drc(t, i=-1):
        return (
            -constants.R
            * temperature
            * _derivative(lambda x: func(t, x, i), x0=0.0, dx=dx, n=1, order=order)
        )

    return drc


def get_k(
    scheme,
    compounds=None,
    bias=0.0,
    tunneling="eckart",
    qrrho=True,
    scale="l mol-1 s-1",
    temperature=298.15,
    pressure=constants.atm,
    delta_freeenergies=None,
    molecularity=None,
    volume=None,
):
    """Obtain reaction rate constants.

    Parameters
    ----------
    scheme : Scheme
    compounds : dict-like, optional
    bias : array-like, optional
        Energy to be added to free energies.
    tunneling : str or None, optional
        Choose between "eckart", "wigner" or None.
    qrrho : bool, optional
        Apply both the quasi-rigid rotor harmonic oscilator (QRRHO)
        approximations of M. Head-Gordon (enthalpy correction, see
        doi:10.1021/jp509921r) and S. Grimme (entropy correction, see
        doi:10.1002/chem.201200497) on top of the classical RRHO.
    scale : str, optional
        Reaction rate units. Possible values are "cm3 mol-1 s-1",
        "l mol-1 s-1", "m3 mol-1 s-1", "cm3 particle-1 s-1", "mmHg-1 s-1" and
        "atm-1 s-1".
    temperature : array-like, optional
        Absolute temperature in Kelvin.
    pressure : array-like, optional
        Reference gas pressure.
    delta_freeenergies : array-like, optional
    molecularity : array-like, optional
        Reaction order, i.e., number of molecules that come together to react.
    volume : float, optional
        Molar volume.

    Returns
    -------
    array-like

    Notes
    -----
    Some symbols are accepted as alternatives in "scale": "M-1", "ml" and
    "torr-1" are understood as "l mol-1", "cm3" and "mmHg-1", respectively.

    Examples
    --------
    >>> model = parse_model("data/ethane/B97-3c/model.k")
    >>> get_k(model.scheme, model.compounds, tunneling=None)
    array([7.35299453e+10])

    You can set a simple energy bias, either as a constant or compound-wise:

    >>> get_k(model.scheme, model.compounds, bias=100.0, tunneling=None)
    array([7.35299453e+10])
    >>> get_k(model.scheme, model.compounds, bias=-100.0, tunneling=None)
    array([7.35299453e+10])
    >>> get_k(model.scheme, model.compounds, bias=[100.0, -100.0], tunneling=None)
    array([7.97081495e+10])

    The units of the returned reaction rate constants can be selected (beware
    that the values below have not been validated yet):

    >>> model = parse_model("data/tanaka1996/UMP2/6-311G(2df,2pd)/model.jk")
    >>> get_k(model.scheme, model.compounds, tunneling=None)
    array([14820222.69476697])
    >>> get_k(model.scheme, model.compounds, tunneling=None, scale="atm-1 s-1")
    array([605762.44228638])
    >>> get_k(model.scheme, model.compounds, tunneling=None,
    ...       scale="cm3 particle-1 s-1")
    array([2.46095588e-14])
    >>> get_k(model.scheme, model.compounds, scale="cm3 particle-1 s-1")
    array([6.26670474e-14])
    """
    scheme = _core._check_scheme(scheme)
    if compounds is not None:
        compounds = _io._check_compounds(compounds)
    if delta_freeenergies is None:
        freeenergies = get_freeenergies(
            compounds,
            bias=bias,
            qrrho=qrrho,
            temperature=temperature,
            pressure=pressure,
        )
        delta_freeenergies = get_delta(scheme.B, freeenergies)

    if molecularity is None:
        molecularity = _thermo.get_molecularity(scheme.A)

    k = rates.eyring(
        delta_freeenergies,
        molecularity,
        temperature=temperature,
        pressure=pressure,
        volume=volume,
    )
    logger.info(
        "(classical) reaction rate constants: "
        f"{', '.join([f'{v:7.3g}' for v in k])} atm⁻ⁿ⁺¹·s⁻¹"
    )
    if tunneling is not None:
        if compounds is not None:
            kappa = get_kappa(
                scheme, compounds, method=tunneling, temperature=temperature
            )
            k *= kappa
        else:
            # TODO(schneiderfelipe): when get_kappa accept deltas, this will
            # be probably unnecessary.
            logger.warning(
                "assuming unitary tunneling coefficients due to incomplete "
                "compound data"
            )
        logger.info(
            "(adjusted) reaction rate constants: "
            f"{', '.join([f'{v:7.3g}' for v in k])} atm⁻ⁿ⁺¹·s⁻¹"
        )

    # TODO(schneiderfelipe): ensure diffusional limit for reactions in
    # solvation using Collins-Kimball theory.
    return rates.convert_rate_constant(
        k, scale, molecularity=molecularity, temperature=temperature, pressure=pressure
    )


# TODO(schneiderfelipe): accept deltas and make compounds optional.
def get_kappa(scheme, compounds, method="eckart", temperature=298.15):
    r"""Obtain tunneling transmission coefficients.

    One tunneling transmission coefficient is calculated for each reaction. If
    a reaction lacks a transition state (i.e., a dummy half-equilibrium), its
    transmission coefficient is set to unity.

    Parameters
    ----------
    scheme : Scheme
    compounds : dict-like
    tunneling : str or None, optional
        Choose between "eckart" or "wigner".
    temperature : array-like, optional
        Absolute temperature in Kelvin.

    Returns
    -------
    array-like

    Examples
    --------
    The following example is an estimate for the rate of methyl rotation in ethane.
    The calculated value is off by less than 2% from the experimental value
    (:math:`\frac{1}{12 \times 10^{-12}} \text{s}^{-1} = 8.33 \times 10^{10}
    \text{s}^{-1}`, see doi:10.1126/science.1132178). (As a side note, the
    experimental barrier is 12.04 kJ/mol :cite:`Hirota_1979`.)

    >>> model = parse_model("data/ethane/B97-3c/model.k")
    >>> kappa = get_kappa(model.scheme, model.compounds)
    >>> kappa
    array([1.10949425])
    >>> kappa * get_k(model.scheme, model.compounds, tunneling=None)
    array([8.e+10])

    Beware that the values below have not been validated yet:

    >>> model = parse_model("data/tanaka1996/UMP2/6-311G(2df,2pd)/model.jk")
    >>> get_kappa(model.scheme, model.compounds)
    array([2.54645148])
    """
    scheme = _core._check_scheme(scheme)
    compounds = _io._check_compounds(compounds)

    if method == "eckart":
        energies = [compounds[name].energy for name in scheme.compounds]
        delta_forward = get_delta(scheme.B, energies)  # B - A
        delta_backward = delta_forward - get_delta(
            scheme.A, energies
        )  # B - C == B - A - (C - A)

    kappas = []
    for i, ts in enumerate(
        get_transition_states(scheme.A, scheme.B, scheme.is_half_equilibrium)
    ):
        if ts is None:
            kappas.append(1.0)
        else:
            transition_state = scheme.compounds[ts]
            vibfreq = compounds[transition_state].vibfreqs[0]

            if method == "eckart":
                with warnings.catch_warnings():
                    warnings.filterwarnings("error")
                    try:
                        kappa = tunnel.eckart(
                            vibfreq,
                            delta_forward[i],
                            delta_backward[i],
                            temperature=temperature,
                        )
                    except RuntimeWarning as e:
                        logger.warning(f"using Wigner tunneling correction: {e}")
                        kappa = tunnel.wigner(vibfreq, temperature=temperature)
            elif method == "wigner":
                kappa = tunnel.wigner(vibfreq, temperature=temperature)
            else:
                raise ValueError(f"unavailable method: '{method}'")

            kappas.append(kappa)

    # TODO(schneiderfelipe): is this correct? shouldn't we correct shapes
    # somewhere else?
    kappas = _np.array(kappas).flatten()
    logger.info(
        "(quantum) tunneling coefficients: "
        f"{', '.join([f'{kappa:7.3g}' for kappa in kappas])}"
    )
    return kappas