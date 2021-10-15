#!/usr/bin/env python3

"""High-level interface."""

import logging
import warnings

import numpy as np
from scipy.misc import derivative

import overreact as rx
from overreact import constants
from overreact import coords
from overreact import rates
from overreact import tunnel

logger = logging.getLogger(__name__)


def get_internal_energies(compounds, qrrho=True, temperature=298.15):
    """Obtain internal energies for compounds.

    Parameters
    ----------
    compounds : dict-like
    qrrho : bool, optional
        Apply the quasi-rigid rotor harmonic oscillator (QRRHO) approximation of
        M. Head-Gordon (see doi:10.1021/jp509921r) on top of the classical
        RRHO.
    temperature : array-like, optional
        Absolute temperature in Kelvin.

    Returns
    -------
    array-like

    Examples
    --------
    >>> import overreact as rx
    >>> from overreact import constants

    >>> model = rx.parse_model("data/ethane/B97-3c/model.k")
    >>> internal_energies = get_internal_energies(model.compounds)
    >>> (internal_energies - internal_energies.min()) / constants.kcal
    array([0. , 2.20053981])

    """
    compounds = rx.io._check_compounds(compounds)
    internal_energies = []
    for name in compounds:
        logger.info(f"calculate internal energy: {name}")

        # TODO(schneiderfelipe): inertia might benefit from caching
        moments, _, _ = coords.inertia(
            compounds[name].atommasses, compounds[name].atomcoords
        )

        internal_energy = rx.thermo.calc_internal_energy(
            energy=compounds[name].energy,
            degeneracy=compounds[name].mult,
            moments=moments,
            vibfreqs=compounds[name].vibfreqs,
            qrrho=qrrho,
            temperature=temperature,
        )
        internal_energies.append(internal_energy)
    return np.array(internal_energies)


def get_enthalpies(compounds, qrrho=True, temperature=298.15):
    """Obtain enthalpies for compounds.

    Parameters
    ----------
    compounds : dict-like
    qrrho : bool, optional
        Apply the quasi-rigid rotor harmonic oscillator (QRRHO) approximation of
        M. Head-Gordon (see doi:10.1021/jp509921r) on top of the classical
        RRHO.
    temperature : array-like, optional
        Absolute temperature in Kelvin.

    Returns
    -------
    array-like

    Examples
    --------
    >>> import overreact as rx
    >>> from overreact import constants

    >>> model = rx.parse_model("data/ethane/B97-3c/model.k")
    >>> enthalpies = get_enthalpies(model.compounds)
    >>> (enthalpies - enthalpies.min()) / constants.kcal
    array([0. , 2.20053981])

    The enthalpies at absolute zero can easily be obtained (this is used,
    e.g., in the calculation of the Eckart tunneling coefficient). We can use
    this to calculate, for instance, the thermal contributions to the
    enthalpy:

    >>> zero_enthalpies = get_enthalpies(model.compounds, temperature=0)
    >>> (enthalpies - zero_enthalpies) / constants.kcal
    array([2.78, 2.50])
    """
    compounds = rx.io._check_compounds(compounds)
    enthalpies = []
    for name in compounds:
        logger.info(f"calculate enthalpy: {name}")

        # TODO(schneiderfelipe): inertia might benefit from caching
        moments, _, _ = coords.inertia(
            compounds[name].atommasses, compounds[name].atomcoords
        )

        enthalpy = rx.thermo.calc_enthalpy(
            energy=compounds[name].energy,
            degeneracy=compounds[name].mult,
            moments=moments,
            vibfreqs=compounds[name].vibfreqs,
            qrrho=qrrho,
            temperature=temperature,
        )
        enthalpies.append(enthalpy)
    return np.array(enthalpies)


def get_entropies(
    compounds,
    environment=None,
    method="standard",
    qrrho=True,
    temperature=298.15,
    pressure=constants.atm,
):
    """Obtain entropies for compounds.

    Parameters
    ----------
    compounds : dict-like
    environment : str or None, optional
        Choose between "gas" and a solvent. This is chosen for you by default,
        based on the names of each compound. If given, all compounds will
        have the same behavior.
    method : str, optional
        Choose between "standard", "izato" (doi:10.1039/C9CP03226F) and "garza"
        (doi:10.1021/acs.jctc.9b00214).
    qrrho : bool, optional
        Apply the quasi-rigid rotor harmonic oscillator (QRRHO) approximation of
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
    >>> import overreact as rx
    >>> from overreact import constants

    >>> model = rx.parse_model("data/ethane/B97-3c/model.k")
    >>> entropies = get_entropies(model.compounds)
    >>> (entropies - entropies.min()) / constants.calorie
    array([1.4, 0. ])

    You can consider all compounds as solvated if you want:

    >>> sol_entropies = get_entropies(model.compounds, environment="solvent")
    >>> (sol_entropies - entropies) / constants.calorie
    array([-6.35360874, -6.35360874])
    """
    compounds = rx.io._check_compounds(compounds)
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

        if environment is None:
            environment = rx.core._get_environment(name)
        entropy = rx.thermo.calc_entropy(
            atommasses=compounds[name].atommasses,
            atomcoords=compounds[name].atomcoords,
            energy=compounds[name].energy,
            degeneracy=compounds[name].mult,
            moments=moments,
            symmetry_number=symmetry_number,
            vibfreqs=compounds[name].vibfreqs,
            environment=environment,
            method=method,
            qrrho=qrrho,
            temperature=temperature,
            pressure=pressure,
        )

        if compounds[name].symmetry is not None:
            # The negative sign here seems correct. See equations (9) and (10)
            # of doi:10.1002/qua.25686.
            entropy -= rx.change_reference_state(
                compounds[name].symmetry, 1, temperature=temperature, pressure=pressure
            )

        entropies.append(entropy)
    return np.array(entropies)


def _check_qrrho(qrrho):
    """Get options for QRRHO for both enthalpy and entropy.

    Parameters
    ----------
    qrrho : bool or tuple-like
        Apply both the quasi-rigid rotor harmonic oscillator (QRRHO)
        approximations of M. Head-Gordon (enthalpy correction, see
        doi:10.1021/jp509921r) and S. Grimme (entropy correction, see
        doi:10.1002/chem.201200497) on top of the classical RRHO.

    Returns
    -------
    qrrho_enthalpy, qrrho_entropy : bool

    Raises
    ------
    ValueError
        If the QRRHO specification could not be recognized.

    Examples
    --------
    >>> _check_qrrho(True)
    (True, True)
    >>> _check_qrrho(False)
    (False, False)
    >>> _check_qrrho((True, False))
    (True, False)
    >>> _check_qrrho((False, True))
    (False, True)
    """
    if qrrho is True:
        return True, True
    elif qrrho is False:
        return False, False
    elif isinstance(qrrho, tuple):
        return qrrho
    else:
        raise ValueError(f"unrecognized QRRHO specification: {qrrho}")


def get_freeenergies(
    compounds,
    bias=0.0,
    environment=None,
    method="standard",
    qrrho=True,
    temperature=298.15,
    pressure=constants.atm,
):
    """Obtain free energies for compounds.

    Parameters
    ----------
    compounds : dict-like
    bias : array-like, optional
        Energy to be added to free energies.
    environment : str or None, optional
        Choose between "gas" and a solvent. This is chosen for you by default,
        based on the names of each compound. If given, all compounds will
        have the same behavior.
    method : str, optional
        Choose between "standard", "izato" (doi:10.1039/C9CP03226F) and "garza"
        (doi:10.1021/acs.jctc.9b00214).
    qrrho : bool or tuple-like, optional
        Apply both the quasi-rigid rotor harmonic oscillator (QRRHO)
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
    >>> import overreact as rx
    >>> from overreact import constants

    >>> model = rx.parse_model("data/ethane/B97-3c/model.k")
    >>> freeenergies = get_freeenergies(model.compounds, qrrho=(False, True))
    >>> (freeenergies - freeenergies.min()) / constants.kcal
    array([0. , 2.62281461])

    >>> freeenergies = get_freeenergies(model.compounds)
    >>> (freeenergies - freeenergies.min()) / constants.kcal
    array([0. , 2.62862818])

    You can consider all compounds as solvated if you want:

    >>> sol_freeenergies = get_freeenergies(model.compounds, environment="solvent")
    >>> (sol_freeenergies - freeenergies) / constants.kcal
    array([1.89432845, 1.89432845])

    You can set a simple energy bias, either as a constant or compound-wise:

    >>> get_freeenergies(model.compounds, bias=1.0) - freeenergies
    array([1., 1.])
    >>> get_freeenergies(model.compounds, bias=-1.0) - freeenergies
    array([-1., -1.])
    >>> get_freeenergies(model.compounds, bias=[1.0, -1.0]) - freeenergies
    array([ 1., -1.])
    """
    qrrho_enthalpy, qrrho_entropy = _check_qrrho(qrrho)
    enthalpies = get_enthalpies(
        compounds, qrrho=qrrho_enthalpy, temperature=temperature
    )
    entropies = get_entropies(
        compounds,
        environment=environment,
        method=method,
        qrrho=qrrho_entropy,
        temperature=temperature,
        pressure=pressure,
    )
    # TODO(schneiderfelipe): log the contribution of bias
    return enthalpies - temperature * entropies + np.asarray(bias)


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
    r"""Obtain reaction rate constants.

    Parameters
    ----------
    scheme : Scheme
    compounds : dict-like, optional
    bias : array-like, optional
        Energy to be added to free energies.
    tunneling : str or None, optional
        Choose between "eckart", "wigner" or None (or "none").
    qrrho : bool or tuple-like, optional
        Apply both the quasi-rigid rotor harmonic oscillator (QRRHO)
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
        If set, this is used to calculate `delta_moles` for
        `equilibrium_constant`, which effectively calculates a solution
        equilibrium constant between reactants and the transition state for
        gas phase data. You should set this to `None` if your free energies
        were already adjusted for solution Gibbs free energies.
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
    Below is an example of a rotating ethane. How many turns it does per
    second? We use Eckart tunneling by default, but this can be changed (check
    doi:10.1126/science.1132178 to see which is closest to the experiment):

    >>> model = rx.parse_model("data/ethane/B97-3c/model.k")
    >>> get_k(model.scheme, model.compounds)
    array([8.16e+10])
    >>> get_k(model.scheme, model.compounds, qrrho=(False, True))
    array([8.24968117e+10])
    >>> get_k(model.scheme, model.compounds, qrrho=False)
    array([8.26909266e+10])

    >>> get_k(model.scheme, model.compounds, tunneling="wigner")
    array([7.99e+10])
    >>> get_k(model.scheme, model.compounds, tunneling=None)
    array([7.35e+10])

    The units of the returned reaction rate constants can be selected for
    non-unimolecular processes:

    >>> model = rx.parse_model("data/tanaka1996/UMP2/cc-pVTZ/model.jk")
    >>> get_k(model.scheme, model.compounds, temperature=200.0,
    ...       scale="cm3 particle-1 s-1")
    array([1.0e-14])

    (By the way, the experimental reaction rate constant for this reaction is
    :math:`1.03 \times 10^{-13} \text{cm}^3 \text{particle}^{-1} \text{s}^{-1}`.)
    The returned units are "M-1 s-1" by default:

    >>> get_k(model.scheme, model.compounds) \
    ... == get_k(model.scheme, model.compounds, scale="l mol-1 s-1")
    array([ True])

    You can also turn the tunneling correction off by using the string "none":

    >>> get_k(model.scheme, model.compounds, tunneling="none") \
    ... == get_k(model.scheme, model.compounds, tunneling=None)
    array([ True])

    You can set a simple energy bias, either as a constant or compound-wise:

    >>> get_k(model.scheme, model.compounds, bias=1.0 * constants.kcal,
    ...       temperature=300.0, scale="cm3 particle-1 s-1")
    array([5.14e-13])
    >>> get_k(model.scheme, model.compounds,
    ...       bias=np.array([0.0, 0.0, -1.4, 0.0, 0.0]) * constants.kcal,
    ...       temperature=300.0, scale="cm3 particle-1 s-1")
    array([1.1e-12])
    """
    qrrho_enthalpy, qrrho_entropy = _check_qrrho(qrrho)
    scheme = rx.core._check_scheme(scheme)
    if compounds is not None:
        compounds = rx.io._check_compounds(compounds)
    if delta_freeenergies is None:
        freeenergies = get_freeenergies(
            compounds,
            bias=bias,
            qrrho=(qrrho_enthalpy, qrrho_entropy),
            # NOTE(schneiderfelipe): ensure we get rate constants in M-1 s-1.
            # TODO(schneiderfelipe): this strategy will have to change
            # somewhat when we improve solvation entropy models.
            environment="solvent",
            temperature=temperature,
            pressure=pressure,
        )

        # TODO(schneiderfelipe): log the contribution of reaction symmetry
        delta_freeenergies = rx.get_delta(
            scheme.B, freeenergies
        ) - temperature * rx.get_reaction_entropies(
            scheme.B, temperature=temperature, pressure=pressure
        )

    if molecularity is None:
        molecularity = rx.thermo.get_molecularity(scheme.A)

    k = rates.eyring(
        delta_freeenergies,
        # NOTE(schneiderfelipe): using molecularity here messes rate constant
        # units (by a factor of M-1 s-1 to atm-1 s-1), so we leave it as is.
        #
        # molecularity=molecularity,
        temperature=temperature,
        pressure=pressure,
        volume=volume,
    )

    # make reaction rate constants for equilibria as close as possible to one
    i = 0
    while i < len(scheme.is_half_equilibrium):
        if scheme.is_half_equilibrium[i]:
            pair = k[i : i + 2]
            _K = pair[0] / pair[1]

            k[i : i + 2] = pair / pair.min()
            assert np.isclose(_K, k[i] / k[i + 1])

            # loop over pairs of equilibria
            i += 1
        i += 1

    logger.info(
        "(classical) reaction rate constants: "
        f"{', '.join([f'{v:7.3g}' for v in k])} atm⁻ⁿ⁺¹·s⁻¹"
    )
    if tunneling not in {"none", None}:
        if compounds is not None:
            kappa = get_kappa(
                scheme,
                compounds,
                method=tunneling,
                qrrho=qrrho_enthalpy,
                temperature=temperature,
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
            "(tunneling) reaction rate constants: "
            f"{', '.join([f'{v:7.3g}' for v in k])} atm⁻ⁿ⁺¹·s⁻¹"
        )

    # TODO(schneiderfelipe): ensure diffusional limit for reactions in
    # solvation using Collins-Kimball theory. This includes half-equilibria.
    return rates.convert_rate_constant(
        k,
        new_scale=scale,
        # NOTE(schneiderfelipe): all the code above should always generate
        # rate constants in M-1 s-1, so that we convert them here.
        old_scale="l mol-1 s-1",
        molecularity=molecularity,
        temperature=temperature,
        pressure=pressure,
    )


# TODO(schneiderfelipe): accept deltas and make compounds optional.
def get_kappa(scheme, compounds, method="eckart", qrrho=True, temperature=298.15):
    r"""Obtain tunneling transmission coefficients.

    One tunneling transmission coefficient is calculated for each reaction. If
    a reaction lacks a transition state (i.e., a dummy half-equilibrium), its
    transmission coefficient is set to unity.

    Parameters
    ----------
    scheme : Scheme
    compounds : dict-like
    method : str or None, optional
        Choose between "eckart", "wigner" or None (or "none").
    qrrho : bool, optional
        Apply both the quasi-rigid rotor harmonic oscillator (QRRHO)
        approximations of M. Head-Gordon (enthalpy correction, see
        doi:10.1021/jp509921r) and S. Grimme (entropy correction, see
        doi:10.1002/chem.201200497) on top of the classical RRHO.
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

    >>> model = rx.parse_model("data/ethane/B97-3c/model.k")
    >>> kappa = get_kappa(model.scheme, model.compounds)
    >>> kappa
    array([1.110])
    >>> get_kappa(model.scheme, model.compounds, method="none")
    array([1.0])
    >>> kappa * get_k(model.scheme, model.compounds, tunneling=None)
    array([8.e+10])
    >>> get_kappa(model.scheme, model.compounds, method="none") \
    ... == get_kappa(model.scheme, model.compounds, method=None)
    array([ True])
    """
    scheme = rx.core._check_scheme(scheme)
    compounds = rx.io._check_compounds(compounds)

    if method == "eckart":
        # TODO(schneiderfelipe): We need electronic energies + ZPE here, so we
        # get smaller transmission coefficients.
        #
        # energies = [compounds[name].energy for name in scheme.compounds]
        energies = get_enthalpies(compounds, qrrho=qrrho, temperature=0.0)
        delta_forward = rx.get_delta(scheme.B, energies)  # B - A
        delta_backward = delta_forward - rx.get_delta(
            scheme.A, energies
        )  # B - C == B - A - (C - A)

    kappas = []
    for i, ts in enumerate(
        rx.get_transition_states(scheme.A, scheme.B, scheme.is_half_equilibrium)
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
            elif method in {"none", None}:
                kappa = 1.0
            else:
                raise ValueError(f"unavailable method: '{method}'")

            kappas.append(kappa)

    # TODO(schneiderfelipe): is this correct? shouldn't we correct shapes
    # somewhere else?
    kappas = np.asarray(kappas).flatten()
    logger.info(
        "(quantum) tunneling coefficients: "
        f"{', '.join([f'{kappa:7.3g}' for kappa in kappas])}"
    )
    return kappas


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

    Notes
    -----
    This is a work in progress!

    Examples
    --------
    >>> model = rx.parse_model("data/tanaka1996/UMP2/cc-pVTZ/model.jk")
    """
    temperature = np.asarray(temperature)

    x0 = np.zeros(len(scheme.compounds))

    def func(t, x=0.0, i=-1):
        bias = np.asarray(x0).copy()
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
        _, r = rx.get_y(rx.get_dydt(scheme, k), y0=y0, t_span=t_span, method=method)

        return np.log(r(t))

    def drc(t, i=-1):
        return (
            -constants.R
            * temperature
            * derivative(lambda x: func(t, x, i), x0=0.0, dx=dx, n=1, order=order)
        )

    return drc
