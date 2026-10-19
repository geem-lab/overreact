"""
Module containing the high-level application programming interface.

If you intend to use **overreact** as a library in a project, you should
probably start here.

Values per compound or per reaction lie along the last axis of arrays.
Conditions such as `temperature` and `pressure` broadcast against them, so a
scalar applies everywhere, an array of shape `(n,)` gives one value per
compound or reaction, and an array of shape `(m, 1)` adds a leading axis of
`m` conditions:

>>> import numpy as np
>>> import overreact as rx
>>> scheme = rx.Scheme.parse("A -> B")
>>> get_k(scheme, delta_freeenergies=[1e4], temperature=[[300.0], [400.0]])
array([[1.13e+11],
       [4.12e+11]])

Both halves of an equilibrium share the same conditions.
"""

from __future__ import annotations

__all__ = [
    "get_enthalpies",
    "get_entropies",
    "get_freeenergies",
    "get_internal_energies",
    "get_k",
    "get_kappa",
]


import logging
import warnings
from functools import partial
from typing import TYPE_CHECKING

import numpy as np

import overreact as rx
from overreact import _constants as constants
from overreact import coords, rates, thermo, tunnel
from overreact._misc import _derivative as derivative

if TYPE_CHECKING:
    from overreact.core import Scheme

logger = logging.getLogger(__name__)


def get_internal_energies(
    compounds: dict,
    qrrho: bool = True,
    temperature: float = 298.15,
):
    """Obtain internal energies for compounds at a given temperature.

    Parameters
    ----------
    compounds : dict-like
        A descriptor of the compounds.
        Mostly likely, this comes from a parsed input file.
        See `overreact.io.parse_model`.
    qrrho : bool, optional
        Apply the quasi-rigid rotor harmonic oscillator (QRRHO) approximation of
        M. Head-Gordon and others (see
        [*J. Phys. Chem. C* **2015**, 119, 4, 1840-1850](http://dx.doi.org/10.1021/jp509921r))
        on top of the classical RRHO.
    temperature : array-like, optional
        Absolute temperature in Kelvin.

    Returns
    -------
    array-like

    Examples
    --------
    >>> import overreact as rx
    >>> from overreact import _constants as constants
    >>> model = rx.parse_model("data/ethane/B97-3c/model.k")
    >>> internal_energies = get_internal_energies(model.compounds)
    >>> (internal_energies - internal_energies.min()) / constants.kcal
    array([0. , 2.20053981])

    """

    def internal_energy(name, compound):
        logger.info(f"calculate internal energy: {name}")
        return partial(thermo.calc_internal_energy, **_rrho(compound), qrrho=qrrho)

    return _per_compound(compounds, internal_energy, temperature=temperature)


def get_enthalpies(
    compounds: dict,
    qrrho: bool = True,
    temperature: float = 298.15,
):
    """Obtain enthalpies for compounds at a given temperature.

    Parameters
    ----------
    compounds : dict-like
        A descriptor of the compounds.
        Mostly likely, this comes from a parsed input file.
        See `overreact.io.parse_model`.
    qrrho : bool, optional
        Apply the quasi-rigid rotor harmonic oscillator (QRRHO) approximation of
        M. Head-Gordon and others (see
        [*J. Phys. Chem. C* **2015**, 119, 4, 1840-1850](http://dx.doi.org/10.1021/jp509921r))
        on top of the classical RRHO.
    temperature : array-like, optional
        Absolute temperature in Kelvin.

    Returns
    -------
    array-like

    Examples
    --------
    >>> import overreact as rx
    >>> from overreact import _constants as constants
    >>> model = rx.parse_model("data/ethane/B97-3c/model.k")
    >>> enthalpies = get_enthalpies(model.compounds)
    >>> (enthalpies - enthalpies.min()) / constants.kcal
    array([0. , 2.20053981])

    The enthalpies at absolute zero can easily be obtained (this is used,
    e.g., in the calculation of the Eckart tunneling coefficient, see
    `overreact.tunnel.eckart`). We can use this to calculate, for instance,
    the thermal contributions to the enthalpy:

    >>> zero_enthalpies = get_enthalpies(model.compounds, temperature=0)
    >>> (enthalpies - zero_enthalpies) / constants.kcal
    array([2.78, 2.50])
    """

    def enthalpy(name, compound):
        logger.info(f"calculate enthalpy: {name}")
        return partial(thermo.calc_enthalpy, **_rrho(compound), qrrho=qrrho)

    return _per_compound(compounds, enthalpy, temperature=temperature)


def get_entropies(
    compounds: dict,
    environment: str | None = None,
    method: str = "standard",
    qrrho: bool = True,
    temperature: float = 298.15,
    pressure: float = constants.atm,
):
    """Obtain entropies for compounds at a given temperature and pressure.

    Each compound is in the standard state of its environment: 1 M in
    solution, and `pressure` in the gas phase.

    Parameters
    ----------
    compounds : dict-like
        A descriptor of the compounds.
        Mostly likely, this comes from a parsed input file.
        See `overreact.io.parse_model`.
    environment : str or None, optional
        Choose between "gas" and a solvent. This is chosen for you by default,
        based on the names of each compound (e.g. `A(g)` or `A` is gas,
        `A(w)` or `A(...)` is solvated). In case this is given, all compounds
        will have the same behavior.
    method : str, optional
        This is a placeholder for future functionality.
        There are plans to implement more sophisticated methods for calculating
        entropies such as in
        [*Phys. Chem. Chem. Phys.*, **2019**, 21, 18920-18929](https://doi.org/10.1039/C9CP03226F)
        and
        [*J. Chem. Theory Comput.* **2019**, 15, 5, 3204-3214](https://doi.org/10.1021/acs.jctc.9b00214).
        Head over to the
        [discussions](https://github.com/geem-lab/overreact/discussions) if
        you're interested and would like to contribute.
        Leave this as "standard" for now.
    qrrho : bool, optional
        Apply the quasi-rigid rotor harmonic oscillator (QRRHO) approximation of
        S. Grimme (see
        [*Theory. Chem. Eur. J.*, **2012**, 18: 9955-9964](https://doi.org/10.1002/chem.201200497))
        on top of the classical RRHO.
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
    >>> from overreact import _constants as constants
    >>> model = rx.parse_model("data/ethane/B97-3c/model.k")
    >>> entropies = get_entropies(model.compounds)
    >>> (entropies - entropies.min()) / constants.calorie
    array([1.4, 0. ])

    You can consider all compounds as solvated if you want:

    >>> sol_entropies = get_entropies(model.compounds, environment="solvent")
    >>> (sol_entropies - entropies) / constants.calorie
    array([-6.35360874, -6.35360874])
    """

    def entropy(name, compound):
        logger.info(f"calculate entropy: {name}")
        if "point_group" in compound:
            point_group = compound.point_group
        else:
            point_group = coords.find_point_group(
                compound.atommasses, compound.atomcoords
            )
        calc = partial(
            thermo.calc_entropy,
            **_rrho(compound),
            atommasses=compound.atommasses,
            atomcoords=compound.atomcoords,
            symmetry_number=coords.symmetry_number(point_group),
            environment=environment or rx.core._get_environment(name),
            method=method,
            qrrho=qrrho,
        )
        if compound.symmetry is None:
            return calc

        def symmetric(temperature, pressure):
            # The negative sign here seems correct. See equations (9) and (10)
            # of doi:10.1002/qua.25686.
            return calc(
                temperature=temperature, pressure=pressure
            ) - thermo.change_reference_state(
                compound.symmetry, 1, temperature=temperature, pressure=pressure
            )

        return symmetric

    return _per_compound(compounds, entropy, temperature=temperature, pressure=pressure)


def _per_compound(compounds, quantity, **conditions):
    """Evaluate a quantity of each compound along the last axis, broadcasting conditions.

    ``quantity(name, compound)`` returns the quantity as a function of conditions.
    """
    compounds = rx.io._check_compounds(compounds)
    shape = np.broadcast_shapes(*map(np.shape, conditions.values()), (len(compounds),))
    values = np.empty(shape)
    for i, (name, compound) in enumerate(compounds.items()):
        values[..., i] = np.vectorize(quantity(name, compound), otypes=[float])(
            **{
                key: np.broadcast_to(value, shape)[..., i]
                for key, value in conditions.items()
            }
        )
    return values


def _rrho(compound):
    """Return the rigid rotor harmonic oscillator arguments of a compound."""
    return {
        "energy": compound.energy,
        "degeneracy": compound.mult,
        "moments": coords.inertia(compound.atommasses, compound.atomcoords)[0],
        "vibfreqs": compound.vibfreqs,
    }


def _check_qrrho(
    qrrho: bool | tuple[bool, bool],
) -> tuple[bool, bool]:
    """Get options for QRRHO for both enthalpy and entropy.

    Parameters
    ----------
    qrrho : bool or tuple-like
        Apply both the quasi-rigid rotor harmonic oscillator (QRRHO)
        approximations of M. Head-Gordon and others (enthalpy correction, see
        [*J. Phys. Chem. C* **2015**, 119, 4, 1840-1850](http://dx.doi.org/10.1021/jp509921r))
        and S. Grimme (entropy correction, see
        [*Theory. Chem. Eur. J.*, **2012**, 18: 9955-9964](https://doi.org/10.1002/chem.201200497))
        on top of the classical RRHO.

    Returns
    -------
    qrrho_enthalpy, qrrho_entropy : bool

    Raises
    ------
    ValueError
        If the QRRHO specification is not understood.

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
        msg = f"unrecognized QRRHO specification: {qrrho}"
        raise ValueError(msg)


def get_freeenergies(
    compounds: dict,
    bias: float = 0.0,
    environment: str | None = None,
    method: str = "standard",
    qrrho: bool | tuple[bool, bool] = True,
    temperature: float = 298.15,
    pressure: float = constants.atm,
):
    """Obtain free energies for compounds at a given temperature and pressure.

    Each compound is in the standard state of its environment: 1 M in
    solution, and `pressure` in the gas phase.

    Parameters
    ----------
    compounds : dict-like
        A descriptor of the compounds.
        Mostly likely, this comes from a parsed input file.
        See `overreact.io.parse_model`.
    bias : array-like, optional
        Energy to be added to free energies.
    environment : str or None, optional
        Choose between "gas" and a solvent. This is chosen for you by default,
        based on the names of each compound. If given, all compounds will
        have the same behavior.
    method : str, optional
        This is a placeholder for future functionality.
        There are plans to implement more sophisticated methods for calculating
        entropies such as in
        [*Phys. Chem. Chem. Phys.*, **2019**, 21, 18920-18929](https://doi.org/10.1039/C9CP03226F)
        and
        [*J. Chem. Theory Comput.* **2019**, 15, 5, 3204-3214](https://doi.org/10.1021/acs.jctc.9b00214).
        Head over to the
        [discussions](https://github.com/geem-lab/overreact/discussions) if
        you're interested and would like to contribute.
        Leave this as "standard" for now.
    qrrho : bool or tuple-like, optional
        Apply both the quasi-rigid rotor harmonic oscillator (QRRHO)
        approximations of M. Head-Gordon and others (enthalpy correction, see
        [*J. Phys. Chem. C* **2015**, 119, 4, 1840-1850](http://dx.doi.org/10.1021/jp509921r))
        and S. Grimme (entropy correction, see
        [*Theory. Chem. Eur. J.*, **2012**, 18: 9955-9964](https://doi.org/10.1002/chem.201200497))
        on top of the classical RRHO.
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
    >>> from overreact import _constants as constants
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
        compounds,
        qrrho=qrrho_enthalpy,
        temperature=temperature,
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
    scheme: Scheme,
    compounds: dict | None = None,
    bias: float = 0.0,
    tunneling: str = "eckart",
    qrrho: bool | tuple[bool, bool] = True,
    scale: str = "l mol-1 s-1",
    temperature: float = 298.15,
    pressure: float = constants.atm,
    delta_freeenergies: float | None = None,
    molecularity: float | None = None,
    volume: float | None = None,
) -> float:
    r"""Obtain reaction rate constants for a given reaction scheme.

    Parameters
    ----------
    scheme : Scheme
        A descriptor of the reaction scheme.
        Mostly likely, this comes from a parsed input file.
        See `overreact.io.parse_model`.
    compounds : dict-like, optional
        A descriptor of the compounds.
        Mostly likely, this comes from a parsed input file.
        See `overreact.io.parse_model`.
    bias : array-like, optional
        Energy added to the free energy of every name, or of each name if one
        per name, also when `delta_freeenergies` are given.
    tunneling : str or None, optional
        Either "eckart", "wigner", or None for no tunneling correction.
    qrrho : bool or tuple-like, optional
        Apply both the quasi-rigid rotor harmonic oscillator (QRRHO)
        approximations of M. Head-Gordonand others (enthalpy correction, see
        [*J. Phys. Chem. C* **2015**, 119, 4, 1840-1850](http://dx.doi.org/10.1021/jp509921r))
        and S. Grimme (entropy correction, see
        [*Theory. Chem. Eur. J.*, **2012**, 18: 9955-9964](https://doi.org/10.1002/chem.201200497))
        on top of the classical RRHO.
    scale : str, optional
        Reaction rate units. Possible values are "cm3 mol-1 s-1",
        "l mol-1 s-1", "m3 mol-1 s-1", "cm3 particle-1 s-1", "mmHg-1 s-1" and
        "atm-1 s-1".
    temperature : array-like, optional
        Absolute temperature in Kelvin, the same for all compounds if computed
        from them.
    pressure : array-like, optional
        Reference gas pressure.
    delta_freeenergies : array-like, optional
        Activation free energies, used instead of computing them from
        `compounds`, in the standard states of `get_freeenergies` (see notes).
        For equilibria, the reverse half must be minus the forward half, within
        1e-5 J/mol.
    molecularity : array-like, optional
        Number of reactant molecules of each reaction, used to convert units
        to `scale`. Defaults to `Scheme.molecularity`.
    volume : float, optional
        Molar volume of gases in m³/mol, :math:`\frac{R T}{p}` by default.

    Returns
    -------
    array-like

    Raises
    ------
    TypeError
        If neither `compounds` nor `delta_freeenergies` are given.
    ValueError
        If there is not one delta free energy per reaction along the last
        axis, if temperatures differ between compounds or between the halves
        of an equilibrium, or if the reverse half of an equilibrium does not
        undo its forward half.

    Notes
    -----
    Free energies are in the standard state of each compound's environment,
    which its name gives (see `get_freeenergies`): 1 M in solution, and
    `pressure` in the gas phase. Gas-phase free energies are converted to
    1 M, so rate constants are molar before conversion to `scale`, and both
    ways of giving free energies agree:

    >>> model = rx.parse_model("data/tanaka1996/UMP2/cc-pVTZ/model.jk")
    >>> freeenergies = get_freeenergies(model.compounds)
    >>> get_k(model.scheme, model.compounds, tunneling=None)
    array([1.54e+07])
    >>> get_k(model.scheme, tunneling=None,
    ...       delta_freeenergies=model.scheme.activation_freeenergies(freeenergies))
    array([1.54e+07])

    Some symbols are accepted as alternatives in "scale": "M-1", "ml" and
    "torr-1" are understood as "l mol-1", "cm3" and "mmHg-1", respectively.

    Examples
    --------
    Below is an example of an estimate for the rate of methyl rotation in
    ethane (a trivial attempt to reproduce
    [*Science*, **2006**, 313, 5795, 1951-1955](https://doi.org/10.1126/science.1132178)).
    **How many turns it does per second?**

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

    The calculated value is off by less than 2% from the experimental value
    (:math:`\frac{1}{12 \times 10^{-12}} \text{s}^{-1} = 8.33 \times 10^{10} \text{s}^{-1}`).
    We use Eckart tunneling by default, but see the effect of changing it
    above.

    The units of the returned reaction rate constants can be selected for
    non-unimolecular processes. The following is an attempt to reproduce
    [*J Atmos Chem*, **1996** 23, 37-49](https://doi.org/10.1007/BF00058703) for
    the reaction of proton-withdrawal by a chloride radical from the methane
    molecule
    :math:`\ce{CH4 + \cdot Cl -> [H3C\cdots H\cdots Cl]^\ddagger -> H3C\cdot + HCl}`:

    >>> model = rx.parse_model("data/tanaka1996/UMP2/cc-pVTZ/model.jk")
    >>> get_k(model.scheme, model.compounds, temperature=300,
    ...       scale="cm3 particle-1 s-1")
    array([9.60e-14])

    (By the way, according to the Jet Propulsion Laboratory,
    [Publication No. 19-5](https://jpldataeval.jpl.nasa.gov/),
    the experimental reaction rate constant for this reaction is
    :math:`1.0 \times 10^{-13} \text{cm}^3 \text{particle}^{-1} \text{s}^{-1}`.)

    The returned units are "M-1 s-1" by default:

    >>> get_k(model.scheme, model.compounds) \
    ... == get_k(model.scheme, model.compounds, scale="l mol-1 s-1")
    array([ True])

    You can set a simple energy bias, either as a constant or compound-wise:

    >>> get_k(model.scheme, model.compounds, bias=1.0 * constants.kcal,
    ...       temperature=300.0, scale="cm3 particle-1 s-1")
    array([5.14e-13])
    >>> get_k(model.scheme, model.compounds,
    ...       bias=np.array([0.0, 0.0, 0.0, 0.0, -1.4]) * constants.kcal,
    ...       temperature=300.0, scale="cm3 particle-1 s-1")
    array([1.1e-12])
    """
    qrrho_enthalpy, qrrho_entropy = _check_qrrho(qrrho)
    scheme = rx.core._check_scheme(scheme)
    if compounds is not None:
        compounds = rx.io._check_compounds(compounds)
        compounds = {name: compounds[name] for name in scheme.names}
    if molecularity is None:
        molecularity = scheme.molecularity

    def activation_freeenergies():
        """Return activation free energies computed from compounds."""
        if np.shape(temperature)[-1:] not in {(), (1,)}:
            msg = "temperature must be the same for all compounds, e.g. of shape (m, 1)"
            raise ValueError(msg)
        logger.info(
            "activation symmetry entropies: "
            f"{rx._misc.format_numbers(scheme.activation_entropies)} J·mol⁻¹·K⁻¹",
        )
        return scheme.activation_freeenergies(
            get_freeenergies(
                compounds,
                qrrho=(qrrho_enthalpy, qrrho_entropy),
                temperature=temperature,
                pressure=pressure,
            ),
            temperature=temperature,
        )

    def tunneling_coefficients():
        """Return tunneling coefficients, or ones if they can't be computed."""
        if tunneling is None:
            return 1.0
        if compounds is None:
            # TODO(schneiderfelipe): when get_kappa accept deltas, this will
            # be probably unnecessary.
            logger.warning(
                "assuming unitary tunneling coefficients due to incomplete compound data",
            )
            return 1.0
        return get_kappa(
            scheme,
            compounds,
            tunneling=tunneling,
            qrrho=qrrho_enthalpy,
            temperature=temperature,
        )

    if delta_freeenergies is None:
        if compounds is None:
            msg = "either compounds or delta_freeenergies must be given"
            raise TypeError(msg)
        delta_freeenergies = activation_freeenergies()

    delta_freeenergies, temperatures = np.broadcast_arrays(
        np.array(delta_freeenergies, dtype=float, ndmin=1), temperature
    )
    if delta_freeenergies.shape[-1] != len(scheme.reactions):
        msg = f"expected {len(scheme.reactions)} delta free energies, one per reaction, along the last axis, got shape {delta_freeenergies.shape}"
        raise ValueError(msg)
    forward, reverse = np.array(scheme.equilibria, dtype=int).reshape(-1, 2).T
    for problem, mismatch in [
        (
            "both halves must have the same temperature",
            temperatures[..., forward] != temperatures[..., reverse],
        ),
        (
            "the reverse half must undo its forward half",
            ~np.isclose(
                delta_freeenergies[..., reverse],
                -delta_freeenergies[..., forward],
                rtol=0.0,
                atol=1e-5,
                equal_nan=True,
            ),
        ),
    ]:
        if mismatch.any():
            msg = f"in equilibrium '{scheme.reactions[forward[np.nonzero(mismatch)[-1][0]]]}', {problem}"
            raise ValueError(msg)

    # add biases, and take gas-phase free energies from `pressure` to 1 M
    delta_freeenergies = (
        delta_freeenergies
        + scheme.activation_deltas(
            np.broadcast_to(bias, (*np.shape(bias)[:-1], len(scheme.names)))
        )
        + temperatures
        * thermo.change_reference_state(
            temperature=temperatures, pressure=pressure, volume=volume
        )
        * scheme.activation_deltas(
            [rx.core._get_environment(name) == "gas" for name in scheme.names]
        )
    )

    k = np.empty(delta_freeenergies.shape)
    plain = ~np.array(scheme._is_half_equilibrium, dtype=bool)
    k[..., plain] = rates.eyring(
        delta_freeenergies[..., plain], temperature=temperatures[..., plain]
    )

    # equilibria only fix ratios: make the smallest rate constant of each pair one
    equilibrium_constants = thermo.equilibrium_constant(
        delta_freeenergies[..., forward], temperature=temperatures[..., forward]
    )
    denom = np.minimum(equilibrium_constants, 1.0)
    zero = denom == 0.0
    if zero.any():
        logger.warning(
            "found half-equilibrium reaction with zero rate constant: skipping equilibrium normalization",
        )
    denom[zero] = 1.0
    k[..., forward], k[..., reverse] = equilibrium_constants / denom, 1.0 / denom
    logger.info(
        f"(classical) reaction rate constants: {rx._misc.format_numbers(k)} M⁻ⁿ⁺¹·s⁻¹",
    )

    k = k * tunneling_coefficients()
    logger.info(
        f"(tunneling) reaction rate constants: {rx._misc.format_numbers(k)} M⁻ⁿ⁺¹·s⁻¹",
    )

    # TODO(schneiderfelipe): ensure diffusional limit for reactions in
    # solvation using Collins-Kimball theory. This includes half-equilibria.
    return rates.convert_rate_constant(
        k,
        new_scale=scale,
        old_scale="l mol-1 s-1",
        molecularity=molecularity,
        temperature=temperature,
    )


# TODO(schneiderfelipe): accept deltas and make compounds optional.
def get_kappa(
    scheme: Scheme,
    compounds: dict,
    tunneling: str | None = "eckart",
    qrrho: bool = True,
    temperature: float = 298.15,
):
    r"""Obtain tunneling transmission coefficients at a given temperature.

    One tunneling transmission coefficient is calculated for each reaction. If
    a reaction lacks a transition state (i.e., a half-equilibrium reaction),
    its transmission coefficient is set to unity.

    Parameters
    ----------
    scheme : Scheme
        A descriptor of the reaction scheme.
        Mostly likely, this comes from a parsed input file.
        See `overreact.io.parse_model`.
    compounds : dict-like
        A descriptor of the compounds.
        Mostly likely, this comes from a parsed input file.
        See `overreact.io.parse_model`.
    tunneling : str or None, optional
        Either "eckart", "wigner", or None for no tunneling correction.
    qrrho : bool, optional
        Apply both the quasi-rigid rotor harmonic oscillator (QRRHO)
        approximations of M. Head-Gordon and others (enthalpy correction, see
        [*J. Phys. Chem. C* **2015**, 119, 4, 1840-1850](http://dx.doi.org/10.1021/jp509921r))
        and S. Grimme (entropy correction, see
        [*Theory. Chem. Eur. J.*, **2012**, 18: 9955-9964](https://doi.org/10.1002/chem.201200497))
        on top of the classical RRHO.
    temperature : array-like, optional
        Absolute temperature in Kelvin.

    Returns
    -------
    array-like

    Raises
    ------
    ValueError
        If `tunneling` is not supported.

    Examples
    --------
    Below is an example of an estimate of how much quantum tunneling
    contributes to the rate of methyl rotation in ethane (see
    [*Science*, **2006**, 313, 5795, 1951-1955](https://doi.org/10.1126/science.1132178)
    for some interesting experimental data on this reaction).

    >>> model = rx.parse_model("data/ethane/B97-3c/model.k")
    >>> kappa = get_kappa(model.scheme, model.compounds)
    >>> kappa
    array([1.110])
    >>> get_kappa(model.scheme, model.compounds, tunneling=None)
    array([1.0])

    You can calculate each piece of the reaction rate constant by hand,
    if you want. Just make sure that you don't calculate the tunneling
    coefficient twice:

    >>> kappa * get_k(model.scheme, model.compounds, tunneling=None)
    array([8.e+10])
    """
    if tunneling not in {"eckart", "wigner", None}:
        msg = f"unavailable tunneling correction: {tunneling!r}"
        raise ValueError(msg)
    scheme = rx.core._check_scheme(scheme)
    compounds = rx.io._check_compounds(compounds)
    compounds = {name: compounds[name] for name in scheme.names}

    if tunneling == "eckart":
        # NOTE(schneiderfelipe): We need electronic energies + ZPE here, so we
        # get smaller transmission coefficients.
        energies = get_enthalpies(compounds, qrrho=qrrho, temperature=0.0)
        delta_forward = scheme.activation_deltas(energies)
        delta_backward = delta_forward - scheme.reaction_deltas(energies)

    def eckart(vibfreq, delta_forward, delta_backward, temperature):
        """Return the Eckart coefficient, or the Wigner one where Eckart fails."""
        with warnings.catch_warnings(), np.errstate(invalid="warn"):
            warnings.filterwarnings("error")
            try:
                return tunnel.eckart(
                    vibfreq,
                    delta_forward,
                    delta_backward,
                    temperature=temperature,
                )
            except RuntimeWarning as e:
                logger.warning(
                    f"using Wigner tunneling correction: {e}",
                )
                return tunnel.wigner(vibfreq, temperature=temperature)

    temperatures = np.broadcast_to(
        temperature,
        np.broadcast_shapes(np.shape(temperature), (len(scheme.reactions),)),
    )
    kappas = np.ones(temperatures.shape)
    for i, ts in enumerate(scheme.transition_states):
        if ts is None:
            continue
        vibfreq = compounds[ts].vibfreqs[0]

        if tunneling == "eckart":
            # np.vectorize reports floating-point flags that eckart already handled
            with np.errstate(invalid="ignore"):
                kappas[..., i] = np.vectorize(
                    partial(eckart, vibfreq, delta_forward[i], delta_backward[i]),
                    otypes=[float],
                )(temperatures[..., i])
        elif tunneling == "wigner":
            kappas[..., i] = tunnel.wigner(vibfreq, temperature=temperatures[..., i])

    logger.info(
        f"(quantum) tunneling coefficients: {rx._misc.format_numbers(kappas)}",
    )
    return kappas


def get_drc(
    scheme,
    compounds,
    y0,
    t_span=None,
    method="RK23",
    qrrho=True,
    scale="l mol-1 s-1",
    temperature=298.15,
    dx=1.5e3,  # joules
    order=3,
):
    """Calculate the degree of rate control for a single compound.

    Notes
    -----
    **This is a work in progress!**

    It is a good starting point for sensitivity analyses in general.
    Head over to the
    [discussions](https://github.com/geem-lab/overreact/discussions) if
    you're interested and would like to contribute.

    Examples
    --------
    >>> model = rx.parse_model("data/tanaka1996/UMP2/cc-pVTZ/model.jk")
    """
    temperature = np.asarray(temperature)

    x0 = np.zeros(len(scheme.names))

    def func(t, x=0.0, i=-1):
        bias = np.asarray(x0).copy()
        bias[i] += x

        k = get_k(
            scheme,
            compounds=compounds,
            bias=bias,
            qrrho=qrrho,
            scale=scale,
            temperature=temperature,
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
