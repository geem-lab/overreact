#!/usr/bin/env python3  # noqa: EXE001

"""Module dedicated to the calculation of thermodynamic properties in gas phase."""

import logging

import numpy as np

from overreact import _constants as constants
from overreact import coords

logger = logging.getLogger(__name__)


def calc_trans_energy(temperature=298.15):
    r"""Calculate the translational energy of an ideal gas.

    Take a look at <https://socratic.org/questions/5715a4e711ef6b17257e0033>.

    Parameters
    ----------
    temperature : array-like, optional
        Absolute temperature in Kelvin.

    Returns
    -------
    float
        Translational energy in J/mol.

    Examples
    --------
    >>> calc_trans_energy()
    3718.
    >>> calc_trans_energy(373.15)
    4653.

    """
    temperature = np.asarray(temperature)

    translational_energy = 1.5 * constants.R * temperature
    logger.info(f"translational energy = {translational_energy} J/mol")  # noqa: G004
    return translational_energy


def calc_elec_energy(energy=0.0, degeneracy=1, temperature=298.15):
    """Calculate electronic energy.

    This eventually adds a correction based on accessible excited states, but
    simply return the ground state energy if only information on the ground
    state is given.

    Parameters
    ----------
    energy : array-like, optional
        Energies for different states, in J/mol.
    degeneracy : array-like, optional
        Degeneracies of the states of the molecule. This is normally the same
        as spin multiplicity, but might be total angular momentum degeneracy in
        some cases (e.g., fluorine).
    temperature : array-like, optional
        Absolute temperature in Kelvin.

    Returns
    -------
    float
        Electronic energy in J/mol.

    Examples
    --------
    Normally, only the ground state is considered. In this case, only the
    energy and degeneracy of the ground state are needed. The degeneracy
    normally means spin multiplicity in this case. In fact, the energy in this
    case is independent of the ground state degeneracy and, if not given,
    non-degenerate is assumed:

    >>> calc_elec_energy(1234.0, degeneracy=4)
    1234.0
    >>> np.isclose(calc_elec_energy(1234.0, 3), calc_elec_energy(1234.0, 2))
    True
    >>> calc_elec_energy(1234.0)  # singlet by default
    1234.0

    But you can consider excited states too. Below is a calculation with
    degeneracies and excitation energies for the fluorine atom as taken from
    the NIST atomic data
    (<https://physics.nist.gov/PhysRefData/Handbook/Tables/fluorinetable5.htm>).
    Similar data can also be found in Table 5-1 of Statistical Thermodynamics,
    McQuarrie (ground energy is assumed zero to simplify things):

    >>> j = np.array([3/2, 1/2, 5/2, 3/2, 1/2, 3/2, 1/2])
    >>> degeneracy = 2 * j + 1
    >>> energy = np.array([0.000, 404.141, 102405.714, 102680.439,  # cm-1
    ...                    102840.378, 104731.048, 105056.283])
    >>> calc_elec_energy(
    ...     energy * 100 * constants.h * constants.c * constants.N_A, degeneracy
    ... )
    321.00

    """
    temperature = np.asarray(temperature)

    min_energy = np.asarray(energy).min()
    if np.isclose(temperature, 0.0):
        logger.warning("assuming ground state as electronic energy at zero temperature")
        return min_energy

    energy = energy - min_energy
    q_elec_terms = degeneracy * np.exp(-energy / (constants.R * temperature))
    q_elec = np.sum(q_elec_terms)

    electronic_energy = min_energy + np.sum(energy * q_elec_terms) / q_elec
    logger.info(f"electronic energy = {electronic_energy} J/mol")  # noqa: G004
    return electronic_energy


def calc_elec_entropy(energy=0.0, degeneracy=1, temperature=298.15):
    """Calculate electronic entropy.

    This eventually adds a correction based on accessible excited states, but
    simply return a term based on the the ground state degeneracy if only
    information on the ground state is given.

    Parameters
    ----------
    energy : array-like, optional
        Energies for different states, in J/mol.
    degeneracy : array-like, optional
        Degeneracies of the states of the molecule. This is normally the same
        as spin multiplicity, but might be total angular momentum degeneracy in
        some cases (e.g., fluorine).
    temperature : array-like, optional
        Absolute temperature in Kelvin.

    Returns
    -------
    float
        Electronic entropy in J/mol·K.

    Examples
    --------
    Normally, only the ground state is considered. In this case, only the
    energy and degeneracy of the ground state are needed. The degeneracy
    normally means spin multiplicity in this case. In fact, the entropy in this
    case is independent of the ground state energy and, if not given, zero
    energy is assumed:

    >>> calc_elec_entropy(degeneracy=4)
    11.526
    >>> np.isclose(calc_elec_entropy(degeneracy=3),
    ...            calc_elec_entropy(1234.0, 3))
    True
    >>> np.isclose(calc_elec_entropy(0.0, 2), calc_elec_entropy(1234.0, 2))
    True
    >>> calc_elec_entropy(1234.0, 2)
    5.763
    >>> calc_elec_entropy()  # singlet by default
    0.0
    >>> calc_elec_entropy(1234.0)  # singlet by default
    0.0

    But you can consider excited states too. Below is a calculation with
    degeneracies and excitation energies for the fluorine atom as taken from
    the NIST atomic data
    (<https://physics.nist.gov/PhysRefData/Handbook/Tables/fluorinetable5.htm>).
    Similar data can also be found in Table 5-1 of Statistical Thermodynamics,
    McQuarrie (ground energy is assumed zero to simplify things):

    >>> j = np.array([3/2, 1/2, 5/2, 3/2, 1/2, 3/2, 1/2])
    >>> degeneracy = 2 * j + 1
    >>> energy = np.array([0.000, 404.141, 102405.714, 102680.439,  # cm-1
    ...                    102840.378, 104731.048, 105056.283])
    >>> calc_elec_entropy(energy * 100 * constants.h * constants.c * constants.N_A,
    ...                   degeneracy)
    13.175

    """
    temperature = np.asarray(temperature)

    if np.isclose(temperature, 0.0):
        logger.warning("assuming electronic entropy zero at zero temperature")
        return 0.0

    min_energy = np.asarray(energy).min()
    energy = energy - min_energy

    q_elec_terms = degeneracy * np.exp(-energy / (constants.R * temperature))
    q_elec = np.sum(q_elec_terms)

    electronic_entropy = constants.R * np.log(q_elec) + np.sum(
        energy * q_elec_terms,
    ) / (temperature * q_elec)
    logger.info(f"electronic entropy = {electronic_entropy} J/mol·K")  # noqa: G004
    return electronic_entropy


def calc_rot_energy(
    moments=None,
    independent=False,  # noqa: FBT002
    weights=1.0,
    temperature=298.15,
):  # noqa: RUF100
    r"""Calculate the rotational energy of an ideal gas.

    This function uses the truncation of equation 6-48 of Statistical
    Thermodynamics, McQuarrie, which is valid for temperatures larger than five
    times the rotational temperatures and correct up to second order in the
    inverse temperature. Observe that this is more precise than equation 8-20
    of the same book.

    Parameters
    ----------
    moments : array-like
        Primary moments of inertia in ascending order. Units are in amu·Å².
    independent : bool, optional
        Whether rotational degrees of freedom should be considered independent
        (this is to be mainly used in the quasi-RRHO approach of M.
        Head-Gordon, see
        [*J. Phys. Chem. C* **2015**, 119, 4, 1840-1850](http://dx.doi.org/10.1021/jp509921r)).
    weights : array-like, optional
    temperature : array-like, optional
        Absolute temperature in Kelvin.

    Returns
    -------
    float
        Rotational energy in J/mol.

    Examples
    --------
    >>> i = 8.53818341e1
    >>> calc_rot_energy([i, i])
    2479.
    >>> np.allclose(calc_rot_energy(i), calc_rot_energy([i]))
    True
    >>> ia = (constants.hbar ** 2 / (2.0 * constants.k * 13.6)) \
    ...     / (constants.atomic_mass * constants.angstrom ** 2)
    >>> ib = (constants.hbar ** 2 / (2.0 * constants.k * 8.92)) \
    ...     / (constants.atomic_mass * constants.angstrom ** 2)
    >>> calc_rot_energy([ia, ib, ib])  # NH3
    3674.

    If no moments of inertia are given, a monoatomic gas is assumed:

    >>> calc_rot_energy()
    0.0

    """
    temperature = np.asarray(temperature)

    if np.isclose(temperature, 0.0):
        logger.warning("assuming rotational energy zero at zero temperature")
        return 0.0

    rotational_temperatures = _rotational_temperature(moments)
    if not rotational_temperatures.size:
        logger.warning("assuming zero rotational energy for atomic system")
        return 0.0

    if not independent and np.any(rotational_temperatures >= 0.2 * temperature):
        logger.warning(
            f"rotational temperatures probably too high for {temperature} K: "  # noqa: E501, G004
            f"{rotational_temperatures[rotational_temperatures >= 0.2 * temperature]}",
        )

    n = len(rotational_temperatures) if not independent else 1.0
    gamma = (
        np.sum(weights * n)
        - np.sum(weights * rotational_temperatures) / (3.0 * temperature)  # extra
        - np.sum(weights * rotational_temperatures**2)
        / (45.0 * temperature**2)  # extra
    )

    rotational_energy = constants.R * temperature * gamma / 2.0
    if not independent:
        logger.info(f"rotational energy = {rotational_energy} J/mol")  # noqa: G004
    return rotational_energy


def calc_rot_entropy(  # noqa: PLR0913
    atommasses=None,
    atomnos=None,
    atomcoords=None,
    moments=None,
    symmetry_number=1,
    environment="gas",
    method="standard",
    independent=False,  # noqa: FBT002
    weights=1.0,
    temperature=298.15,
    pressure=constants.atm,
):
    r"""Calculate the rotational entropy of an ideal gas.

    This function incorporates the truncation of equation 6-50 of Statistical
    Thermodynamics, McQuarrie, which is valid for temperatures larger than five
    times the rotational temperatures and correct up to second order in the
    inverse temperature, and equation 8-22 of the same book. This means this
    function has an extra term for each rotational degree of freedom that is
    exact for diatomics and approximate (?) for polyatomic molecules.

    For the liquid phase, the extra terms in the method of A. Garza are summed
    (doi:10.1021/acs.jctc.9b00214). This should be used together with
    ``method="garza"`` in `calc_trans_entropy`. I may implement the "izato"
    method for rotational entropy in the future (doi:10.1039/C9CP03226F), but
    this is *not* currently available.

    Parameters
    ----------
    atommasses : array-like
        Atomic masses or molecular mass in atomic mass units (amu).
    atomnos : array-like, optional
    atomcoords : array-like, optional
        Atomic coordinates.
    moments : array-like
        Primary moments of inertia in ascending order. Units are in amu·Å².
    symmetry_number : int, optional
    environment : str, optional
        Choose between "gas" and a solvent.
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
    independent : bool, optional
        Whether rotational degrees of freedom should be considered independent
        (this is to be mainly used in the quasi-RRHO approach of S. Grimme, see
        [*Theory. Chem. Eur. J.*, **2012**, 18: 9955-9964](https://doi.org/10.1002/chem.201200497)).
    weights : array-like, optional
    temperature : array-like, optional
        Absolute temperature in Kelvin.
    pressure : array-like, optional
        Reference gas pressure.

    Returns
    -------
    float
        Rotational entropy in J/mol·K.

    Raises
    ------
    ValueError
        If environment is "solid".

    Examples
    --------
    >>> i = (constants.hbar**2 / (2.0 * constants.k * 15.02)) \
    ...     / (constants.atomic_mass * constants.angstrom ** 2)
    >>> calc_rot_entropy(moments=[i, i])  # HCl
    33.16
    >>> i = (constants.hbar**2 / (2.0 * constants.k * 87.6)) \
    ...     / (constants.atomic_mass * constants.angstrom ** 2)
    >>> calc_rot_entropy(moments=[i, i], symmetry_number=2)  # H2
    12.73
    >>> ia = (constants.hbar ** 2 / (2.0 * constants.k * 13.6)) \
    ...     / (constants.atomic_mass * constants.angstrom ** 2)
    >>> ib = (constants.hbar ** 2 / (2.0 * constants.k * 8.92)) \
    ...     / (constants.atomic_mass * constants.angstrom ** 2)
    >>> calc_rot_entropy(moments=[ia, ib, ib], symmetry_number=3)  # NH3
    50.11

    If no moments of inertia are given, an ideal monoatomic gas is assumed:

    >>> calc_rot_entropy()
    0.0

    >>> from overreact import _datasets as datasets

    >>> data = datasets.logfiles["symmetries"]["water"]
    >>> moments, axes, atomcoords = coords.inertia(data.atommasses, data.atomcoords)
    >>> calc_rot_entropy(moments=moments)
    50.03250369082377
    >>> calc_rot_entropy(moments=moments, environment="water",
    ...                  atommasses=data.atommasses, atomnos=data.atomnos,
    ...                  atomcoords=data.atomcoords)
    50.03250369082377
    >>> calc_rot_entropy(moments=moments, environment="water", method="garza",
    ...                  atommasses=data.atommasses, atomnos=data.atomnos,
    ...                  atomcoords=data.atomcoords)
    47.1
    """
    temperature = np.asarray(temperature)

    if np.isclose(temperature, 0.0):
        logger.warning("assuming rotational entropy zero at zero temperature")
        return 0.0

    rotational_temperatures = _rotational_temperature(moments)
    if not rotational_temperatures.size:
        logger.warning("assuming zero rotational entropy for atomic system")
        return 0.0

    if not independent and np.any(rotational_temperatures >= 0.2 * temperature):
        logger.warning(
            f"rotational temperatures probably too high for {temperature} K: "  # noqa: E501, G004
            f"{rotational_temperatures[rotational_temperatures >= 0.2 * temperature]}",
        )

    n = len(rotational_temperatures) if not independent else 1.0
    gamma = (
        np.sum(weights * n * (1.0 + np.log(temperature)))
        - np.sum(weights * np.log(rotational_temperatures))
        - 2.0 * np.log(symmetry_number)
        # The term below is an extra term that improves some linear molecules,
        # but is almost zero for most other molecules
        - np.sum(weights * rotational_temperatures**2) / (90.0 * temperature**2)
    )
    if not independent and n > 2:  # noqa: PLR2004
        gamma += np.log(np.pi)

    rotational_entropy = constants.R * gamma / 2.0
    if environment in {"gas", None} or method == "standard":
        pass
    elif environment == "solid":
        raise ValueError(  # noqa: TRY003
            f"environment not yet implemented: {environment}",  # noqa: EM102
        )  # noqa: RUF100
    else:
        assert atomnos is not None, "atomnos must be given"
        assert atomcoords is not None, "atomcoords must be given"
        vdw_volume = coords.get_molecular_volume(atomnos, atomcoords)
        cav_volume, N_cav, _ = coords._garza(  # noqa: N806, SLF001
            vdw_volume,
            environment,
            full_output=True,
            temperature=temperature,
            pressure=pressure,
        )

        r_cav = np.cbrt(3.0 * cav_volume / (4.0 * np.pi))
        r_g = coords.gyradius(atommasses, atomcoords)
        prefactor = N_cav * 4.0 * np.pi / 3

        rotational_entropy = (
            rotational_entropy
            + _sackur_tetrode(atommasses, prefactor * (r_cav - r_g) ** 3, temperature)
            - _sackur_tetrode(atommasses, prefactor * r_cav**3, temperature)
        )
    if not independent:
        logger.info(f"rotational entropy = {rotational_entropy} J/mol·K")  # noqa: G004
    return rotational_entropy


def calc_vib_energy(vibfreqs=None, qrrho=True, temperature=298.15):  # noqa: FBT002
    r"""Calculate the vibrational energy of an ideal gas.

    This function uses equation 8-7 of Statistical Thermodynamics, McQuarrie,
    which also includes the zero point energy (ZPE). In case you want to
    calculate only the ZPE, set the temperature to zero.

    Parameters
    ----------
    vibfreqs : array-like
        Frequency magnitudes in cm-1.
    qrrho : bool, optional
        Apply the quasi-rigid rotor harmonic oscilator (QRRHO) approximation of
        M. Head-Gordon and others (see
        [*J. Phys. Chem. C* **2015**, 119, 4, 1840-1850](http://dx.doi.org/10.1021/jp509921r)) on top of the classical
        RRHO.
    temperature : array-like, optional
        Absolute temperature in Kelvin.

    Returns
    -------
    float
        Vibrational energy in J/mol.

    Examples
    --------
    >>> calc_vib_energy(3374 \
    ...     * constants.k * constants.centi \
    ...     / (constants.h * constants.c))  # nitrogen molecule
    14026.
    >>> calc_vib_energy(310 \
    ...     * constants.k * constants.centi / (constants.h * constants.c),
    ...                 temperature=1000)  # elemental iodine
    8.e3

    >>> vibfreqs = np.array([3360, 954, 954, 1890]) \
    ...     * constants.k * constants.centi / (constants.h * constants.c)
    >>> calc_vib_energy(vibfreqs)  # CO2
    3.045e4

    The following zero point energy (ZPE) of CO2 agrees with a calculation at
    CCSD(T)/aug-cc-pVDZ:

    >>> calc_vib_energy(vibfreqs, temperature=0.0) / constants.kcal
    7.1

    If no frequencies are given, an ideal monoatomic gas is assumed:

    >>> calc_vib_energy()
    0.0

    """  # noqa: E501
    vibrational_temperature = _vibrational_temperature(vibfreqs)
    if not vibrational_temperature.size:
        logger.warning("assuming zero vibrational energy for atomic system")
        return 0.0

    weights = _head_gordon_damping(vibfreqs) if qrrho else 1.0

    # the zero point energy (ZPE) is given below
    gamma = np.sum(weights * vibrational_temperature) / 2.0

    if np.isclose(temperature, 0.0):
        logger.warning("assuming zero point as vibrational energy at zero temperature")
    else:
        energy_fraction = vibrational_temperature / temperature
        gamma += np.sum(
            weights * vibrational_temperature / (np.exp(energy_fraction) - 1.0),
        )

    vibrational_energy = constants.R * gamma
    if qrrho:
        vibrational_energy += calc_rot_energy(
            moments=_vibrational_moment(vibfreqs),
            independent=True,
            weights=1.0 - weights,
            temperature=temperature,
        )
    logger.info(f"vibrational energy = {vibrational_energy} J/mol")  # noqa: G004
    return vibrational_energy


# TODO(schneiderfelipe): construct corrections using anharmonicity (probably
# using corrections from a Morse potential). See also problem 6-24 of
# Statistical Thermodynamics, McQuarrie.
def calc_vib_entropy(vibfreqs=None, qrrho=True, temperature=298.15):  # noqa: FBT002
    r"""Calculate the vibrational entropy of an ideal gas.

    This function calculates the third and fourth terms of equation 6-54 of
    Statistical Thermodynamics, McQuarrie.

    Parameters
    ----------
    vibfreqs : array-like
        Frequency magnitudes in cm-1.
    qrrho : bool, optional
        Apply the quasi-rigid rotor harmonic oscilator (QRRHO) approximation of
        S. Grimme (see
        [*Theory. Chem. Eur. J.*, **2012**, 18: 9955-9964](https://doi.org/10.1002/chem.201200497)) on top of the classical
        RRHO.
    temperature : array-like, optional
        Absolute temperature in Kelvin.

    Returns
    -------
    float
        Vibrational entropy in J/mol·K.

    Examples
    --------
    >>> import overreact as rx

    >>> calc_vib_entropy(3374 \
    ...     * constants.k * constants.centi \
    ...     / (constants.h * constants.c))  # nitrogen molecule
    0.0013
    >>> vibfreqs = np.array([3360, 954, 954, 1890]) \
    ...     * constants.k * constants.centi / (constants.h * constants.c)
    >>> calc_vib_entropy(vibfreqs)  # CO2
    3.06

    A molecule can be loaded and its vibrational entropy calculated right away:

    >>> data = rx.io.read_logfile("data/hickel1992/UM06-2X/6-311++G(d,p)/NH3·OH.out")
    >>> 298.15 * calc_vib_entropy(data.vibfreqs) / constants.kcal
    1.89

    If no frequencies are given, an ideal monoatomic gas is assumed:

    >>> calc_vib_entropy()
    0.0

    """  # noqa: E501
    if np.isclose(temperature, 0.0):
        logger.warning("assuming vibrational entropy zero at zero temperature")
        return 0.0

    # TODO(schneiderfelipe): should we use _check_vibfreqs here and remove from
    # _vibrational_temperature and _head_gordon_damping?
    vibrational_temperature = _vibrational_temperature(vibfreqs)
    if not vibrational_temperature.size:
        logger.warning("assuming zero vibrational entropy for atomic system")
        return 0.0

    weights = _head_gordon_damping(vibfreqs) if qrrho else 1.0

    energy_fraction = vibrational_temperature / temperature
    gamma = np.sum(
        weights
        * (
            energy_fraction / (np.exp(energy_fraction) - 1.0)
            - np.log(1.0 - np.exp(-energy_fraction))
        ),
    )

    vibrational_entropy = constants.R * gamma
    if qrrho:
        vibrational_entropy += calc_rot_entropy(
            moments=_vibrational_moment(vibfreqs),
            independent=True,
            weights=1.0 - weights,
            temperature=temperature,
        )
    logger.info(f"vibrational entropy = {vibrational_entropy} J/mol·K")  # noqa: G004
    return vibrational_entropy


def _sackur_tetrode(atommasses, volume, temperature=298.15):
    """Calculate the Sackur-Tetrode equation.

    Parameters
    ----------
    atommasses : array-like
        Atomic masses or molecular mass in atomic mass units (amu).
    volume : float
        Molar volume.
    temperature : array-like, optional
        Absolute temperature in Kelvin.

    Returns
    -------
    float

    Examples
    --------
    >>> _sackur_tetrode(18.01528, 0.0993e-30 * constants.N_A)  # water est. free volume
    37.36
    """
    temperature = np.asarray(temperature)

    total_mass = np.sum(atommasses) * constants.atomic_mass
    debroglie_wavelength = constants.h / np.sqrt(
        2.0 * np.pi * total_mass * constants.k * temperature,
    )
    q_trans = volume / (constants.N_A * debroglie_wavelength**3)
    assert q_trans > 1.0, (  # noqa: PLR2004
        f"de Broglie wavelength {debroglie_wavelength} is too large for the gas to "
        "satisfy Maxwell-Boltzmann statistics (classical regime)"
    )
    return constants.R * (np.log(q_trans) + 2.5)


def _rotational_temperature(moments=None):
    """Calculate rotational temperatures.

    This function returns rotational temperatures associated with all non-zero
    moments of inertia.

    Parameters
    ----------
    moments : array-like
        Primary moments of inertia in ascending order. Units are in amu·Å².

    Returns
    -------
    array-like

    Examples
    --------
    >>> i = 2.96199592e-1
    >>> _rotational_temperature()
    array([], ...)
    >>> _rotational_temperature(i)
    array([81.88521438])
    >>> _rotational_temperature([i])
    array([81.88521438])
    >>> _rotational_temperature([i, i])
    array([81.88521438, 81.88521438])
    >>> _rotational_temperature([0.0, i])
    array([81.88521438])
    >>> _rotational_temperature([0.0, 0.0, i])
    array([81.88521438])
    >>> _rotational_temperature([0.0, i, i])
    array([81.88521438, 81.88521438])
    >>> _rotational_temperature([i, i, i])
    array([81.88521438, 81.88521438, 81.88521438])
    """
    if moments is None:
        # assuming atomic system
        return np.array([])
    moments = np.atleast_1d(moments)
    moments[
        np.abs(moments) < 1e-63  # noqa: PLR2004
    ] = 0  # set almost zeros to exact zeros  # noqa: PLR2004, RUF100
    moments = (
        moments[np.nonzero(moments)] * constants.atomic_mass * constants.angstrom**2
    )

    rotational_temperatures = constants.hbar**2 / (2.0 * constants.k * moments)
    logger.debug(f"rotational temperatures = {rotational_temperatures} K")  # noqa: G004
    return rotational_temperatures


def _vibrational_temperature(vibfreqs=None):
    r"""Calculate vibrational temperatures.

    This function returns vibrational temperatures associated with all positive
    frequency magnitudes. This function ignores negative frequencies.

    Parameters
    ----------
    vibfreqs : array-like
        Frequency magnitudes in cm-1.

    Returns
    -------
    array-like

    Examples
    --------
    >>> vibfreq = 3374 * constants.k * constants.centi \
    ...     / (constants.h * constants.c)  # nitrogen molecule
    >>> _vibrational_temperature()
    array([], ...)
    >>> _vibrational_temperature(vibfreq)
    array([3374.])
    >>> _vibrational_temperature([vibfreq])
    array([3374.])
    >>> _vibrational_temperature([vibfreq, vibfreq])
    array([3374., 3374.])

    This function uses `_check_vibfreqs` and as such employs its default
    cutoff for imaginary frequencies. This means that positive values are used
    as expected, but small non-positive values are inverted (sign flipped):

    >>> _vibrational_temperature([-60.0, vibfreq])
    array([3374.])
    >>> _vibrational_temperature([-60.0, -60.0, vibfreq])
    array([3374.])
    >>> _vibrational_temperature([-60.0, vibfreq, vibfreq])
    array([3374., 3374.])
    >>> _vibrational_temperature([vibfreq, vibfreq, vibfreq])
    array([3374., 3374., 3374.])
    """
    nu = _check_vibfreqs(vibfreqs) * constants.c / constants.centi

    vibrational_temperatures = constants.h * nu / constants.k
    logger.debug(
        f"vibrational temperatures = {vibrational_temperatures} K",  # noqa: G004
    )
    return vibrational_temperatures


def _check_vibfreqs(vibfreqs=None, cutoff=-50.0):
    """Check vibrational frequencies and return them.

    This is mostly a helper to `_vibrational_temperature` and
    `_head_gordon_damping`.

    Parameters
    ----------
    vibfreqs : array-like
        Frequency magnitudes in cm-1.
    cutoff : float
        Imaginary frequencies smaller (in magnitude) than this value will be
        inverted (sign flipped).

    Returns
    -------
    array-like

    Notes
    -----
    We probably profit from QRRHO in cases where small imaginary vibrational
    frequencies are present. Obviously, for this to make sense in a the QRRHO
    context, values are only acceptable when smaller than ~100 cm-1. More
    reasonably, values should be smaller than ~50 cm-1. This is the current
    default. It is your responsability to ensure reliable values, which
    broadly means no imaginary frequency larger than this cutoff (unless we
    are dealing with transition states).

    Examples
    --------
    >>> _check_vibfreqs()
    array([], ...)
    >>> _check_vibfreqs(100.0)
    array([100.])
    >>> _check_vibfreqs([5.0, 10.0])
    array([ 5., 10.])
    >>> _check_vibfreqs([-5.0, 15.0, 100.0])
    array([  5.,  15., 100.])
    >>> _check_vibfreqs([-55.0, 15.0, 100.0])
    array([ 15., 100.])
    >>> _check_vibfreqs([-55.0, -5.0, 15.0, 100.0])
    array([  5.,  15., 100.])
    """
    if vibfreqs is None:
        return np.array([])
    vibfreqs = np.atleast_1d(vibfreqs)

    if len(vibfreqs[vibfreqs < 0]) > 0:
        logger.warning(
            f"imaginary frequencies found: using the absolute value of all above {-cutoff}i cm-1, ignoring the rest",  # noqa: E501, G004
        )

    return np.abs(vibfreqs[vibfreqs > cutoff])


# B_av was chosen as 1.0e-44 / (atomic_mass * angstrom**2)
def _vibrational_moment(vibfreqs=None, B_av=602.2140762081121):  # noqa: N803
    """Calculate moments of inertia for a free rotors with the same frequencies.

    This is part of the quasi-RRHO approach of S. Grimme, see
        [*Theory. Chem. Eur. J.*, **2012**, 18: 9955-9964](https://doi.org/10.1002/chem.201200497).

    Parameters
    ----------
    vibfreqs : array-like
        Frequency magnitudes in cm-1.
    B_av : float, optional

    Returns
    -------
    array-like
        Equivalent moments of inertia. Units are in amu·Å².

    Examples
    --------
    >>> _vibrational_moment(500.0)
    array([0.1059010])
    >>> _vibrational_moment(100.0)
    array([0.52913])
    >>> _vibrational_moment([5.0, 10.0])
    array([10.41,  5.250])
    """
    # TODO(schneiderfelipe): should we receive vibrational temperatures and
    # avoid calling it twice when calling calc_vib_entropy?

    # I sincerely have absolutely no clue as to why we should divide by np.pi
    # and not by 2 * np.pi. The original paper says nothing about it (and the
    # equation 4 there is probably wrong), but the expression below reproduces
    # both Figure 2 of the original paper and the results returned by ORCA.
    moments = constants.hbar**2 / (
        2.0 * constants.k * _vibrational_temperature(vibfreqs) / np.pi
    )
    moments = moments / (constants.atomic_mass * constants.angstrom**2)
    return moments * B_av / (moments + B_av)


# omega was chosen as (k * 298.15 / (2.0 * h)) * centi / c
def _head_gordon_damping(vibfreqs, omega=103.61231288246945, alpha=4):
    """Calculate the Head-Gordon damping function.

    This is part of the quasi-RRHO approach of S. Grimme, see
        [*Theory. Chem. Eur. J.*, **2012**, 18: 9955-9964](https://doi.org/10.1002/chem.201200497).

    Parameters
    ----------
    vibfreqs : array-like
        Frequency magnitudes in cm-1.
    omega, alpha : float, optional

    Returns
    -------
    array-like

    Examples
    --------
    >>> _head_gordon_damping(500.0)
    array([0.998])
    >>> _head_gordon_damping(100.0)
    array([0.5])
    >>> _head_gordon_damping(103.61231288246945)
    array([0.5])

    This function uses `_check_vibfreqs` and as such employs its default
    cutoff for imaginary frequencies. This means that weights for positive
    values are returned as expected, but small non-positive values are
    inverted (sign flipped):

    >>> _head_gordon_damping([5.0, 10.0])
    array([6.e-06, 1.e-04])
    >>> _head_gordon_damping([-5.0, 5.0, 15.0, 100.0])
    array([6.e-06, 6.e-06, 4.e-04, 5.e-01])
    >>> _head_gordon_damping([-55.0, -5.0, 5.0, 15.0, 100.0])
    array([6.e-06, 6.e-06, 4.e-04, 5.e-01])
    """
    vibfreqs = _check_vibfreqs(vibfreqs)
    return 1.0 / (1.0 + (omega / vibfreqs) ** alpha)


def molar_volume(temperature=298.15, pressure=constants.atm):
    """Calculate the ideal gas molar volume.

    Parameters
    ----------
    temperature : array-like, optional
        Absolute temperature in Kelvin.
    pressure : array-like, optional
        Reference gas pressure.

    Returns
    -------
    float
        Molar volume in cubic meters per mole.

    Examples
    --------
    >>> molar_volume(temperature=273.15)
    0.0224140
    >>> molar_volume(temperature=273.15, pressure=constants.bar)
    0.0227110
    >>> molar_volume()
    0.0244654
    >>> molar_volume(pressure=constants.bar)
    0.0247896

    Below we calculate the molar volume at 298.15 K and 1 atm in Å³ per molecule:

    >>> molar_volume() / (constants.angstrom ** 3 * constants.N_A)
    40625.758632362515
    """
    temperature = np.asarray(temperature)

    molar_volume = constants.R * temperature / np.asarray(pressure)
    logger.debug(f"molar volume = {molar_volume} Å³")  # noqa: G004
    return molar_volume
