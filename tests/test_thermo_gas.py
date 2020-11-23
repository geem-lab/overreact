#!/usr/bin/env python3

"""Tests for gas phase using the `thermo` module."""

import numpy as np
import pytest

from overreact import constants
from overreact import coords
from overreact import core
from overreact.datasets import logfiles
from overreact import _thermo

# TODO(schneiderfelipe): test summability of enthalpy and entropy terms: two
# equal degrees of freedom should sum to two times a single degree of freedom:
#
# >>> np.isclose(_thermo._gas.calc_vib_energy([100, 100]),
# ...            2 * _thermo._gas.calc_vib_energy([100]))
# True
#
# TODO(schneiderfelipe): test high temperature limits such as 3 RT and so on.
# TODO(schneiderfelipe): test zero temperature limits such as zero entropy and
# so on.


def test_sanity_for_relative_thermochemistry():
    """Ensure we have decent quality for (relative) thermochemical analysis.

    This partially ensures we do similar analysis as Gaussian, see
    https://gaussian.com/thermo/.
    """
    energies = np.array([-77.662998, -1.117506, -78.306179, -0.466582])
    enthalpy_corrections = np.array([0.075441, 0.015792, 0.094005, 0.002360])
    freeenergy_corrections = np.array([0.046513, 0.001079, 0.068316, -0.010654])
    enthalpies = energies + enthalpy_corrections
    freeenergies = energies + freeenergy_corrections

    scheme = core.parse_reactions("C2H5 + H2 -> C2H6 + H")
    delta_enthalpy = _thermo.get_delta(scheme.A, enthalpies)[0]
    delta_freeenergy = _thermo.get_delta(scheme.A, freeenergies)[0]
    assert delta_enthalpy == pytest.approx(0.012876, 8e-5)
    assert delta_enthalpy * (constants.hartree * constants.N_A) / (
        constants.kcal
    ) == pytest.approx(8.08, 9e-4)
    assert delta_freeenergy == pytest.approx(0.017813)
    assert delta_freeenergy * (constants.hartree * constants.N_A) / (
        constants.kcal
    ) == pytest.approx(11.18, 9e-4)


def test_sanity_for_absolute_thermochemistry():
    """Ensure we have decent quality for (absolute) thermochemical analysis.

    This partially ensures we do similar analysis as Gaussian, see
    https://gaussian.com/thermo/.
    """
    moments = (
        np.array([23.57594, 88.34097, 88.34208])
        * constants.atomic_mass
        * constants.bohr ** 2
    )
    assert _thermo._gas._rotational_temperature(
        moments / (constants.atomic_mass * constants.angstrom ** 2)
    ) == pytest.approx([3.67381, 0.98044, 0.98043], 2e-5)
    assert (
        constants.hbar ** 2 / (2.0 * constants.h * moments)
    ) / constants.giga == pytest.approx([76.55013, 20.42926, 20.42901])

    vibtemps = np.array(
        [
            602.31,
            1607.07,
            1607.45,
            1683.83,
            1978.85,
            1978.87,
            2303.03,
            2389.95,
            2389.96,
            2404.55,
            2417.29,
            2417.30,
            4202.52,
            4227.44,
            4244.32,
            4244.93,
            4291.74,
            4292.31,
        ]
    )
    vibfreqs = vibtemps * constants.k * constants.centi / (constants.h * constants.c)
    assert _thermo._gas._vibrational_temperature(vibfreqs) == pytest.approx(vibtemps)
    zpe = _thermo._gas.calc_vib_energy(vibfreqs, temperature=0.0)
    assert zpe == pytest.approx(204885.0, 7e-5)
    assert zpe / constants.kcal == pytest.approx(48.96870, 7e-5)
    assert zpe / (constants.hartree * constants.N_A) == pytest.approx(0.078037, 8e-5)

    elec_internal_energy = _thermo._gas.calc_elec_energy(0.0, 1.0)
    trans_internal_energy = _thermo._gas.calc_trans_energy()
    rot_internal_energy = _thermo._gas.calc_rot_energy(
        moments / (constants.atomic_mass * constants.angstrom ** 2)
    )
    vib_internal_energy = _thermo._gas.calc_vib_energy(vibfreqs)
    internal_energy = _thermo.calc_internal_energy(
        0.0, 1.0, moments / (constants.atomic_mass * constants.angstrom ** 2), vibfreqs
    )
    assert elec_internal_energy == pytest.approx(0.0)
    assert trans_internal_energy / constants.kcal == pytest.approx(0.889, 4e-4)
    assert rot_internal_energy / constants.kcal == pytest.approx(0.889, 3e-3)
    assert vib_internal_energy / constants.kcal == pytest.approx(49.213, 7e-5)
    assert internal_energy == pytest.approx(
        elec_internal_energy
        + trans_internal_energy
        + rot_internal_energy
        + vib_internal_energy
    )
    assert internal_energy / constants.kcal == pytest.approx(50.990, 1e-4)

    atommasses = [12.0, 12.0, 1.00783, 1.00783, 1.00783, 1.00783, 1.00783, 1.00783]
    elec_entropy = _thermo._gas.calc_elec_entropy(0.0, 1.0)
    trans_entropy = _thermo.calc_trans_entropy(atommasses)
    rot_entropy = _thermo._gas.calc_rot_entropy(
        moments=moments / (constants.atomic_mass * constants.angstrom ** 2)
    )
    vib_entropy = _thermo._gas.calc_vib_entropy(vibfreqs)
    entropy = _thermo.calc_entropy(
        atommasses,
        energy=0.0,
        degeneracy=1.0,
        moments=moments / (constants.atomic_mass * constants.angstrom ** 2),
        vibfreqs=vibfreqs,
    )
    assert elec_entropy == pytest.approx(0.0)
    assert trans_entropy / constants.calorie == pytest.approx(36.134, 4e-6)
    assert rot_entropy / constants.calorie == pytest.approx(19.848, 1e-6)
    assert vib_entropy / constants.calorie == pytest.approx(1.136, 2e-3)
    assert entropy == pytest.approx(
        elec_entropy + trans_entropy + rot_entropy + vib_entropy
    )
    assert entropy / constants.calorie == pytest.approx(57.118, 3e-5)


def test_enthalpy_ideal_gases():
    """Calculate enthalpy of some ideal gases.

    We only check whether enthalpy matches the correct relationship with internal energy
    for a series of representative cases.
    """
    temperature = 298.15

    # He
    j = np.array([0, 1, 0])
    degeneracy = 2 * j + 1
    energy = np.array([0.000, 159855.9745, 166277.4403])
    internal_energy = _thermo.calc_internal_energy(
        energy=energy * 100 * constants.h * constants.c * constants.N_A,
        degeneracy=degeneracy,
        temperature=temperature,
    )
    enthalpy = _thermo.calc_enthalpy(
        energy=energy * 100 * constants.h * constants.c * constants.N_A,
        degeneracy=degeneracy,
        temperature=temperature,
    )
    assert enthalpy - internal_energy == pytest.approx(constants.R * temperature)

    # Ne, Ar, Kr, Xe
    internal_energy = _thermo.calc_internal_energy(temperature=temperature)
    enthalpy = _thermo.calc_enthalpy(temperature=temperature)
    assert enthalpy - internal_energy == pytest.approx(constants.R * temperature)

    # C
    j = np.array([0, 1, 2, 2, 0])
    degeneracy = 2 * j + 1
    energy = np.array([0.00000, 16.41671, 43.41350, 10192.66, 21648.02])
    internal_energy = _thermo.calc_internal_energy(
        energy=energy * 100 * constants.h * constants.c * constants.N_A,
        degeneracy=degeneracy,
        temperature=temperature,
    )
    enthalpy = _thermo.calc_enthalpy(
        energy=energy * 100 * constants.h * constants.c * constants.N_A,
        degeneracy=degeneracy,
        temperature=temperature,
    )
    assert enthalpy - internal_energy == pytest.approx(constants.R * temperature)

    # H2
    i = (constants.hbar ** 2 / (2.0 * constants.k * 85.3)) / (
        constants.atomic_mass * constants.angstrom ** 2
    )
    vibfreq = 6125 * constants.k * constants.centi / (constants.h * constants.c)
    internal_energy = _thermo.calc_internal_energy(
        moments=[0, i, i], vibfreqs=vibfreq, temperature=temperature
    )
    enthalpy = _thermo.calc_enthalpy(
        moments=[0, i, i], vibfreqs=vibfreq, temperature=temperature
    )
    assert enthalpy - internal_energy == pytest.approx(constants.R * temperature)

    # O2
    degeneracy = 3
    i = (constants.hbar ** 2 / (2.0 * constants.k * 2.07)) / (
        constants.atomic_mass * constants.angstrom ** 2
    )
    vibfreq = 2256 * constants.k * constants.centi / (constants.h * constants.c)
    internal_energy = _thermo.calc_internal_energy(
        degeneracy=degeneracy,
        moments=[0, i, i],
        vibfreqs=vibfreq,
        temperature=temperature,
    )
    enthalpy = _thermo.calc_enthalpy(
        degeneracy=degeneracy,
        moments=[0, i, i],
        vibfreqs=vibfreq,
        temperature=temperature,
    )
    assert enthalpy - internal_energy == pytest.approx(constants.R * temperature)

    # HCl
    i = (constants.hbar ** 2 / (2.0 * constants.k * 15.02)) / (
        constants.atomic_mass * constants.angstrom ** 2
    )
    vibfreq = 4227 * constants.k * constants.centi / (constants.h * constants.c)
    internal_energy = _thermo.calc_internal_energy(
        moments=[0, i, i], vibfreqs=vibfreq, temperature=temperature
    )
    enthalpy = _thermo.calc_enthalpy(
        moments=[0, i, i], vibfreqs=vibfreq, temperature=temperature
    )
    assert enthalpy - internal_energy == pytest.approx(constants.R * temperature)

    # CO2
    i = (constants.hbar ** 2 / (2.0 * constants.k * 0.561)) / (
        constants.atomic_mass * constants.angstrom ** 2
    )
    vibfreqs = (
        np.array([3360, 954, 954, 1890])
        * constants.k
        * constants.centi
        / (constants.h * constants.c)
    )
    internal_energy = _thermo.calc_internal_energy(
        moments=[0, i, i], vibfreqs=vibfreqs, temperature=temperature
    )
    enthalpy = _thermo.calc_enthalpy(
        moments=[0, i, i], vibfreqs=vibfreqs, temperature=temperature
    )
    assert enthalpy - internal_energy == pytest.approx(constants.R * temperature)

    # NH3
    ia = (constants.hbar ** 2 / (2.0 * constants.k * 13.6)) / (
        constants.atomic_mass * constants.angstrom ** 2
    )
    ib = (constants.hbar ** 2 / (2.0 * constants.k * 8.92)) / (
        constants.atomic_mass * constants.angstrom ** 2
    )
    vibfreqs = (
        np.array([4800, 1360, 4880, 4880, 2330, 2330])
        * constants.k
        * constants.centi
        / (constants.h * constants.c)
    )
    internal_energy = _thermo.calc_internal_energy(
        moments=[ia, ia, ib], vibfreqs=vibfreqs, temperature=temperature
    )
    enthalpy = _thermo.calc_enthalpy(
        moments=[ia, ia, ib], vibfreqs=vibfreqs, temperature=temperature
    )
    assert enthalpy - internal_energy == pytest.approx(constants.R * temperature)

    # C6H6
    data = logfiles["symmetries"]["benzene"]
    moments = coords.inertia(data.atommasses, data.atomcoords)[0]
    vibfreqs = np.asanyarray(data.vibfreqs)
    internal_energy = _thermo.calc_internal_energy(
        moments=moments, vibfreqs=vibfreqs, temperature=temperature
    )
    enthalpy = _thermo.calc_enthalpy(
        moments=moments, vibfreqs=vibfreqs, temperature=temperature
    )
    assert enthalpy - internal_energy == pytest.approx(constants.R * temperature)


def test_entropy_ideal_monoatomic_gases():
    """Reproduce experimental entropies of some ideal monoatomic gases.

    Reference experimental data is from Table 5-3 of Statistical
    Thermodynamics, McQuarrie. Atomic energy states are from the NIST atomic
    data
    (<https://physics.nist.gov/PhysRefData/Handbook/Tables/fluorinetable5.htm>);
    similar data can also be found in Table 5-1 of the same book. Atomic masses
    were retrieved from <https://www.ptable.com/>.
    """
    temperature = 298.0

    # He
    j = np.array([0, 1, 0])
    degeneracy = 2 * j + 1
    energy = np.array([0.000, 159855.9745, 166277.4403])
    assert _thermo.calc_entropy(
        4.0026,
        energy=energy * 100 * constants.h * constants.c * constants.N_A,
        degeneracy=degeneracy,
        temperature=temperature,
    ) / constants.calorie == pytest.approx(30.13, 1e-3)

    # Ne
    assert _thermo.calc_entropy(
        20.180, temperature=temperature
    ) / constants.calorie == pytest.approx(34.95, 1e-3)

    # Ar
    assert _thermo.calc_entropy(
        39.948, temperature=temperature
    ) / constants.calorie == pytest.approx(36.98, 1e-4)

    # Kr
    assert _thermo.calc_entropy(
        83.798, temperature=temperature
    ) / constants.calorie == pytest.approx(39.19, 1e-4)

    # Xe
    assert _thermo.calc_entropy(
        131.29, temperature=temperature
    ) / constants.calorie == pytest.approx(40.53, 1e-4)

    # C
    j = np.array([0, 1, 2, 2, 0])
    degeneracy = 2 * j + 1
    energy = np.array([0.00000, 16.41671, 43.41350, 10192.66, 21648.02])
    assert _thermo.calc_entropy(
        12.011,
        energy=energy * 100 * constants.h * constants.c * constants.N_A,
        degeneracy=degeneracy,
        temperature=temperature,
    ) / constants.calorie == pytest.approx(37.76, 1e-4)

    # Na
    assert _thermo.calc_entropy(
        22.990, degeneracy=2, temperature=temperature
    ) / constants.calorie == pytest.approx(36.72, 1e-3)

    # Al
    j = np.array([1 / 2, 3 / 2, 1 / 2])
    degeneracy = 2 * j + 1
    energy = np.array([0.000, 112.061, 25347.756])
    assert _thermo.calc_entropy(
        26.982,
        energy=energy * 100 * constants.h * constants.c * constants.N_A,
        degeneracy=degeneracy,
        temperature=temperature,
    ) / constants.calorie == pytest.approx(39.30, 1e-4)

    # Ag
    assert _thermo.calc_entropy(
        107.87, degeneracy=2, temperature=temperature
    ) / constants.calorie == pytest.approx(41.32, 1e-4)

    # Hg
    assert _thermo.calc_entropy(
        200.59, temperature=temperature
    ) / constants.calorie == pytest.approx(41.8, 1e-3)


def test_internal_energy_ideal_monoatomic_gases():
    """Calculate internal energies of some ideal monoatomic gases.

    Atomic energy states are from the NIST atomic data
    (<https://physics.nist.gov/PhysRefData/Handbook/Tables/fluorinetable5.htm>);
    similar data can also be found in Table 5-1 of Statistical Thermodynamics,
    McQuarrie. Atomic masses were retrieved from <https://www.ptable.com/>.
    """
    temperature = 298.15

    # He
    j = np.array([0, 1, 0])
    degeneracy = 2 * j + 1
    energy = np.array([0.000, 159855.9745, 166277.4403])
    assert _thermo.calc_internal_energy(
        energy=energy * 100 * constants.h * constants.c * constants.N_A,
        degeneracy=degeneracy,
        temperature=temperature,
    ) == pytest.approx(3718.44, 1e-5)

    # Ne, Ar, Kr, Xe
    assert _thermo.calc_internal_energy(temperature=temperature) == pytest.approx(
        3718.44, 1e-5
    )

    # C
    j = np.array([0, 1, 2, 2, 0])
    degeneracy = 2 * j + 1
    energy = np.array([0.00000, 16.41671, 43.41350, 10192.66, 21648.02])
    assert _thermo.calc_internal_energy(
        energy=energy * 100 * constants.h * constants.c * constants.N_A,
        degeneracy=degeneracy,
        temperature=temperature,
    ) == pytest.approx(4057.05, 1e-5)

    # Na
    assert _thermo.calc_internal_energy(
        degeneracy=2, temperature=temperature
    ) == pytest.approx(3718.44, 1e-5)

    # Al
    j = np.array([1 / 2, 3 / 2, 1 / 2])
    degeneracy = 2 * j + 1
    energy = np.array([0.000, 112.061, 25347.756])
    assert _thermo.calc_internal_energy(
        energy=energy * 100 * constants.h * constants.c * constants.N_A,
        degeneracy=degeneracy,
        temperature=temperature,
    ) == pytest.approx(4439.68, 1e-6)

    # Ag
    assert _thermo.calc_internal_energy(
        degeneracy=2, temperature=temperature
    ) == pytest.approx(3718.44, 1e-5)

    # Hg
    assert _thermo.calc_internal_energy(temperature=temperature) == pytest.approx(
        3718.44, 1e-5
    )


def test_entropy_ideal_diatomic_gases():
    """Reproduce experimental entropies of some ideal diatomic gases.

    Reference experimental data is from Table 6-3 of Statistical
    Thermodynamics, McQuarrie. Vibrational and rotational temperatures are from
    Table 6-1 of the same book. Atomic masses were retrieved from
    <https://www.ptable.com/>.
    """
    temperature = 298.15

    # H2
    i = (constants.hbar ** 2 / (2.0 * constants.k * 85.3)) / (
        constants.atomic_mass * constants.angstrom ** 2
    )
    symmetry_number = 2
    vibfreq = 6125 * constants.k * constants.centi / (constants.h * constants.c)
    assert _thermo.calc_entropy(
        2 * 1.008,
        moments=[0, i, i],
        symmetry_number=symmetry_number,
        vibfreqs=vibfreq,
        temperature=temperature,
    ) / constants.calorie == pytest.approx(31.2, 1e-3)

    # O2
    degeneracy = 3
    i = (constants.hbar ** 2 / (2.0 * constants.k * 2.07)) / (
        constants.atomic_mass * constants.angstrom ** 2
    )
    symmetry_number = 2
    vibfreq = 2256 * constants.k * constants.centi / (constants.h * constants.c)
    assert _thermo.calc_entropy(
        2 * 15.999,
        degeneracy=degeneracy,
        moments=[0, i, i],
        symmetry_number=symmetry_number,
        vibfreqs=vibfreq,
        temperature=temperature,
    ) / constants.calorie == pytest.approx(49.0, 1e-5)

    # N2
    i = (constants.hbar ** 2 / (2.0 * constants.k * 2.88)) / (
        constants.atomic_mass * constants.angstrom ** 2
    )
    symmetry_number = 2
    vibfreq = 3374 * constants.k * constants.centi / (constants.h * constants.c)
    assert _thermo.calc_entropy(
        2 * 14.007,
        moments=[0, i, i],
        symmetry_number=symmetry_number,
        vibfreqs=vibfreq,
        temperature=temperature,
    ) / constants.calorie == pytest.approx(45.7, 1e-2)

    # Cl2
    i = (constants.hbar ** 2 / (2.0 * constants.k * 0.351)) / (
        constants.atomic_mass * constants.angstrom ** 2
    )
    symmetry_number = 2
    vibfreq = 808 * constants.k * constants.centi / (constants.h * constants.c)
    assert _thermo.calc_entropy(
        2 * 35.45,
        moments=[0, i, i],
        symmetry_number=symmetry_number,
        vibfreqs=vibfreq,
        temperature=temperature,
    ) / constants.calorie == pytest.approx(53.3, 1e-2)

    # HCl
    i = (constants.hbar ** 2 / (2.0 * constants.k * 15.02)) / (
        constants.atomic_mass * constants.angstrom ** 2
    )
    vibfreq = 4227 * constants.k * constants.centi / (constants.h * constants.c)
    assert _thermo.calc_entropy(
        [1.008, 35.45], moments=[0, i, i], vibfreqs=vibfreq, temperature=temperature
    ) / constants.calorie == pytest.approx(44.6, 1e-3)

    # HBr
    i = (constants.hbar ** 2 / (2.0 * constants.k * 12.02)) / (
        constants.atomic_mass * constants.angstrom ** 2
    )
    vibfreq = 3787 * constants.k * constants.centi / (constants.h * constants.c)
    assert _thermo.calc_entropy(
        [1.008, 79.904], moments=[0, i, i], vibfreqs=vibfreq, temperature=temperature
    ) / constants.calorie == pytest.approx(47.4, 1e-2)

    # HI
    i = (constants.hbar ** 2 / (2.0 * constants.k * 9.06)) / (
        constants.atomic_mass * constants.angstrom ** 2
    )
    vibfreq = 3266 * constants.k * constants.centi / (constants.h * constants.c)
    assert _thermo.calc_entropy(
        [1.008, 126.90], moments=[0, i, i], vibfreqs=vibfreq, temperature=temperature
    ) / constants.calorie == pytest.approx(49.3, 1e-2)

    # CO
    i = (constants.hbar ** 2 / (2.0 * constants.k * 2.77)) / (
        constants.atomic_mass * constants.angstrom ** 2
    )
    vibfreq = 3103 * constants.k * constants.centi / (constants.h * constants.c)
    assert _thermo.calc_entropy(
        [12.011, 15.999], moments=[0, i, i], vibfreqs=vibfreq, temperature=temperature
    ) / constants.calorie == pytest.approx(
        46.2 + constants.R * np.log(2) / constants.calorie, 1e-2
    )


def test_internal_energy_ideal_diatomic_gases():
    """Calculate internal energies of some ideal diatomic gases.

    Vibrational and rotational temperatures are from Table 6-1 of Statistical
    Thermodynamics, McQuarrie. Atomic masses were retrieved from
    <https://www.ptable.com/>.
    """
    temperature = 298.15

    # H2
    i = (constants.hbar ** 2 / (2.0 * constants.k * 85.3)) / (
        constants.atomic_mass * constants.angstrom ** 2
    )
    vibfreq = 6125 * constants.k * constants.centi / (constants.h * constants.c)
    assert _thermo.calc_internal_energy(
        moments=[0, i, i], vibfreqs=vibfreq, temperature=temperature
    ) == pytest.approx(31419.52, 4e-7)

    # O2
    degeneracy = 3
    i = (constants.hbar ** 2 / (2.0 * constants.k * 2.07)) / (
        constants.atomic_mass * constants.angstrom ** 2
    )
    vibfreq = 2256 * constants.k * constants.centi / (constants.h * constants.c)
    assert _thermo.calc_internal_energy(
        degeneracy=degeneracy,
        moments=[0, i, i],
        vibfreqs=vibfreq,
        temperature=temperature,
    ) == pytest.approx(15580.08, 2e-5)

    # N2
    i = (constants.hbar ** 2 / (2.0 * constants.k * 2.88)) / (
        constants.atomic_mass * constants.angstrom ** 2
    )
    vibfreq = 3374 * constants.k * constants.centi / (constants.h * constants.c)
    assert _thermo.calc_internal_energy(
        moments=[0, i, i], vibfreqs=vibfreq, temperature=temperature
    ) == pytest.approx(20216.25, 3e-6)

    # Cl2
    i = (constants.hbar ** 2 / (2.0 * constants.k * 0.351)) / (
        constants.atomic_mass * constants.angstrom ** 2
    )
    vibfreq = 808 * constants.k * constants.centi / (constants.h * constants.c)
    assert _thermo.calc_internal_energy(
        moments=[0, i, i], vibfreqs=vibfreq, temperature=temperature
    ) == pytest.approx(10034.30, 4e-4)

    # HCl
    i = (constants.hbar ** 2 / (2.0 * constants.k * 15.02)) / (
        constants.atomic_mass * constants.angstrom ** 2
    )
    vibfreq = 4227 * constants.k * constants.centi / (constants.h * constants.c)
    assert _thermo.calc_internal_energy(
        moments=[0, i, i], vibfreqs=vibfreq, temperature=temperature
    ) == pytest.approx(23728.27, 2e-6)

    # HBr
    i = (constants.hbar ** 2 / (2.0 * constants.k * 12.02)) / (
        constants.atomic_mass * constants.angstrom ** 2
    )
    vibfreq = 3787 * constants.k * constants.centi / (constants.h * constants.c)
    assert _thermo.calc_internal_energy(
        moments=[0, i, i], vibfreqs=vibfreq, temperature=temperature
    ) == pytest.approx(21907.52, 2e-6)

    # HI
    i = (constants.hbar ** 2 / (2.0 * constants.k * 9.06)) / (
        constants.atomic_mass * constants.angstrom ** 2
    )
    vibfreq = 3266 * constants.k * constants.centi / (constants.h * constants.c)
    assert _thermo.calc_internal_energy(
        moments=[0, i, i], vibfreqs=vibfreq, temperature=temperature
    ) == pytest.approx(19750.22, 3e-6)

    # CO
    i = (constants.hbar ** 2 / (2.0 * constants.k * 2.77)) / (
        constants.atomic_mass * constants.angstrom ** 2
    )
    vibfreq = 3103 * constants.k * constants.centi / (constants.h * constants.c)
    assert _thermo.calc_internal_energy(
        moments=[0, i, i], vibfreqs=vibfreq, temperature=temperature
    ) == pytest.approx(19090.38, 4e-6)


def test_entropy_ideal_polyatomic_gases():
    """Reproduce experimental entropies of some ideal polyatomic gases.

    Reference experimental data is from Table 8-3 of Statistical
    Thermodynamics, McQuarrie. Vibrational and rotational temperatures are from
    Table 8-1 of the same book. Atomic masses were retrieved from
    <https://www.ptable.com/>. The exception is benzene, whose vibrational and
    rotational information was obtained from a revPBE-D4-gCP/def2-SVP
    calculation.
    """
    temperature = 298.15

    # CO2
    i = (constants.hbar ** 2 / (2.0 * constants.k * 0.561)) / (
        constants.atomic_mass * constants.angstrom ** 2
    )
    symmetry_number = 2
    vibfreqs = (
        np.array([3360, 954, 954, 1890])
        * constants.k
        * constants.centi
        / (constants.h * constants.c)
    )
    assert _thermo.calc_entropy(
        [12.011, 15.999, 15.999],
        moments=[0, i, i],
        symmetry_number=symmetry_number,
        vibfreqs=vibfreqs,
        temperature=temperature,
    ) / constants.calorie == pytest.approx(51.0, 1e-2)

    # NH3
    ia = (constants.hbar ** 2 / (2.0 * constants.k * 13.6)) / (
        constants.atomic_mass * constants.angstrom ** 2
    )
    ib = (constants.hbar ** 2 / (2.0 * constants.k * 8.92)) / (
        constants.atomic_mass * constants.angstrom ** 2
    )
    symmetry_number = 3
    vibfreqs = (
        np.array([4800, 1360, 4880, 4880, 2330, 2330])
        * constants.k
        * constants.centi
        / (constants.h * constants.c)
    )
    assert _thermo.calc_entropy(
        [14.007, 1.008, 1.008, 1.008],
        moments=[ia, ia, ib],
        symmetry_number=symmetry_number,
        vibfreqs=vibfreqs,
        temperature=temperature,
    ) / constants.calorie == pytest.approx(46.0, 1e-2)

    # NO2
    ia = (constants.hbar ** 2 / (2.0 * constants.k * 11.5)) / (
        constants.atomic_mass * constants.angstrom ** 2
    )
    ib = (constants.hbar ** 2 / (2.0 * constants.k * 0.624)) / (
        constants.atomic_mass * constants.angstrom ** 2
    )
    ic = (constants.hbar ** 2 / (2.0 * constants.k * 0.590)) / (
        constants.atomic_mass * constants.angstrom ** 2
    )
    symmetry_number = 2
    vibfreqs = (
        np.array([1900, 1980, 2330])
        * constants.k
        * constants.centi
        / (constants.h * constants.c)
    )
    assert _thermo.calc_entropy(
        [14.007, 15.999, 15.999],
        degeneracy=2,
        moments=[ia, ib, ic],
        symmetry_number=symmetry_number,
        vibfreqs=vibfreqs,
        temperature=temperature,
    ) / constants.calorie == pytest.approx(57.5, 1e-2)

    # ClO2
    ia = (constants.hbar ** 2 / (2.0 * constants.k * 2.50)) / (
        constants.atomic_mass * constants.angstrom ** 2
    )
    ib = (constants.hbar ** 2 / (2.0 * constants.k * 0.478)) / (
        constants.atomic_mass * constants.angstrom ** 2
    )
    ic = (constants.hbar ** 2 / (2.0 * constants.k * 0.400)) / (
        constants.atomic_mass * constants.angstrom ** 2
    )
    symmetry_number = 2
    vibfreqs = (
        np.array([1360, 640, 1600])
        * constants.k
        * constants.centi
        / (constants.h * constants.c)
    )
    assert _thermo.calc_entropy(
        [35.45, 15.999, 15.999],
        moments=[ia, ib, ic],
        symmetry_number=symmetry_number,
        vibfreqs=vibfreqs,
        temperature=temperature,
    ) / constants.calorie == pytest.approx(59.6, 1e-2)

    # CH4
    i = (constants.hbar ** 2 / (2.0 * constants.k * 7.54)) / (
        constants.atomic_mass * constants.angstrom ** 2
    )
    symmetry_number = 12
    vibfreqs = (
        np.array([4170, 2180, 2180, 4320, 4320, 4320, 1870, 1870, 1870])
        * constants.k
        * constants.centi
        / (constants.h * constants.c)
    )
    assert _thermo.calc_entropy(
        [12.011, 1.008, 1.008, 1.008, 1.008],
        moments=[i, i, i],
        symmetry_number=symmetry_number,
        vibfreqs=vibfreqs,
        temperature=temperature,
    ) / constants.calorie == pytest.approx(44.5, 1e-3)

    # CH3Cl
    ia = (constants.hbar ** 2 / (2.0 * constants.k * 7.32)) / (
        constants.atomic_mass * constants.angstrom ** 2
    )
    ib = (constants.hbar ** 2 / (2.0 * constants.k * 0.637)) / (
        constants.atomic_mass * constants.angstrom ** 2
    )
    symmetry_number = 3
    vibfreqs = (
        np.array([4270, 1950, 1050, 4380, 4380, 2140, 2140, 1460, 1460])
        * constants.k
        * constants.centi
        / (constants.h * constants.c)
    )
    assert _thermo.calc_entropy(
        [12.011, 1.008, 1.008, 1.008, 35.45],
        moments=[ia, ib, ib],
        symmetry_number=symmetry_number,
        vibfreqs=vibfreqs,
        temperature=temperature,
    ) / constants.calorie == pytest.approx(56.0, 1e-3)

    # CCl4
    i = (constants.hbar ** 2 / (2.0 * constants.k * 0.0823)) / (
        constants.atomic_mass * constants.angstrom ** 2
    )
    symmetry_number = 12
    vibfreqs = (
        np.array([660, 310, 310, 1120, 1120, 1120, 450, 450, 450])
        * constants.k
        * constants.centi
        / (constants.h * constants.c)
    )
    assert _thermo.calc_entropy(
        [12.011, 35.45, 35.45, 35.45, 35.45],
        moments=[i, i, i],
        symmetry_number=symmetry_number,
        vibfreqs=vibfreqs,
        temperature=temperature,
    ) / constants.calorie == pytest.approx(73.9, 1e-2)

    # C6H6
    data = logfiles["symmetries"]["benzene"]
    moments = coords.inertia(data.atommasses, data.atomcoords)[0]
    point_group = coords.find_point_group(data.atommasses, data.atomcoords)
    assert point_group == "D6h"
    symmetry_number = coords.symmetry_number(point_group)
    vibfreqs = np.asanyarray(data.vibfreqs)
    assert _thermo.calc_entropy(
        data.atommasses,
        moments=moments,
        symmetry_number=symmetry_number,
        vibfreqs=vibfreqs,
        temperature=temperature,
    ) / constants.calorie == pytest.approx(64.4, 1e-2)


def test_internal_energy_ideal_polyatomic_gases():
    """Calculate internal energies of some ideal polyatomic gases.

    Vibrational and rotational temperatures are from Table 8-1 of Statistical
    Thermodynamics, McQuarrie. Atomic masses were retrieved from
    <https://www.ptable.com/>. The exception is benzene, whose vibrational and
    rotational information was obtained from a revPBE-D4-gCP/def2-SVP
    calculation.
    """
    temperature = 298.15

    # CO2
    i = (constants.hbar ** 2 / (2.0 * constants.k * 0.561)) / (
        constants.atomic_mass * constants.angstrom ** 2
    )
    vibfreqs = (
        np.array([3360, 954, 954, 1890])
        * constants.k
        * constants.centi
        / (constants.h * constants.c)
    )
    assert _thermo.calc_internal_energy(
        moments=[0, i, i], vibfreqs=vibfreqs, temperature=temperature
    ) == pytest.approx(36655.77, 2e-4)

    # NH3
    ia = (constants.hbar ** 2 / (2.0 * constants.k * 13.6)) / (
        constants.atomic_mass * constants.angstrom ** 2
    )
    ib = (constants.hbar ** 2 / (2.0 * constants.k * 8.92)) / (
        constants.atomic_mass * constants.angstrom ** 2
    )
    vibfreqs = (
        np.array([4800, 1360, 4880, 4880, 2330, 2330])
        * constants.k
        * constants.centi
        / (constants.h * constants.c)
    )
    assert _thermo.calc_internal_energy(
        moments=[ia, ia, ib], vibfreqs=vibfreqs, temperature=temperature
    ) == pytest.approx(93077.53, 2e-5)

    # NO2
    ia = (constants.hbar ** 2 / (2.0 * constants.k * 11.5)) / (
        constants.atomic_mass * constants.angstrom ** 2
    )
    ib = (constants.hbar ** 2 / (2.0 * constants.k * 0.624)) / (
        constants.atomic_mass * constants.angstrom ** 2
    )
    ic = (constants.hbar ** 2 / (2.0 * constants.k * 0.590)) / (
        constants.atomic_mass * constants.angstrom ** 2
    )
    vibfreqs = (
        np.array([1900, 1980, 2330])
        * constants.k
        * constants.centi
        / (constants.h * constants.c)
    )
    assert _thermo.calc_internal_energy(
        degeneracy=2, moments=[ia, ib, ic], vibfreqs=vibfreqs, temperature=temperature
    ) == pytest.approx(33291.99, 3e-5)

    # ClO2
    ia = (constants.hbar ** 2 / (2.0 * constants.k * 2.50)) / (
        constants.atomic_mass * constants.angstrom ** 2
    )
    ib = (constants.hbar ** 2 / (2.0 * constants.k * 0.478)) / (
        constants.atomic_mass * constants.angstrom ** 2
    )
    ic = (constants.hbar ** 2 / (2.0 * constants.k * 0.400)) / (
        constants.atomic_mass * constants.angstrom ** 2
    )
    vibfreqs = (
        np.array([1360, 640, 1600])
        * constants.k
        * constants.centi
        / (constants.h * constants.c)
    )
    assert _thermo.calc_internal_energy(
        moments=[ia, ib, ic], vibfreqs=vibfreqs, temperature=temperature
    ) == pytest.approx(23284.32, 4e-4)

    # CH4
    i = (constants.hbar ** 2 / (2.0 * constants.k * 7.54)) / (
        constants.atomic_mass * constants.angstrom ** 2
    )
    vibfreqs = (
        np.array([4170, 2180, 2180, 4320, 4320, 4320, 1870, 1870, 1870])
        * constants.k
        * constants.centi
        / (constants.h * constants.c)
    )
    assert _thermo.calc_internal_energy(
        moments=[i, i, i], vibfreqs=vibfreqs, temperature=temperature
    ) == pytest.approx(120179.00, 2e-5)

    # CH3Cl
    ia = (constants.hbar ** 2 / (2.0 * constants.k * 7.32)) / (
        constants.atomic_mass * constants.angstrom ** 2
    )
    ib = (constants.hbar ** 2 / (2.0 * constants.k * 0.637)) / (
        constants.atomic_mass * constants.angstrom ** 2
    )
    vibfreqs = (
        np.array([4270, 1950, 1050, 4380, 4380, 2140, 2140, 1460, 1460])
        * constants.k
        * constants.centi
        / (constants.h * constants.c)
    )
    assert _thermo.calc_internal_energy(
        moments=[ia, ib, ib], vibfreqs=vibfreqs, temperature=temperature
    ) == pytest.approx(104496.66, 4e-5)

    # CCl4
    i = (constants.hbar ** 2 / (2.0 * constants.k * 0.0823)) / (
        constants.atomic_mass * constants.angstrom ** 2
    )
    vibfreqs = (
        np.array([660, 310, 310, 1120, 1120, 1120, 450, 450, 450])
        * constants.k
        * constants.centi
        / (constants.h * constants.c)
    )
    assert _thermo.calc_internal_energy(
        moments=[i, i, i], vibfreqs=vibfreqs, temperature=temperature
    ) == pytest.approx(39684.88, 7e-3)

    # C6H6
    data = logfiles["symmetries"]["benzene"]
    moments = coords.inertia(data.atommasses, data.atomcoords)[0]
    vibfreqs = np.asanyarray(data.vibfreqs)
    assert _thermo.calc_internal_energy(
        moments=moments, vibfreqs=vibfreqs, temperature=temperature
    ) == pytest.approx(267700.49, 2e-4)


def test_heat_capacity_ideal_gases():
    """Calculate heat capacities of some ideal monoatomic gases.

    Reference data is from Table 8-2 of Statistical Thermodynamics, McQuarrie.
    Vibrational and rotational temperatures are from Table 8-1 of the same
    book. Atomic masses were retrieved from <https://www.ptable.com/>.
    """
    temperature = 300.0

    # CO2
    i = (constants.hbar ** 2 / (2.0 * constants.k * 0.561)) / (
        constants.atomic_mass * constants.angstrom ** 2
    )
    vibfreqs = (
        np.array([3360, 954, 954, 1890])
        * constants.k
        * constants.centi
        / (constants.h * constants.c)
    )
    assert _thermo.calc_heat_capacity(
        moments=[0, i, i], vibfreqs=vibfreqs, temperature=temperature
    ) / constants.R == pytest.approx(3.49, 1e-3)

    # N2O
    i = (constants.hbar ** 2 / (2.0 * constants.k * 0.603)) / (
        constants.atomic_mass * constants.angstrom ** 2
    )
    vibfreqs = (
        np.array([3200, 850, 850, 1840])
        * constants.k
        * constants.centi
        / (constants.h * constants.c)
    )
    assert _thermo.calc_heat_capacity(
        moments=[0, i, i], vibfreqs=vibfreqs, temperature=temperature
    ) / constants.R == pytest.approx(
        3.65, 1e-3
    )  # book is wrong, this is correct

    # NH3
    ia = (constants.hbar ** 2 / (2.0 * constants.k * 13.6)) / (
        constants.atomic_mass * constants.angstrom ** 2
    )
    ib = (constants.hbar ** 2 / (2.0 * constants.k * 8.92)) / (
        constants.atomic_mass * constants.angstrom ** 2
    )
    vibfreqs = (
        np.array([4800, 1360, 4880, 4880, 2330, 2330])
        * constants.k
        * constants.centi
        / (constants.h * constants.c)
    )
    assert _thermo.calc_heat_capacity(
        moments=[ia, ia, ib], vibfreqs=vibfreqs, temperature=temperature
    ) / constants.R == pytest.approx(3.28, 1e-3)

    # CH4
    i = (constants.hbar ** 2 / (2.0 * constants.k * 7.54)) / (
        constants.atomic_mass * constants.angstrom ** 2
    )
    vibfreqs = (
        np.array([4170, 2180, 2180, 4320, 4320, 4320, 1870, 1870, 1870])
        * constants.k
        * constants.centi
        / (constants.h * constants.c)
    )
    assert _thermo.calc_heat_capacity(
        moments=[i, i, i], vibfreqs=vibfreqs, temperature=temperature
    ) / constants.R == pytest.approx(3.30, 1e-2)

    # H2O
    ia = (constants.hbar ** 2 / (2.0 * constants.k * 40.1)) / (
        constants.atomic_mass * constants.angstrom ** 2
    )
    ib = (constants.hbar ** 2 / (2.0 * constants.k * 20.9)) / (
        constants.atomic_mass * constants.angstrom ** 2
    )
    ic = (constants.hbar ** 2 / (2.0 * constants.k * 13.4)) / (
        constants.atomic_mass * constants.angstrom ** 2
    )
    vibfreqs = (
        np.array([5360, 5160, 2290])
        * constants.k
        * constants.centi
        / (constants.h * constants.c)
    )
    assert _thermo.calc_heat_capacity(
        moments=[ia, ib, ic], vibfreqs=vibfreqs, temperature=temperature
    ) / constants.R == pytest.approx(3.03, 1e-3)


def test_get_delta_works():
    """Ensure safe usage of get_delta."""
    assert _thermo.get_delta([-1, 2], [-5.0, 5.0]) == pytest.approx(15.0)
    assert _thermo.get_delta([[-1, -1], [2, 1]], [-5.0, 5.0]) == pytest.approx(
        [15.0, 10.0]
    )


def test_equilibrium_constant_works():
    """Ensure equilibrium_constant gives correct numbers."""
    assert _thermo.equilibrium_constant(0.0) == 1.0
    assert _thermo.equilibrium_constant(1000.0) == pytest.approx(0.668, 1e-4)
    assert _thermo.equilibrium_constant(1718.5) == pytest.approx(0.5, 1e-4)
    assert _thermo.equilibrium_constant(68497.0) == pytest.approx(0.0)

    assert _thermo.equilibrium_constant(0.0, temperature=14.01) == 1.0
    assert _thermo.equilibrium_constant(1000.0, temperature=14.01) == pytest.approx(
        1.87e-4, 1e-3
    )
    assert _thermo.equilibrium_constant(80.8, temperature=14.01) == pytest.approx(
        0.5, 1e-3
    )
    assert _thermo.equilibrium_constant(68497.0, temperature=14.01) == pytest.approx(
        0.0
    )

    # The following reproduces some data from doi:10.1021/ic202081z for equilibria
    # relevant to the decomposition of aqueous nitrous acid. Table 2 gives Gibbs free
    # energies at 298 K for two methods (G3B3 and CBS-QB3), whose average is used for
    # the calculation of equilibrium constants. The authors could not reproduce
    # experimental data, which might be due to lack of symmetry, large errors in
    # solvation entropy or other factors being used. Overall, it is a very interesting
    # paper:
    g3b3 = np.array([25.2, 15.4, -19.4, -24.3, 40.7])
    cbs_qb3 = np.array([27.0, 11.6, -16.5, -32.9, 38.6])
    delta_freeenergy = constants.kilo * (g3b3 + cbs_qb3) / 2
    assert _thermo.equilibrium_constant(
        delta_freeenergy, temperature=298.0
    ) == pytest.approx([2.65e-5, 4.30e-3, 1.39e3, 1.04e5, 1.14e-7], 1.55e-2)


def test_molar_volume_is_precise():
    """Ensure our molar volumes are as precise as possible.

    Values below were taken from
    <https://en.wikipedia.org/wiki/Molar_volume#Ideal_gases>, which were
    calculated to the same precision using the ideal gas constant from 2014
    CODATA.
    """
    assert _thermo.molar_volume(273.15, constants.bar) == pytest.approx(
        0.02271098038, 1e-5
    )
    assert _thermo.molar_volume(pressure=constants.bar) == pytest.approx(
        0.02478959842, 1e-5
    )
    assert _thermo.molar_volume(273.15) == pytest.approx(0.022414, 1e-5)
    assert _thermo.molar_volume() == pytest.approx(0.024465, 1e-4)


def test_molar_volume_works_with_sequences():
    """Ensure molar volumes can be calculated for many temperatures at once."""
    assert _thermo.molar_volume([273.15, 298.15], constants.bar) == pytest.approx(
        [0.02271098038, 0.02478959842], 1e-5
    )
    assert _thermo.molar_volume(
        [273.15, 298.15], [constants.atm, constants.bar]
    ) == pytest.approx([0.022414, 0.02478959842], 1e-5)
    assert _thermo.molar_volume(
        273.15, [constants.atm, constants.bar]
    ) == pytest.approx([0.022414, 0.02271098038], 1e-5)


def test_change_reference_state_works_for_symmetry():
    """Ensure that change_reference_state works for symmetry contributions."""
    assert 200.0 * _thermo.change_reference_state(4, 1, sign=-1, temperature=200.0) / (
        constants.kcal
    ) == pytest.approx(-0.60, 9e-2)
    assert 298.15 * _thermo.change_reference_state(4, 1, sign=-1) / (
        constants.kcal
    ) == pytest.approx(-0.85, 9e-2)
    assert 300.0 * _thermo.change_reference_state(4, 1, sign=-1, temperature=300.0) / (
        constants.kcal
    ) == pytest.approx(-0.80, 9e-2)
    assert 400.0 * _thermo.change_reference_state(4, 1, sign=-1, temperature=400.0) / (
        constants.kcal
    ) == pytest.approx(-1.10, 1e-2)

    # calculating many symmetry corrections at once
    temperatures = np.array([0, 200, 298.15, 300, 400])
    assert temperatures * _thermo.change_reference_state(
        6, 12, sign=1, temperature=temperatures
    ) / constants.kcal == pytest.approx([0.0, -0.3, -0.4, -0.4, -0.6], 9e-2)


# TODO(schneiderfelipe): separate all tests of the QRRHO model in a separate test
# file.
# TODO(schneiderfelipe): test and compare solutions to small imaginary
# frequencies using the QRRHO model.
def test_head_gordon_damping():
    """Ensure the Head-Gordon damping for the treatment of QRRHO is done correctly."""
    assert _thermo._gas._head_gordon_damping(-70.0) == pytest.approx([])
    assert _thermo._gas._head_gordon_damping(-40.0) == pytest.approx(
        _thermo._gas._head_gordon_damping(40.0)
    )
    assert _thermo._gas._head_gordon_damping(-10.0) == pytest.approx(
        _thermo._gas._head_gordon_damping(10.0)
    )
    assert _thermo._gas._head_gordon_damping(1.0) == pytest.approx(8.67669882e-9)
    assert _thermo._gas._head_gordon_damping(10.0) == pytest.approx(8.67594611e-5)
    assert _thermo._gas._head_gordon_damping(100.0) == pytest.approx(0.5, 8e-2)
    assert _thermo._gas._head_gordon_damping(200.0) == pytest.approx(1.0, 7e-2)
    assert _thermo._gas._head_gordon_damping(300.0) == pytest.approx(1.0, 2e-2)
    assert _thermo._gas._head_gordon_damping(1000.0) == pytest.approx(1.0, 2e-4)

    assert _thermo._gas._head_gordon_damping(
        [-70.0, -10.0, 10.0, 100.0, 200.0, 300.0, 1000.0]
    ) == pytest.approx([8.67594611e-5, 8.67594611e-5, 0.5, 1.0, 1.0, 1.0], 8e-2)
