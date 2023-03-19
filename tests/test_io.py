#!/usr/bin/env python3  # noqa: INP001, EXE001

"""Tests for module io."""

import numpy as np
import pytest

import overreact as rx
from overreact import _constants as constants
from overreact import coords


def test_parse_model_raises_filenotfounderror():
    """Ensure parse_model raises FileNotFoundError when appropriate."""
    with pytest.raises(FileNotFoundError):
        rx.io.parse_model("not/available")

    with pytest.raises(FileNotFoundError):
        rx.io.parse_model("this/model/does/not/exist.k")

    with pytest.raises(FileNotFoundError):
        rx.io.parse_model("unreachable.jk")


def test_sanity_for_absolute_thermochemistry():  # noqa: PLR0915
    """Ensure we have decent quality for (absolute) thermochemical analysis.

    This partially ensures we do similar analysis as Gaussian, see
    https://gaussian.com/thermo/. Values from ORCA logfiles are tested as well.
    """
    temperature = 298.15
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
        ],
    )
    vibfreqs = vibtemps * constants.k * constants.centi / (constants.h * constants.c)

    # ethane eclipsed
    data = rx.io.read_logfile("data/ethane/B97-3c/eclipsed.out")
    assert data.atommasses == pytest.approx(
        [12.0, 12.0, 1.00783, 1.00783, 1.00783, 1.00783, 1.00783, 1.00783],
        1e-3,
    )
    assert np.sum(data.atommasses) == pytest.approx(30.04695, 8e-4)
    moments, axes, atomcoords = coords.inertia(data.atommasses, data.atomcoords)
    assert moments * constants.angstrom**2 / constants.bohr**2 == pytest.approx(
        [23.57594, 88.34097, 88.34208],
        7e-2,
    )
    assert data.vibfreqs == pytest.approx(vibfreqs, 1.8)  # just for sanity
    zpe = rx.thermo._gas.calc_vib_energy(data.vibfreqs, temperature=0.0)  # noqa: SLF001
    assert zpe == pytest.approx(204885.0, 6e-2)
    assert zpe / constants.kcal == pytest.approx(48.96870, 6e-2)
    assert zpe / (constants.hartree * constants.N_A) == pytest.approx(0.078037, 6e-2)
    thermal_correction = (
        rx.thermo.calc_internal_energy(
            energy=data.energy,
            degeneracy=data.mult,
            moments=moments,
            vibfreqs=data.vibfreqs,
        )
        - data.energy
    )
    assert thermal_correction / (constants.hartree * constants.N_A) == pytest.approx(
        0.081258,
        6e-2,
    )
    enthalpy_correction = (
        rx.thermo.calc_enthalpy(
            energy=data.energy,
            degeneracy=data.mult,
            moments=moments,
            vibfreqs=data.vibfreqs,
        )
        - data.energy
    )
    assert enthalpy_correction / (constants.hartree * constants.N_A) == pytest.approx(
        0.082202,
        6e-2,
    )
    assert enthalpy_correction - thermal_correction == pytest.approx(
        constants.R * temperature,
    )
    entropy = rx.thermo.calc_entropy(
        atommasses=data.atommasses,
        energy=data.energy,
        degeneracy=data.mult,
        moments=moments,
        symmetry_number=1,
        vibfreqs=data.vibfreqs,
    )
    freeenergy_correction = enthalpy_correction - temperature * entropy
    assert freeenergy_correction / (constants.hartree * constants.N_A) == pytest.approx(
        0.055064,
        8e-2,
    )
    assert freeenergy_correction / (constants.hartree * constants.N_A) == pytest.approx(
        0.05082405,
        1e-4,
    )  # ORCA logfile
    assert freeenergy_correction / constants.kcal == pytest.approx(
        31.89,
        3e-5,
    )  # ORCA logfile
    assert (data.energy + zpe) / (constants.hartree * constants.N_A) == pytest.approx(
        -79.140431,
        8e-3,
    )
    assert (data.energy + thermal_correction) / (
        constants.hartree * constants.N_A
    ) == pytest.approx(-79.137210, 8e-3)
    assert (data.energy + enthalpy_correction) / (
        constants.hartree * constants.N_A
    ) == pytest.approx(-79.136266, 8e-3)
    assert (data.energy + enthalpy_correction) / (
        constants.hartree * constants.N_A
    ) == pytest.approx(
        -79.70621533,
        7e-8,
    )  # ORCA logfile
    assert -temperature * entropy / (
        constants.hartree * constants.N_A
    ) == pytest.approx(
        -0.02685478,
        4e-6,
    )  # ORCA logfile
    assert -temperature * entropy / constants.kcal == pytest.approx(
        -16.85,
        1e-4,
    )  # ORCA logfile
    assert (data.energy + freeenergy_correction) / (
        constants.hartree * constants.N_A
    ) == pytest.approx(-79.163404, 8e-3)
    assert (data.energy + freeenergy_correction) / (
        constants.hartree * constants.N_A
    ) == pytest.approx(
        -79.73307012,
        7e-8,
    )  # ORCA logfile

    # ethane staggered
    data = rx.io.read_logfile("data/ethane/B97-3c/staggered.out")
    assert data.atommasses == pytest.approx(
        [12.0, 12.0, 1.00783, 1.00783, 1.00783, 1.00783, 1.00783, 1.00783],
        1e-3,
    )
    assert np.sum(data.atommasses) == pytest.approx(30.04695, 8e-4)
    moments, axes, atomcoords = coords.inertia(data.atommasses, data.atomcoords)
    assert moments * constants.angstrom**2 / constants.bohr**2 == pytest.approx(
        [23.57594, 88.34097, 88.34208],
        6e-2,
    )
    assert data.vibfreqs == pytest.approx(vibfreqs, 3e-1)
    zpe = rx.thermo._gas.calc_vib_energy(data.vibfreqs, temperature=0.0)  # noqa: SLF001
    assert zpe == pytest.approx(204885.0, 6e-2)
    assert zpe / constants.kcal == pytest.approx(48.96870, 6e-2)
    assert zpe / (constants.hartree * constants.N_A) == pytest.approx(0.078037, 6e-2)
    thermal_correction = (
        rx.thermo.calc_internal_energy(
            energy=data.energy,
            degeneracy=data.mult,
            moments=moments,
            vibfreqs=data.vibfreqs,
        )
        - data.energy
    )
    assert thermal_correction / (constants.hartree * constants.N_A) == pytest.approx(
        0.081258,
        5e-2,
    )
    enthalpy_correction = (
        rx.thermo.calc_enthalpy(
            energy=data.energy,
            degeneracy=data.mult,
            moments=moments,
            vibfreqs=data.vibfreqs,
        )
        - data.energy
    )
    assert enthalpy_correction / (constants.hartree * constants.N_A) == pytest.approx(
        0.082202,
        5e-2,
    )
    assert enthalpy_correction - thermal_correction == pytest.approx(
        constants.R * temperature,
    )
    entropy = rx.thermo.calc_entropy(
        atommasses=data.atommasses,
        energy=data.energy,
        degeneracy=data.mult,
        moments=moments,
        symmetry_number=1,
        vibfreqs=data.vibfreqs,
    )
    freeenergy_correction = enthalpy_correction - temperature * entropy
    assert freeenergy_correction / (constants.hartree * constants.N_A) == pytest.approx(
        0.055064,
        8e-2,
    )
    assert freeenergy_correction / (constants.hartree * constants.N_A) == pytest.approx(
        0.05092087,
        3e-4,
    )  # ORCA logfile
    assert freeenergy_correction / constants.kcal == pytest.approx(
        31.95,
        2e-4,
    )  # ORCA logfile
    assert (data.energy + zpe) / (constants.hartree * constants.N_A) == pytest.approx(
        -79.140431,
        8e-3,
    )
    assert (data.energy + thermal_correction) / (
        constants.hartree * constants.N_A
    ) == pytest.approx(-79.137210, 8e-3)
    assert (data.energy + enthalpy_correction) / (
        constants.hartree * constants.N_A
    ) == pytest.approx(-79.136266, 8e-3)
    assert (data.energy + enthalpy_correction) / (
        constants.hartree * constants.N_A
    ) == pytest.approx(
        -79.70971287,
        2e-7,
    )  # ORCA logfile
    assert -temperature * entropy / (
        constants.hartree * constants.N_A
    ) == pytest.approx(
        -0.02753672,
        8e-5,
    )  # ORCA logfile
    assert -temperature * entropy / constants.kcal == pytest.approx(
        -17.28,
        9e-4,
    )  # ORCA logfile
    assert (data.energy + freeenergy_correction) / (
        constants.hartree * constants.N_A
    ) == pytest.approx(-79.163404, 8e-3)
    assert (data.energy + freeenergy_correction) / (
        constants.hartree * constants.N_A
    ) == pytest.approx(
        -79.73724959,
        2e-7,
    )  # ORCA logfile


def test_compare_rrho_with_orca_logfile():  # noqa: PLR0915
    """Ensure we have decent quality for RRHO thermochemical analysis.

    Values from ORCA logfiles are tested.
    """
    temperature = 298.15

    # benzene
    data = rx.io.read_logfile("data/symmetries/benzene.out")
    assert np.sum(data.atommasses) == pytest.approx(78.11, 6e-5)
    moments, axes, atomcoords = coords.inertia(data.atommasses, data.atomcoords)
    symmetry_number = coords.symmetry_number(
        coords.find_point_group(data.atommasses, data.atomcoords),
    )
    assert symmetry_number == 12  # ORCA fails to find D6h symmetry!  # noqa: PLR2004
    internal_energy = rx.thermo.calc_internal_energy(
        energy=data.energy,
        degeneracy=data.mult,
        moments=moments,
        vibfreqs=data.vibfreqs,
    )
    zpe = rx.thermo._gas.calc_vib_energy(
        vibfreqs=data.vibfreqs,
        temperature=0.0,
    )
    elec_energy = rx.thermo._gas.calc_elec_energy(  # noqa: SLF001
        energy=data.energy,
        degeneracy=data.mult,
    )
    vib_energy = rx.thermo._gas.calc_vib_energy(vibfreqs=data.vibfreqs)  # noqa: SLF001
    rot_energy = rx.thermo._gas.calc_rot_energy(moments=moments)  # noqa: SLF001
    trans_energy = rx.thermo._gas.calc_trans_energy()  # noqa: SLF001
    enthalpy = rx.thermo.calc_enthalpy(
        energy=data.energy,
        degeneracy=data.mult,
        moments=moments,
        vibfreqs=data.vibfreqs,
    )
    entropy = rx.thermo.calc_entropy(
        atommasses=data.atommasses,
        energy=data.energy,
        degeneracy=data.mult,
        moments=moments,
        symmetry_number=symmetry_number,
        vibfreqs=data.vibfreqs,
    )
    elec_entropy = rx.thermo._gas.calc_elec_entropy(  # noqa: SLF001
        energy=data.energy,
        degeneracy=data.mult,
    )
    vib_entropy = rx.thermo._gas.calc_vib_entropy(
        vibfreqs=data.vibfreqs,
    )
    rot_entropy = rx.thermo._gas.calc_rot_entropy(  # noqa: SLF001
        moments=moments,
        symmetry_number=symmetry_number,
    )
    trans_entropy = rx.thermo.calc_trans_entropy(atommasses=data.atommasses)
    freeenergy = enthalpy - temperature * entropy
    assert elec_energy / (constants.hartree * constants.N_A) == pytest.approx(
        -232.02314787,
    )  # ORCA logfile
    assert (vib_energy - zpe) / (constants.hartree * constants.N_A) == pytest.approx(
        0.00168358,
        7e-4,
    )  # ORCA logfile
    assert (vib_energy - zpe) / constants.kcal == pytest.approx(
        1.06,
        4e-3,
    )  # ORCA logfile
    assert rot_energy / (constants.hartree * constants.N_A) == pytest.approx(
        0.00141627,
        3e-4,
    )  # ORCA logfile
    assert rot_energy / constants.kcal == pytest.approx(0.89, 2e-3)  # ORCA logfile
    assert trans_energy / (constants.hartree * constants.N_A) == pytest.approx(
        0.00141627,
        6e-6,
    )  # ORCA logfile
    assert trans_energy / constants.kcal == pytest.approx(0.89, 2e-3)  # ORCA logfile
    assert (internal_energy - data.energy - zpe) / (
        constants.hartree * constants.N_A
    ) == pytest.approx(
        0.00451613,
        2e-4,
    )  # ORCA logfile
    assert (internal_energy - data.energy - zpe) / constants.kcal == pytest.approx(
        2.83,
        2e-3,
    )  # ORCA logfile
    assert zpe / (constants.hartree * constants.N_A) == pytest.approx(
        0.09744605,
        2e-4,
    )  # ORCA logfile
    assert zpe / constants.kcal == pytest.approx(61.15, 2e-4)  # ORCA logfile
    assert (internal_energy - data.energy) / (
        constants.hartree * constants.N_A
    ) == pytest.approx(
        0.10196218,
        2e-4,
    )  # ORCA logfile
    assert (internal_energy - data.energy) / constants.kcal == pytest.approx(
        63.98,
        2e-4,
    )  # ORCA logfile
    assert internal_energy / (constants.hartree * constants.N_A) == pytest.approx(
        -231.92118569,
    )  # ORCA logfile
    assert (enthalpy - internal_energy) / (
        constants.hartree * constants.N_A
    ) == pytest.approx(
        0.00094421,
        3e-5,
    )  # ORCA logfile
    assert (enthalpy - internal_energy) / constants.kcal == pytest.approx(
        0.59,
        5e-3,
    )  # ORCA logfile
    assert temperature * elec_entropy / (
        constants.hartree * constants.N_A
    ) == pytest.approx(
        0.0,
    )  # ORCA logfile
    assert temperature * elec_entropy / constants.kcal == pytest.approx(
        0.0,
    )  # ORCA logfile
    assert temperature * vib_entropy / (
        constants.hartree * constants.N_A
    ) == pytest.approx(
        0.00226991,
        2e-3,
    )  # ORCA logfile
    assert temperature * vib_entropy / constants.kcal == pytest.approx(
        1.42,
        4e-3,
    )  # ORCA logfile
    assert temperature * rot_entropy / (
        constants.hartree * constants.N_A
    ) == pytest.approx(
        0.00987646,
        4e-6,
    )  # ORCA logfile
    assert temperature * rot_entropy / constants.kcal == pytest.approx(
        6.20,
        4e-4,
    )  # ORCA logfile
    assert temperature * trans_entropy / (
        constants.hartree * constants.N_A
    ) == pytest.approx(
        0.01852142,
        4e-6,
    )  # ORCA logfile
    assert temperature * trans_entropy / constants.kcal == pytest.approx(
        11.62,
        3e-4,
    )  # ORCA logfile
    assert enthalpy / (constants.hartree * constants.N_A) == pytest.approx(
        -231.92024148,
    )  # ORCA logfile
    assert -temperature * entropy / (
        constants.hartree * constants.N_A
    ) == pytest.approx(
        -0.03066779,
        1e-4,
    )  # ORCA logfile
    assert -temperature * entropy / constants.kcal == pytest.approx(
        -19.25,
        4e-4,
    )  # ORCA logfile
    assert freeenergy / (constants.hartree * constants.N_A) == pytest.approx(
        -231.95090927,
    )  # ORCA logfile
    assert (freeenergy - data.energy) / (
        constants.hartree * constants.N_A
    ) == pytest.approx(
        0.07223860,
        3e-4,
    )  # ORCA logfile
    assert (freeenergy - data.energy) / constants.kcal == pytest.approx(
        45.33,
        3e-4,
    )  # ORCA logfile


def test_compare_qrrho_with_orca_logfile():  # noqa: PLR0915
    """Ensure we have decent quality for QRRHO thermochemical analysis.

    Values from ORCA logfiles are tested.
    """
    temperature = 298.15

    # triphenylphosphine
    data = rx.io.read_logfile("data/symmetries/triphenylphosphine.out")
    assert np.sum(data.atommasses) == pytest.approx(262.29, 8e-6)
    moments, axes, atomcoords = coords.inertia(data.atommasses, data.atomcoords)
    symmetry_number = coords.symmetry_number(
        coords.find_point_group(data.atommasses, data.atomcoords),
    )
    assert symmetry_number == 3  # noqa: PLR2004
    internal_energy = rx.thermo.calc_internal_energy(
        energy=data.energy,
        degeneracy=data.mult,
        moments=moments,
        vibfreqs=data.vibfreqs,
        qrrho=False,
    )
    zpe = rx.thermo._gas.calc_vib_energy(  # noqa: SLF001
        vibfreqs=data.vibfreqs,
        qrrho=False,
        temperature=0.0,
    )
    elec_energy = rx.thermo._gas.calc_elec_energy(  # noqa: SLF001
        energy=data.energy,
        degeneracy=data.mult,
    )
    vib_energy = rx.thermo._gas.calc_vib_energy(
        vibfreqs=data.vibfreqs,
        qrrho=False,
    )
    rot_energy = rx.thermo._gas.calc_rot_energy(moments=moments)  # noqa: SLF001
    trans_energy = rx.thermo._gas.calc_trans_energy()  # noqa: SLF001
    enthalpy = rx.thermo.calc_enthalpy(
        energy=data.energy,
        degeneracy=data.mult,
        moments=moments,
        vibfreqs=data.vibfreqs,
        qrrho=False,
    )
    entropy = rx.thermo.calc_entropy(
        atommasses=data.atommasses,
        energy=data.energy,
        degeneracy=data.mult,
        moments=moments,
        symmetry_number=symmetry_number,
        vibfreqs=data.vibfreqs,
    )
    elec_entropy = rx.thermo._gas.calc_elec_entropy(  # noqa: SLF001
        energy=data.energy,
        degeneracy=data.mult,
    )
    vib_entropy = rx.thermo._gas.calc_vib_entropy(
        vibfreqs=data.vibfreqs,
    )
    rot_entropy = rx.thermo._gas.calc_rot_entropy(  # noqa: SLF001
        moments=moments,
        symmetry_number=symmetry_number,
    )
    trans_entropy = rx.thermo.calc_trans_entropy(atommasses=data.atommasses)
    freeenergy = enthalpy - temperature * entropy
    assert elec_energy / (constants.hartree * constants.N_A) == pytest.approx(
        -1035.902509903170,
    )  # ORCA logfile
    assert (vib_energy - zpe) / (constants.hartree * constants.N_A) == pytest.approx(
        0.01332826,
        6e-5,
    )  # ORCA logfile
    assert (vib_energy - zpe) / constants.kcal == pytest.approx(
        8.36,
        4e-4,
    )  # ORCA logfile
    assert rot_energy / (constants.hartree * constants.N_A) == pytest.approx(
        0.00141627,
        2e-5,
    )  # ORCA logfile
    assert rot_energy / constants.kcal == pytest.approx(0.89, 2e-3)  # ORCA logfile
    assert trans_energy / (constants.hartree * constants.N_A) == pytest.approx(
        0.00141627,
        6e-6,
    )  # ORCA logfile
    assert trans_energy / constants.kcal == pytest.approx(0.89, 2e-3)  # ORCA logfile
    assert (internal_energy - data.energy - zpe) / (
        constants.hartree * constants.N_A
    ) == pytest.approx(
        0.01616080,
        5e-5,
    )  # ORCA logfile
    assert (internal_energy - data.energy - zpe) / constants.kcal == pytest.approx(
        10.14,
        6e-5,
    )  # ORCA logfile
    assert zpe / (constants.hartree * constants.N_A) == pytest.approx(
        0.26902318,
    )  # ORCA logfile
    assert zpe / constants.kcal == pytest.approx(168.81, 3e-5)  # ORCA logfile
    assert (internal_energy - data.energy) / (
        constants.hartree * constants.N_A
    ) == pytest.approx(
        0.28518398,
        3e-6,
    )  # ORCA logfile
    assert (internal_energy - data.energy) / constants.kcal == pytest.approx(
        178.96,
        3e-5,
    )  # ORCA logfile
    assert internal_energy / (constants.hartree * constants.N_A) == pytest.approx(
        -1035.61732592,
    )  # ORCA logfile
    assert (enthalpy - internal_energy) / (
        constants.hartree * constants.N_A
    ) == pytest.approx(
        0.00094421,
        3e-5,
    )  # ORCA logfile
    assert (enthalpy - internal_energy) / constants.kcal == pytest.approx(
        0.59,
        5e-3,
    )  # ORCA logfile
    assert temperature * elec_entropy / (
        constants.hartree * constants.N_A
    ) == pytest.approx(
        0.0,
    )  # ORCA logfile
    assert temperature * elec_entropy / constants.kcal == pytest.approx(
        0.0,
    )  # ORCA logfile
    assert temperature * vib_entropy / (
        constants.hartree * constants.N_A
    ) == pytest.approx(
        0.02326220,
        2e-3,
    )  # ORCA logfile
    assert temperature * vib_entropy / constants.kcal == pytest.approx(
        14.60,
        2e-3,
    )  # ORCA logfile
    assert temperature * rot_entropy / (
        constants.hartree * constants.N_A
    ) == pytest.approx(
        0.01498023,
        4e-6,
    )  # ORCA logfile
    assert temperature * rot_entropy / constants.kcal == pytest.approx(
        9.40,
        3e-5,
    )  # ORCA logfile
    assert temperature * trans_entropy / (
        constants.hartree * constants.N_A
    ) == pytest.approx(
        0.02023694,
        4e-6,
    )  # ORCA logfile
    assert temperature * trans_entropy / constants.kcal == pytest.approx(
        12.70,
        9e-5,
    )  # ORCA logfile
    assert enthalpy / (constants.hartree * constants.N_A) == pytest.approx(
        -1035.61638171,
    )  # ORCA logfile
    assert -temperature * entropy / (
        constants.hartree * constants.N_A
    ) == pytest.approx(
        -0.058479359999999994,
        6e-4,
    )  # ORCA logfile
    assert -temperature * entropy / constants.kcal == pytest.approx(
        -36.70,
        7e-4,
    )  # ORCA logfile
    assert freeenergy / (constants.hartree * constants.N_A) == pytest.approx(
        -1035.6748610799998,
    )  # ORCA logfile
    assert (freeenergy - data.energy) / (
        constants.hartree * constants.N_A
    ) == pytest.approx(
        0.22764882,
        2e-4,
    )  # ORCA logfile
    assert (freeenergy - data.energy) / constants.kcal == pytest.approx(
        142.85,
        2e-4,
    )  # ORCA logfile


def test_read_logfile():
    """Ensure read_logfile returns correct data."""
    fields = {
        "logfile",
        "energy",
        "mult",
        "atomnos",
        "atommasses",
        "atomcoords",
        "vibdisps",
        "vibfreqs",
        "hessian",
    }

    data = rx.io.read_logfile("data/tanaka1996/UMP2/6-311G(2df,2pd)/Cl·.out")
    assert set(data) == fields
    assert data.logfile == "data/tanaka1996/UMP2/6-311G(2df,2pd)/Cl·.out"
    assert data.energy == pytest.approx(-1206891740.7180765, 3e-5)
    assert data.mult == 2  # noqa: PLR2004
    assert data.atomnos == pytest.approx(np.array([17]))
    assert data.atommasses == pytest.approx(np.array([35.453]))
    assert data.atomcoords == pytest.approx(np.array([[0.0, 0.0, 0.0]]))

    fields.remove("hessian")
    data = rx.io.read_logfile("data/symmetries/chlorobromofluoromethane.out")
    assert set(data) == fields
    assert data.logfile == "data/symmetries/chlorobromofluoromethane.out"
    assert data.energy == -8327995636.7634325  # noqa: PLR2004
    assert data.mult == 1
    assert data.atomnos == pytest.approx(np.array([6, 35, 17, 9, 1]))
    assert data.atommasses == pytest.approx(
        np.array([12.011, 79.9, 35.453, 18.998, 1.008]),
    )
    assert data.atomcoords == pytest.approx(
        np.array(
            [
                [0.045, 0.146641, -0.010545],
                [-1.101907, -0.97695, 1.173387],
                [1.103319, -0.870505, -1.041381],
                [-0.737102, 0.941032, -0.780045],
                [0.69069, 0.759782, 0.658585],
            ],
        ),
    )
    assert len(data.vibfreqs) == 9  # noqa: PLR2004
    assert data.vibfreqs == pytest.approx(
        np.array(
            [
                209.49,
                290.47,
                408.01,
                612.13,
                715.01,
                1077.37,
                1158.79,
                1277.35,
                3016.45,
            ],
        ),
    )


def test_read_logfile_from_orca_xtb():
    """Ensure read_logfile correctly reads an XTB2 ORCA logfile."""
    fields = {
        "logfile",
        "energy",
        "mult",
        "atomnos",
        "atommasses",
        "atomcoords",
        "vibfreqs",
    }

    data = rx.io.read_logfile("data/symmetries/Xe.out")
    assert set(data) == fields
    assert data.logfile == "data/symmetries/Xe.out"
    assert data.energy == -10194122.6419248  # noqa: PLR2004
    assert data.mult == 1
    assert data.atomnos == pytest.approx(np.array([54]))
    assert data.atommasses == pytest.approx(np.array([131.293]))
    assert data.atomcoords == pytest.approx(np.array([[0.0, 0.0, 0.0]]))
    assert len(data.vibfreqs) == 0
    assert data.vibfreqs == pytest.approx(np.array([]))

    data = rx.io.read_logfile("data/symmetries/N-methyl-maleamic-acid.out")
    assert set(data).issubset(fields)
    assert data.logfile == "data/symmetries/N-methyl-maleamic-acid.out"
    assert data.energy == pytest.approx(-77186778.47357602)
    assert data.mult == 1
    assert data.atomnos == pytest.approx(
        np.array([6, 6, 6, 6, 8, 8, 8, 7, 6, 1, 1, 1, 1, 1, 1, 1]),
    )
    assert data.atommasses == pytest.approx(
        np.array(
            [
                12.011,
                12.011,
                12.011,
                12.011,
                15.999,
                15.999,
                15.999,
                14.007,
                12.011,
                1.008,
                1.008,
                1.008,
                1.008,
                1.008,
                1.008,
                1.008,
            ],
        ),
    )
    assert data.atomcoords == pytest.approx(
        np.array(
            [
                [-1.06389997780294, 0.10248761131916, -0.00416378746087],
                [-0.04944623568470, 1.17586574558391, 0.10368017025102],
                [1.28503975399447, 1.09120288104392, 0.06034577486787],
                [2.20926920197202, -0.06832416861567, -0.10617866981935],
                [1.73198710638407, -1.28122618787304, -0.24203736856330],
                [3.41087559081078, 0.14023927997443, -0.11002435534482],
                [-0.79219370406139, -1.09826426021838, -0.16397895569778],
                [-2.32310796978358, 0.53074548200415, 0.08639380290049],
                [-3.46974241527303, -0.34303095787263, 0.00740932800451],
                [-0.46597227757871, 2.16615081446255, 0.23685714714238],
                [1.85285605798893, 2.00795050069988, 0.15981019507090],
                [0.70285753578241, -1.31471691088694, -0.22226384014426],
                [-3.11702504917192, -1.36335069364200, -0.12910355473485],
                [-4.09785973779508, -0.05953555624245, -0.83679648129593],
                [-4.05175012271585, -0.27936126957015, 0.92652687537148],
                [-2.51093775706549, 1.51470768983332, 0.21446371945248],
            ],
        ),
        abs=1e-2,
    )
    assert len(data.vibfreqs) == 42  # noqa: PLR2004
    assert data.vibfreqs == pytest.approx(
        np.array(
            [
                26.8,
                85.95,
                109.07,
                163.28,
                179.43,
                291.63,
                301.61,
                347.92,
                388.08,
                515.51,
                580.7,
                586.41,
                621.38,
                717.99,
                753.83,
                831.9,
                871.1,
                932.0,
                941.24,
                947.76,
                1072.81,
                1099.9,
                1118.57,
                1194.22,
                1219.35,
                1262.62,
                1306.58,
                1353.12,
                1394.68,
                1423.79,
                1449.12,
                1465.74,
                1597.11,
                1652.92,
                1662.95,
                2303.95,
                3001.39,
                3012.59,
                3048.14,
                3049.84,
                3070.26,
                3427.02,
            ],
        ),
    )
