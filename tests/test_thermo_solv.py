#!/usr/bin/env python3

"""Tests for solvation using the `thermo` module."""

import pytest

import overreact as rx
from overreact import _constants as constants
from overreact import _datasets as datasets
from overreact import coords


def test_entropy_liquid_phase():
    """Validate selected values from Table 1 of doi:10.1021/acs.jctc.9b00214.

    Experimental values are actually used as reference.
    """
    # H2O
    data = datasets.logfiles["symmetries"]["water"]
    moments, _, _ = coords.inertia(data.atommasses, data.atomcoords)
    point_group = coords.find_point_group(data.atommasses, data.atomcoords)
    symmetry_number = coords.symmetry_number(point_group)
    gas_entropy = rx.thermo.calc_entropy(
        data.atommasses,
        data.atomnos,
        data.atomcoords,
        moments=moments,
        symmetry_number=symmetry_number,
        vibfreqs=data.vibfreqs,
    )
    assert gas_entropy / constants.calorie == pytest.approx(45.1, 3e-3)
    solv_entropy = rx.thermo.calc_entropy(
        data.atommasses,
        data.atomnos,
        data.atomcoords,
        moments=moments,
        symmetry_number=symmetry_number,
        vibfreqs=data.vibfreqs,
        environment="water",
        method="garza",
    )
    assert (solv_entropy - gas_entropy) / constants.calorie == pytest.approx(
        -28.4,
        abs=constants.kilo / 298.15,
    )

    # CH3OH
    data = datasets.logfiles["symmetries"]["methanol"]
    moments, _, _ = coords.inertia(data.atommasses, data.atomcoords)
    point_group = coords.find_point_group(data.atommasses, data.atomcoords)
    symmetry_number = coords.symmetry_number(point_group)
    gas_entropy = rx.thermo.calc_entropy(
        data.atommasses,
        data.atomnos,
        data.atomcoords,
        moments=moments,
        symmetry_number=symmetry_number,
        vibfreqs=data.vibfreqs,
    )
    assert gas_entropy / constants.calorie == pytest.approx(57.3, 8e-3)
    solv_entropy = rx.thermo.calc_entropy(
        data.atommasses,
        data.atomnos,
        data.atomcoords,
        moments=moments,
        symmetry_number=symmetry_number,
        vibfreqs=data.vibfreqs,
        environment="methanol",
        method="garza",
    )
    assert (solv_entropy - gas_entropy) / constants.calorie == pytest.approx(
        -26.9,
        abs=constants.kilo / 298.15,
    )
    solv_entropy = rx.thermo.calc_entropy(
        data.atommasses,
        data.atomnos,
        data.atomcoords,
        moments=moments,
        symmetry_number=symmetry_number,
        vibfreqs=data.vibfreqs,
        environment="butanol",
        method="garza",
    )
    assert (solv_entropy - gas_entropy) / constants.calorie == pytest.approx(
        -20.8,
        abs=constants.kilo / 298.15,
    )

    # C2H5OH
    data = datasets.logfiles["symmetries"]["ethanol"]
    moments, _, _ = coords.inertia(data.atommasses, data.atomcoords)
    point_group = coords.find_point_group(data.atommasses, data.atomcoords)
    symmetry_number = coords.symmetry_number(point_group)
    gas_entropy = rx.thermo.calc_entropy(
        data.atommasses,
        data.atomnos,
        data.atomcoords,
        moments=moments,
        symmetry_number=symmetry_number,
        vibfreqs=data.vibfreqs,
    )
    assert gas_entropy / constants.calorie == pytest.approx(67.6, 5e-2)
    solv_entropy = rx.thermo.calc_entropy(
        data.atommasses,
        data.atomnos,
        data.atomcoords,
        moments=moments,
        symmetry_number=symmetry_number,
        vibfreqs=data.vibfreqs,
        environment="ethanol",
        method="garza",
    )
    assert (solv_entropy - gas_entropy) / constants.calorie == pytest.approx(
        -29.4,
        abs=constants.kilo / 298.15,
    )
    solv_entropy = rx.thermo.calc_entropy(
        data.atommasses,
        data.atomnos,
        data.atomcoords,
        moments=moments,
        symmetry_number=symmetry_number,
        vibfreqs=data.vibfreqs,
        environment="water",
        method="garza",
    )
    assert (solv_entropy - gas_entropy) / constants.calorie == pytest.approx(
        -31.6,
        abs=constants.kilo / 298.15,
    )
    solv_entropy = rx.thermo.calc_entropy(
        data.atommasses,
        data.atomnos,
        data.atomcoords,
        moments=moments,
        symmetry_number=symmetry_number,
        vibfreqs=data.vibfreqs,
        environment="butanol",
        method="garza",
    )
    assert (solv_entropy - gas_entropy) / constants.calorie == pytest.approx(
        -23.4,
        abs=constants.kilo / 298.15,
    )
    solv_entropy = rx.thermo.calc_entropy(
        data.atommasses,
        data.atomnos,
        data.atomcoords,
        moments=moments,
        symmetry_number=symmetry_number,
        vibfreqs=data.vibfreqs,
        environment="benzene",
        method="garza",
    )
    assert (solv_entropy - gas_entropy) / constants.calorie == pytest.approx(
        -18.6,
        abs=constants.kilo / 298.15,
    )


def test_translational_entropy_liquid_phase():
    """Validate calculated volumes from Table S3 of doi:10.1039/C9CP03226F.

    The original data seems to have errors, some severe. I *assumed* this is
    due to too few decimal places in the free volume and adjusted the precision
    for some entries accordingly.

    The same systems are probed for the method of doi:10.1021/acs.jctc.9b00214
    in water and the gas phase translational entropy is also tested.
    """
    # H2O
    data = datasets.logfiles["symmetries"]["water"]
    vdw_volume, cav_volume, err = coords.get_molecular_volume(
        data.atomnos,
        data.atomcoords,
        full_output=True,
        method="izato",
    )
    assert err < 0.263
    assert cav_volume == pytest.approx(30.93, 8e-2)
    assert vdw_volume == pytest.approx(19.16, 8e-2)
    free_volume = rx.thermo._solv.molar_free_volume(
        data.atomnos,
        data.atomcoords,
        method="izato",
    )

    assert rx.thermo.calc_trans_entropy(
        data.atommasses,
        data.atomnos,
        data.atomcoords,
    ) == pytest.approx(144.80408548676766, 5e-2)
    assert rx.thermo.calc_trans_entropy(
        data.atommasses,
        data.atomnos,
        data.atomcoords,
        environment="water",
        method="garza",
    ) == pytest.approx(94.56871145467743, 5e-2)
    assert rx.thermo.calc_trans_entropy(
        data.atommasses,
        data.atomnos,
        data.atomcoords,
        environment="water",
        method="izato",
    ) == pytest.approx(37.36, 5e-2)

    # CH3OH
    data = datasets.logfiles["symmetries"]["methanol"]
    vdw_volume, cav_volume, err = coords.get_molecular_volume(
        data.atomnos,
        data.atomcoords,
        full_output=True,
        method="izato",
    )
    assert err < 0.263
    assert cav_volume == pytest.approx(55.28, 7e-2)
    assert vdw_volume == pytest.approx(36.64, 8e-2)
    free_volume = rx.thermo._solv.molar_free_volume(
        data.atomnos,
        data.atomcoords,
        method="izato",
    )
    assert free_volume / (constants.angstrom**3 * constants.N_A) == pytest.approx(
        0.116,
        5e-2,
    )
    assert rx.thermo.calc_trans_entropy(
        data.atommasses,
        data.atomnos,
        data.atomcoords,
    ) == pytest.approx(151.98582061379608, 1e-2)
    assert rx.thermo.calc_trans_entropy(
        data.atommasses,
        data.atomnos,
        data.atomcoords,
        environment="water",
        method="garza",
    ) == pytest.approx(105.02817010903456, 1e-2)
    assert rx.thermo.calc_trans_entropy(
        data.atommasses,
        data.atomnos,
        data.atomcoords,
        environment="water",
        method="izato",
    ) == pytest.approx(45.85, 1e-2)

    # C2H5OH
    data = datasets.logfiles["symmetries"]["ethanol"]
    vdw_volume, cav_volume, err = coords.get_molecular_volume(
        data.atomnos,
        data.atomcoords,
        full_output=True,
        method="izato",
    )
    assert err < 0.263
    assert vdw_volume == pytest.approx(53.66, 8e-2)
    free_volume = rx.thermo._solv.molar_free_volume(
        data.atomnos,
        data.atomcoords,
        method="izato",
    )
    assert free_volume / (constants.angstrom**3 * constants.N_A) == pytest.approx(
        0.125,
        6e-2,
    )
    assert rx.thermo.calc_trans_entropy(
        data.atommasses,
        data.atomnos,
        data.atomcoords,
    ) == pytest.approx(156.51420201431782, 5e-2)
    assert rx.thermo.calc_trans_entropy(
        data.atommasses,
        data.atomnos,
        data.atomcoords,
        environment="water",
        method="garza",
    ) == pytest.approx(111.58706012673835, 5e-2)
    assert rx.thermo.calc_trans_entropy(
        data.atommasses,
        data.atomnos,
        data.atomcoords,
        environment="water",
        method="izato",
    ) == pytest.approx(51.01, 5e-2)

    # 1-C3H7OH
    data = datasets.logfiles["symmetries"]["1-propanol"]
    vdw_volume, cav_volume, err = coords.get_molecular_volume(
        data.atomnos,
        data.atomcoords,
        full_output=True,
        method="izato",
    )
    assert err < 0.263
    assert cav_volume == pytest.approx(100.25, 7e-2)
    free_volume = rx.thermo._solv.molar_free_volume(
        data.atomnos,
        data.atomcoords,
        method="izato",
    )
    assert free_volume / (constants.angstrom**3 * constants.N_A) == pytest.approx(
        0.144,
        3e-2,
    )
    assert rx.thermo.calc_trans_entropy(
        data.atommasses,
        data.atomnos,
        data.atomcoords,
    ) == pytest.approx(159.8292158120477, 1e-2)
    assert rx.thermo.calc_trans_entropy(
        data.atommasses,
        data.atomnos,
        data.atomcoords,
        environment="water",
        method="garza",
    ) == pytest.approx(116.44404809399771, 1e-2)
    assert rx.thermo.calc_trans_entropy(
        data.atommasses,
        data.atomnos,
        data.atomcoords,
        environment="water",
        method="izato",
    ) == pytest.approx(55.70, 1e-2)

    # 2-C3H7OH
    data = datasets.logfiles["symmetries"]["2-propanol"]
    vdw_volume, cav_volume, err = coords.get_molecular_volume(
        data.atomnos,
        data.atomcoords,
        full_output=True,
        method="izato",
    )
    assert err < 0.263
    assert cav_volume == pytest.approx(100.50, 7e-2)
    assert vdw_volume == pytest.approx(69.95, 8e-2)
    free_volume = rx.thermo._solv.molar_free_volume(
        data.atomnos,
        data.atomcoords,
        method="izato",
    )
    assert free_volume / (constants.angstrom**3 * constants.N_A) == pytest.approx(
        0.148,
        8e-2,
    )
    assert rx.thermo.calc_trans_entropy(
        data.atommasses,
        data.atomnos,
        data.atomcoords,
    ) == pytest.approx(159.8292158120477, 5e-2)
    assert rx.thermo.calc_trans_entropy(
        data.atommasses,
        data.atomnos,
        data.atomcoords,
        environment="water",
        method="garza",
    ) == pytest.approx(116.44717525933929, 5e-2)
    assert rx.thermo.calc_trans_entropy(
        data.atommasses,
        data.atomnos,
        data.atomcoords,
        environment="water",
        method="izato",
    ) == pytest.approx(46.81, 1.9e-1)

    # 1-C4H9OH
    data = datasets.logfiles["symmetries"]["1-butanol"]
    vdw_volume, cav_volume, err = coords.get_molecular_volume(
        data.atomnos,
        data.atomcoords,
        full_output=True,
        method="izato",
    )
    assert err < 0.263
    assert cav_volume == pytest.approx(123.34, 7e-2)
    assert vdw_volume == pytest.approx(87.13, 8e-2)
    free_volume = rx.thermo._solv.molar_free_volume(
        data.atomnos,
        data.atomcoords,
        method="izato",
    )
    assert free_volume / (constants.angstrom**3 * constants.N_A) == pytest.approx(
        0.161,
        3e-2,
    )
    assert rx.thermo.calc_trans_entropy(
        data.atommasses,
        data.atomnos,
        data.atomcoords,
    ) == pytest.approx(162.4455600896596, 5e-2)
    assert rx.thermo.calc_trans_entropy(
        data.atommasses,
        data.atomnos,
        data.atomcoords,
        environment="water",
        method="garza",
    ) == pytest.approx(120.30330699244473, 5e-2)
    assert rx.thermo.calc_trans_entropy(
        data.atommasses,
        data.atomnos,
        data.atomcoords,
        environment="water",
        method="izato",
    ) == pytest.approx(59.90, 5e-2)

    # 2-C4H9OH
    data = datasets.logfiles["symmetries"]["2-butanol"]
    vdw_volume, cav_volume, err = coords.get_molecular_volume(
        data.atomnos,
        data.atomcoords,
        full_output=True,
        method="izato",
    )
    assert err < 0.263
    assert cav_volume == pytest.approx(122.05, 6e-2)
    assert vdw_volume == pytest.approx(86.83, 8e-2)
    free_volume = rx.thermo._solv.molar_free_volume(
        data.atomnos,
        data.atomcoords,
        method="izato",
    )
    assert free_volume / (constants.angstrom**3 * constants.N_A) == pytest.approx(
        0.151,
        6e-2,
    )
    assert rx.thermo.calc_trans_entropy(
        data.atommasses,
        data.atomnos,
        data.atomcoords,
    ) == pytest.approx(162.4455600896596, 5e-2)
    assert rx.thermo.calc_trans_entropy(
        data.atommasses,
        data.atomnos,
        data.atomcoords,
        environment="water",
        method="garza",
    ) == pytest.approx(120.30491171022815, 5e-2)
    assert rx.thermo.calc_trans_entropy(
        data.atommasses,
        data.atomnos,
        data.atomcoords,
        environment="water",
        method="izato",
    ) == pytest.approx(51.64, 1.5e-1)

    # i-C4H9OH
    data = datasets.logfiles["symmetries"]["2-methyl-2-propanol"]
    vdw_volume, cav_volume, err = coords.get_molecular_volume(
        data.atomnos,
        data.atomcoords,
        full_output=True,
        method="izato",
    )
    assert err < 0.2550
    assert cav_volume == pytest.approx(122.17, 7e-2)
    assert vdw_volume == pytest.approx(86.83, 8e-2)
    free_volume = rx.thermo._solv.molar_free_volume(
        data.atomnos,
        data.atomcoords,
        method="izato",
    )
    assert free_volume / (constants.angstrom**3 * constants.N_A) == pytest.approx(
        0.152,
        3e-2,
    )
    assert rx.thermo.calc_trans_entropy(
        data.atommasses,
        data.atomnos,
        data.atomcoords,
    ) == pytest.approx(162.4455600896596, 1e-2)
    assert rx.thermo.calc_trans_entropy(
        data.atommasses,
        data.atomnos,
        data.atomcoords,
        environment="water",
        method="garza",
    ) == pytest.approx(120.29352768299235, 1e-2)
    assert rx.thermo.calc_trans_entropy(
        data.atommasses,
        data.atomnos,
        data.atomcoords,
        environment="water",
        method="izato",
    ) == pytest.approx(58.71, 6e-3)

    # HCOOH
    data = datasets.logfiles["symmetries"]["formic-acid"]
    vdw_volume, cav_volume, err = coords.get_molecular_volume(
        data.atomnos,
        data.atomcoords,
        full_output=True,
        method="izato",
    )
    assert err < 0.263
    assert cav_volume == pytest.approx(59.41, 5e-2)
    assert vdw_volume == pytest.approx(39.04, 4e-2)
    free_volume = rx.thermo._solv.molar_free_volume(
        data.atomnos,
        data.atomcoords,
        method="izato",
    )
    assert free_volume / (constants.angstrom**3 * constants.N_A) == pytest.approx(
        0.133,
        7e-2,
    )
    assert rx.thermo.calc_trans_entropy(
        data.atommasses,
        data.atomnos,
        data.atomcoords,
    ) == pytest.approx(156.50228474338718, 5e-2)
    assert rx.thermo.calc_trans_entropy(
        data.atommasses,
        data.atomnos,
        data.atomcoords,
        environment="water",
        method="garza",
    ) == pytest.approx(109.8726286196741, 5e-2)
    assert rx.thermo.calc_trans_entropy(
        data.atommasses,
        data.atomnos,
        data.atomcoords,
        environment="water",
        method="izato",
    ) == pytest.approx(51.45, 5e-2)

    # CH3COOH
    data = datasets.logfiles["symmetries"]["acetic-acid"]
    vdw_volume, cav_volume, err = coords.get_molecular_volume(
        data.atomnos,
        data.atomcoords,
        full_output=True,
        method="izato",
    )
    assert err < 0.263
    assert cav_volume == pytest.approx(82.32, 5e-2)
    assert vdw_volume == pytest.approx(55.00, 4e-2)
    free_volume = rx.thermo._solv.molar_free_volume(
        data.atomnos,
        data.atomcoords,
        method="izato",
    )
    assert free_volume / (constants.angstrom**3 * constants.N_A) == pytest.approx(
        0.164,
        1.2e-1,
    )
    assert rx.thermo.calc_trans_entropy(
        data.atommasses,
        data.atomnos,
        data.atomcoords,
    ) == pytest.approx(159.820081168819, 5e-2)
    assert rx.thermo.calc_trans_entropy(
        data.atommasses,
        data.atomnos,
        data.atomcoords,
        environment="water",
        method="garza",
    ) == pytest.approx(115.13015825195785, 5e-2)
    assert rx.thermo.calc_trans_entropy(
        data.atommasses,
        data.atomnos,
        data.atomcoords,
        environment="water",
        method="izato",
    ) == pytest.approx(56.52, 5e-2)

    # CH3CN
    data = datasets.logfiles["symmetries"]["acetonitrile"]
    vdw_volume, cav_volume, err = coords.get_molecular_volume(
        data.atomnos,
        data.atomcoords,
        full_output=True,
        method="izato",
    )
    assert err < 0.263
    assert cav_volume == pytest.approx(68.88, 5e-2)
    assert vdw_volume == pytest.approx(45.26, 4e-2)
    free_volume = rx.thermo._solv.molar_free_volume(
        data.atomnos,
        data.atomcoords,
        method="izato",
    )
    assert free_volume / (constants.angstrom**3 * constants.N_A) == pytest.approx(
        0.153,
        5e-2,
    )
    assert rx.thermo.calc_trans_entropy(
        data.atommasses,
        data.atomnos,
        data.atomcoords,
    ) == pytest.approx(155.0765105631046, 5e-2)
    assert rx.thermo.calc_trans_entropy(
        data.atommasses,
        data.atomnos,
        data.atomcoords,
        environment="water",
        method="garza",
    ) == pytest.approx(109.28222352094771, 5e-2)
    assert rx.thermo.calc_trans_entropy(
        data.atommasses,
        data.atomnos,
        data.atomcoords,
        environment="water",
        method="izato",
    ) == pytest.approx(51.25, 5e-2)

    # CH3NO2
    data = datasets.logfiles["symmetries"]["nitromethane"]
    vdw_volume, cav_volume, err = coords.get_molecular_volume(
        data.atomnos,
        data.atomcoords,
        full_output=True,
        method="izato",
    )
    assert err < 0.263
    assert cav_volume == pytest.approx(77.29, 5e-2)
    assert vdw_volume == pytest.approx(51.53, 4e-2)
    free_volume = rx.thermo._solv.molar_free_volume(
        data.atomnos,
        data.atomcoords,
        method="izato",
    )
    assert free_volume / (constants.angstrom**3 * constants.N_A) == pytest.approx(
        0.156,
        4e-2,
    )
    assert rx.thermo.calc_trans_entropy(
        data.atommasses,
        data.atomnos,
        data.atomcoords,
    ) == pytest.approx(160.02360092304738, 5e-2)
    assert rx.thermo.calc_trans_entropy(
        data.atommasses,
        data.atomnos,
        data.atomcoords,
        environment="water",
        method="garza",
    ) == pytest.approx(114.90481716308736, 5e-2)
    assert rx.thermo.calc_trans_entropy(
        data.atommasses,
        data.atomnos,
        data.atomcoords,
        environment="water",
        method="izato",
    ) == pytest.approx(56.34, 5e-2)

    # Acetone
    data = datasets.logfiles["symmetries"]["acetone"]
    vdw_volume, cav_volume, err = coords.get_molecular_volume(
        data.atomnos,
        data.atomcoords,
        full_output=True,
        method="izato",
    )
    assert err < 0.263
    assert cav_volume == pytest.approx(94.20, 6e-2)
    assert vdw_volume == pytest.approx(65.29, 7e-2)
    free_volume = rx.thermo._solv.molar_free_volume(
        data.atomnos,
        data.atomcoords,
        method="izato",
    )
    assert rx.thermo.calc_trans_entropy(
        data.atommasses,
        data.atomnos,
        data.atomcoords,
    ) == pytest.approx(159.4036577560442, 5e-2)
    assert rx.thermo.calc_trans_entropy(
        data.atommasses,
        data.atomnos,
        data.atomcoords,
        environment="water",
        method="garza",
    ) == pytest.approx(115.54256726917262, 5e-2)
    assert rx.thermo.calc_trans_entropy(
        data.atommasses,
        data.atomnos,
        data.atomcoords,
        environment="water",
        method="izato",
    ) == pytest.approx(55.01, 5e-2)

    # DMSO
    data = datasets.logfiles["symmetries"]["dimethyl-sulfoxide"]
    vdw_volume, cav_volume, err = coords.get_molecular_volume(
        data.atomnos,
        data.atomcoords,
        full_output=True,
        method="izato",
    )
    assert err < 0.263
    assert cav_volume == pytest.approx(104.64, 5e-2)
    assert vdw_volume == pytest.approx(71.19, 4e-2)
    free_volume = rx.thermo._solv.molar_free_volume(
        data.atomnos,
        data.atomcoords,
        method="izato",
    )
    assert rx.thermo.calc_trans_entropy(
        data.atommasses,
        data.atomnos,
        data.atomcoords,
    ) == pytest.approx(163.10201307782876, 1e-2)
    assert rx.thermo.calc_trans_entropy(
        data.atommasses,
        data.atomnos,
        data.atomcoords,
        environment="water",
        method="garza",
    ) == pytest.approx(119.86466911598932, 1e-2)
    assert rx.thermo.calc_trans_entropy(
        data.atommasses,
        data.atomnos,
        data.atomcoords,
        environment="water",
        method="izato",
    ) == pytest.approx(60.73, 1e-2)

    # THF
    data = datasets.logfiles["symmetries"]["tetrahydrofuran"]
    vdw_volume, cav_volume, err = coords.get_molecular_volume(
        data.atomnos,
        data.atomcoords,
        full_output=True,
        method="izato",
    )
    assert err < 0.263
    assert cav_volume == pytest.approx(108.71, 7e-2)
    assert vdw_volume == pytest.approx(76.59, 6e-2)
    free_volume = rx.thermo._solv.molar_free_volume(
        data.atomnos,
        data.atomcoords,
        method="izato",
    )
    assert free_volume / (constants.angstrom**3 * constants.N_A) == pytest.approx(
        0.145,
        9e-2,
    )
    assert rx.thermo.calc_trans_entropy(
        data.atommasses,
        data.atomnos,
        data.atomcoords,
    ) == pytest.approx(162.10165626567044, 1e-2)
    assert rx.thermo.calc_trans_entropy(
        data.atommasses,
        data.atomnos,
        data.atomcoords,
        environment="water",
        method="garza",
    ) == pytest.approx(119.25508474102487, 1e-2)
    assert rx.thermo.calc_trans_entropy(
        data.atommasses,
        data.atomnos,
        data.atomcoords,
        environment="water",
        method="izato",
    ) == pytest.approx(57.82, 2e-2)

    # Benzene
    data = datasets.logfiles["symmetries"]["benzene"]
    vdw_volume, cav_volume, err = coords.get_molecular_volume(
        data.atomnos,
        data.atomcoords,
        full_output=True,
        method="izato",
    )
    assert err < 0.263
    assert cav_volume == pytest.approx(118.78, 5e-2)
    assert vdw_volume == pytest.approx(83.36, 4e-2)
    free_volume = rx.thermo._solv.molar_free_volume(
        data.atomnos,
        data.atomcoords,
        method="izato",
    )
    assert free_volume / (constants.angstrom**3 * constants.N_A) == pytest.approx(
        0.164,
        3e-2,
    )
    assert rx.thermo.calc_trans_entropy(
        data.atommasses,
        data.atomnos,
        data.atomcoords,
    ) == pytest.approx(163.09961840530607, 5e-2)
    assert rx.thermo.calc_trans_entropy(
        data.atommasses,
        data.atomnos,
        data.atomcoords,
        environment="water",
        method="garza",
    ) == pytest.approx(120.42382713494072, 5e-2)
    assert rx.thermo.calc_trans_entropy(
        data.atommasses,
        data.atomnos,
        data.atomcoords,
        environment="water",
        method="izato",
    ) == pytest.approx(74.42, 2e-1)

    # CCl4
    data = datasets.logfiles["symmetries"]["tetrachloromethane"]
    vdw_volume, cav_volume, err = coords.get_molecular_volume(
        data.atomnos,
        data.atomcoords,
        full_output=True,
        method="izato",
    )
    assert err < 0.263
    assert cav_volume == pytest.approx(128.54, 5e-2)
    assert vdw_volume == pytest.approx(84.22, 2e-2)
    free_volume = rx.thermo._solv.molar_free_volume(
        data.atomnos,
        data.atomcoords,
        method="izato",
    )
    assert free_volume / (constants.angstrom**3 * constants.N_A) == pytest.approx(
        0.292,
        6e-2,
    )
    assert rx.thermo.calc_trans_entropy(
        data.atommasses,
        data.atomnos,
        data.atomcoords,
    ) == pytest.approx(171.5508534254953, 5e-2)
    assert rx.thermo.calc_trans_entropy(
        data.atommasses,
        data.atomnos,
        data.atomcoords,
        environment="water",
        method="garza",
    ) == pytest.approx(129.19786258624222, 5e-2)
    assert rx.thermo.calc_trans_entropy(
        data.atommasses,
        data.atomnos,
        data.atomcoords,
        environment="water",
        method="izato",
    ) == pytest.approx(61.39, 2e-1)

    # C5H12
    data = datasets.logfiles["symmetries"]["n-pentane"]
    vdw_volume, cav_volume, err = coords.get_molecular_volume(
        data.atomnos,
        data.atomcoords,
        full_output=True,
        method="izato",
    )
    assert err < 0.263
    assert cav_volume == pytest.approx(134.53, 8e-2)
    assert vdw_volume == pytest.approx(95.84, 9e-2)
    free_volume = rx.thermo._solv.molar_free_volume(
        data.atomnos,
        data.atomcoords,
        method="izato",
    )
    assert free_volume / (constants.angstrom**3 * constants.N_A) == pytest.approx(
        0.164,
        3e-2,
    )
    assert rx.thermo.calc_trans_entropy(
        data.atommasses,
        data.atomnos,
        data.atomcoords,
    ) == pytest.approx(162.10926422571157, 1e-2)
    assert rx.thermo.calc_trans_entropy(
        data.atommasses,
        data.atomnos,
        data.atomcoords,
        environment="water",
        method="garza",
    ) == pytest.approx(120.50548449806315, 1e-2)
    assert rx.thermo.calc_trans_entropy(
        data.atommasses,
        data.atomnos,
        data.atomcoords,
        environment="water",
        method="izato",
    ) == pytest.approx(58.84, 1e-2)

    # C6H14
    data = datasets.logfiles["symmetries"]["n-hexane"]
    vdw_volume, cav_volume, err = coords.get_molecular_volume(
        data.atomnos,
        data.atomcoords,
        full_output=True,
        method="izato",
    )
    assert cav_volume == pytest.approx(156.61, 8e-2)
    assert vdw_volume == pytest.approx(112.34, 9e-2)
    free_volume = rx.thermo._solv.molar_free_volume(
        data.atomnos,
        data.atomcoords,
        method="izato",
    )
    assert free_volume / (constants.angstrom**3 * constants.N_A) == pytest.approx(
        0.180,
        4e-2,
    )
    assert rx.thermo.calc_trans_entropy(
        data.atommasses,
        data.atomnos,
        data.atomcoords,
    ) == pytest.approx(164.3249077891098, 1e-2)
    assert rx.thermo.calc_trans_entropy(
        data.atommasses,
        data.atomnos,
        data.atomcoords,
        environment="water",
        method="garza",
    ) == pytest.approx(123.65659036939897, 1e-2)
    assert rx.thermo.calc_trans_entropy(
        data.atommasses,
        data.atomnos,
        data.atomcoords,
        environment="water",
        method="izato",
    ) == pytest.approx(61.98, 1e-2)

    # cyc-C6H12
    data = datasets.logfiles["symmetries"]["cyclohexane-chair"]
    vdw_volume, cav_volume, err = coords.get_molecular_volume(
        data.atomnos,
        data.atomcoords,
        full_output=True,
        method="izato",
    )
    assert err < 0.263
    assert cav_volume == pytest.approx(138.74, 7e-2)
    assert vdw_volume == pytest.approx(100.50, 7e-2)
    free_volume = rx.thermo._solv.molar_free_volume(
        data.atomnos,
        data.atomcoords,
        method="izato",
    )
    assert free_volume / (constants.angstrom**3 * constants.N_A) == pytest.approx(
        0.147,
        8e-2,
    )
    assert rx.thermo.calc_trans_entropy(
        data.atommasses,
        data.atomnos,
        data.atomcoords,
    ) == pytest.approx(164.02968518741582, 1e-2)
    assert rx.thermo.calc_trans_entropy(
        data.atommasses,
        data.atomnos,
        data.atomcoords,
        environment="water",
        method="garza",
    ) == pytest.approx(122.79118854674996, 1e-2)
    assert rx.thermo.calc_trans_entropy(
        data.atommasses,
        data.atomnos,
        data.atomcoords,
        environment="water",
        method="izato",
    ) == pytest.approx(59.83, 1e-2)


def test_sackur_tetrode_given_free_volumes():
    """Reproduce calculated values from Table S3 of doi:10.1039/C9CP03226F.

    The original data seems to have errors, some severe. I assumed this is due
    to too few decimal places in the free volume and adjusted the precision for
    each entry accordingly.
    """
    # H2O
    assert rx.thermo._gas._sackur_tetrode(
        18.01,
        0.0993e-30 * constants.N_A,
    ) == pytest.approx(37.36, 1e-4)

    # CH3OH
    assert rx.thermo._gas._sackur_tetrode(
        32.04,
        0.116e-30 * constants.N_A,
    ) == pytest.approx(45.85, 1e-3)

    # C2H5OH
    assert rx.thermo._gas._sackur_tetrode(
        46.05,
        0.125e-30 * constants.N_A,
    ) == pytest.approx(51.01, 1e-3)

    # 1-C3H7OH
    assert rx.thermo._gas._sackur_tetrode(
        60.06,
        0.144e-30 * constants.N_A,
    ) == pytest.approx(55.70, 1e-2)

    # 2-C3H7OH
    assert rx.thermo._gas._sackur_tetrode(
        60.06,
        0.148e-30 * constants.N_A,
    ) == pytest.approx(46.81, 1.9e-1)

    # 1-C4H9OH
    assert rx.thermo._gas._sackur_tetrode(
        74.07,
        0.161e-30 * constants.N_A,
    ) == pytest.approx(59.90, 5e-2)

    # 2-C4H9OH
    assert rx.thermo._gas._sackur_tetrode(
        74.07,
        0.151e-30 * constants.N_A,
    ) == pytest.approx(51.64, 1.4e-1)

    # i-C4H9OH
    assert rx.thermo._gas._sackur_tetrode(
        74.07,
        0.152e-30 * constants.N_A,
    ) == pytest.approx(58.71, 1e-2)

    # HCOOH
    assert rx.thermo._gas._sackur_tetrode(
        46.01,
        0.133e-30 * constants.N_A,
    ) == pytest.approx(51.45, 1e-3)

    # CH3COOH
    assert rx.thermo._gas._sackur_tetrode(
        60.02,
        0.164e-30 * constants.N_A,
    ) == pytest.approx(56.52, 1e-3)

    # CH3CN
    assert rx.thermo._gas._sackur_tetrode(
        41.03,
        0.153e-30 * constants.N_A,
    ) == pytest.approx(51.25, 1e-3)

    # CH3NO2
    assert rx.thermo._gas._sackur_tetrode(
        61.02,
        0.156e-30 * constants.N_A,
    ) == pytest.approx(56.34, 1e-4)

    # Acetone
    assert rx.thermo._gas._sackur_tetrode(
        58.04,
        0.143e-30 * constants.N_A,
    ) == pytest.approx(55.01, 1e-3)

    # DMSO
    assert rx.thermo._gas._sackur_tetrode(
        78.01,
        0.183e-30 * constants.N_A,
    ) == pytest.approx(60.73, 1e-4)

    # THF
    assert rx.thermo._gas._sackur_tetrode(
        72.06,
        0.145e-30 * constants.N_A,
    ) == pytest.approx(57.82, 1e-3)

    # Benzene
    assert rx.thermo._gas._sackur_tetrode(
        78.05,
        0.164e-30 * constants.N_A,
    ) == pytest.approx(74.42, 2e-1)

    # CCl4
    assert rx.thermo._gas._sackur_tetrode(
        151.88,
        0.292e-30 * constants.N_A,
    ) == pytest.approx(61.39, 1.9e-1)

    # C5H12
    assert rx.thermo._gas._sackur_tetrode(
        72.09,
        0.164e-30 * constants.N_A,
    ) == pytest.approx(58.84, 1e-3)

    # C6H14
    assert rx.thermo._gas._sackur_tetrode(
        87.12,
        0.180e-30 * constants.N_A,
    ) == pytest.approx(61.98, 1e-3)

    # cyc-C6H12
    assert rx.thermo._gas._sackur_tetrode(
        84.09,
        0.147e-30 * constants.N_A,
    ) == pytest.approx(59.83, 1e-3)


def test_change_reference_state_works_for_gas_to_liquid_standard_states():
    """Ensure change_reference_state works for gas to solv. state change."""
    # different reference temperatures
    assert 200.0 * rx.change_reference_state(temperature=200.0) / (
        constants.kcal
    ) == pytest.approx(1.10, 5e-2)
    assert 298.15 * rx.change_reference_state() / constants.kcal == pytest.approx(
        1.85,
        5e-2,
    )
    assert 300.0 * rx.change_reference_state(temperature=300.0) / (
        constants.kcal
    ) == pytest.approx(1.90, 1e-2)
    assert 400.0 * rx.change_reference_state(temperature=400.0) / (
        constants.kcal
    ) == pytest.approx(2.70, 5e-2)

    # different reference pressures
    temperature = 298.15
    assert temperature * rx.change_reference_state(
        1.0 / constants.liter,
        1.0 / rx.thermo.molar_volume(temperature, constants.atm),
        temperature=temperature,
    ) / constants.kcal == pytest.approx(1.89, 1e-2)
    assert temperature * rx.change_reference_state(
        1.0 / constants.liter,
        1.0 / rx.thermo.molar_volume(temperature, constants.bar),
        temperature=temperature,
    ) / constants.kcal == pytest.approx(1.90, 1e-2)

    # volumes instead of concentrations
    temperature = 298.15
    assert -temperature * rx.change_reference_state(
        constants.liter,
        rx.thermo.molar_volume(temperature, constants.atm),
        temperature=temperature,
    ) / constants.kcal == pytest.approx(1.89, 1e-2)
