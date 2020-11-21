#!/usr/bin/env python3

"""Tests for rates module."""

import numpy as np
import pytest

from overreact import api
from overreact import constants
from overreact import core
from overreact import rates
from overreact import _thermo


def test_sanity_for_chemical_kinetics():
    """Ensure we have decent quality for chemical kinetics analysis.

    This partially ensures we do similar analysis as Gaussian, see
    https://gaussian.com/thermo/.
    """
    freeenergies_H = np.array(
        [-98.579127, -454.557870, -553.109488, -98.001318, -455.146251]
    )
    freeenergies_D = np.array(
        [-98.582608, -454.557870, -553.110424, -98.001318, -455.149092]
    )

    scheme = core.parse_reactions("FH + Cl -> FHCl‡ -> F + HCl")
    delta_freeenergies_H = _thermo.get_delta(scheme.B, freeenergies_H)[0]
    delta_freeenergies_D = _thermo.get_delta(scheme.B, freeenergies_D)[0]
    assert delta_freeenergies_H == pytest.approx(0.027509)
    assert delta_freeenergies_H * (constants.hartree * constants.N_A) / (
        constants.kcal
    ) == pytest.approx(17.26, 2e-4)
    assert delta_freeenergies_D == pytest.approx(0.030054)
    assert delta_freeenergies_D * (constants.hartree * constants.N_A) / (
        constants.kcal
    ) == pytest.approx(18.86, 5e-5)

    k_H = api.get_k(
        scheme,
        delta_freeenergies=delta_freeenergies_H * constants.hartree * constants.N_A,
        volume=1.0,
        scale="atm-1 s-1",
    )
    k_D = api.get_k(
        scheme,
        delta_freeenergies=delta_freeenergies_D * constants.hartree * constants.N_A,
        volume=1.0,
        scale="atm-1 s-1",
    )
    assert k_H == pytest.approx(1.38, 4e-4)
    assert k_D == pytest.approx(0.0928, 5e-3)

    kie = k_H / k_D
    assert kie == pytest.approx(1.38 / 0.0928, 4e-3)


def test_basic_example_for_chemical_kinetics():
    """Ensure we can reproduce a basic example from doi:10.1002/qua.25686.

    This uses raw data and no calls from overreact.api.
    """
    temperatures = np.array([200, 298.15, 300, 400])
    delta_freeenergy = np.array([8.0, 10.3, 10.3, 12.6])
    delta_freeenergy -= (
        temperatures
        * _thermo.change_reference_state(temperature=temperatures)
        / (constants.kilo * constants.calorie)
    )  # 1 atm to 1 M
    assert delta_freeenergy == pytest.approx([6.9, 8.4, 8.4, 9.9], 8e-3)

    delta_freeenergy += (
        temperatures
        * _thermo.change_reference_state(4, 1, sign=-1, temperature=temperatures)
        / (constants.kilo * constants.calorie)
    )  # 4-fold symmetry TS
    assert delta_freeenergy == pytest.approx([6.3, 7.6, 7.6, 8.8], 9e-3)

    k = rates.eyring(
        delta_freeenergy * constants.kilo * constants.calorie,
        temperature=temperatures,
        molecularity=2,
    )
    k = rates.convert_rate_constant(k, "cm3 particle-1 s-1", molecularity=2)
    assert np.log10(k) == pytest.approx(
        [-15.65757732, -14.12493874, -14.10237291, -13.25181197], 4.1e-2
    )


def test_eyring_calculates_reaction_barrier():
    """Ensure Eyring rates are correct."""
    barrier1 = 17.26 * constants.kcal
    barrier2 = 18.86 * constants.kcal

    assert rates.eyring(barrier1) == pytest.approx(1.38, 4e-3)
    assert rates.eyring(barrier2) == pytest.approx(0.0928, 3e-3)

    # many temperatures at once
    assert rates.eyring(barrier1, temperature=[14.01, 298.15, 1074.0]) == pytest.approx(
        [0.0, 1.38, 6.8808e9], 4e-3
    )
    assert rates.eyring(barrier2, temperature=[14.01, 298.15, 1074.0]) == pytest.approx(
        [0.0, 0.0928, 3.2513e9], 3e-3
    )

    # many activation barriers at once
    assert rates.eyring([barrier1, barrier2]) == pytest.approx([1.38, 0.0928], 4e-3)

    # many pairs of activation barriers and temperatures
    assert rates.eyring(
        [barrier1, barrier2], temperature=[298.15, 1074.0]
    ) == pytest.approx([1.38, 3.2513e9], 4e-3)


def test_smoluchowski_calculates_diffusion_limited_reaction_rates():
    """Ensure Smoluchowski rates are correct."""
    radii = np.array([2.59, 2.71]) * constants.angstrom
    assert rates.smoluchowski(
        radii,
        reactive_radius=2.6 * constants.angstrom,
        viscosity=[8.90e-4, 8.54e-4, 6.94e-4, 5.77e-4, 4.90e-4, 4.22e-4, 3.69e-4],
        temperature=[298.15, 300.00, 310.00, 320.00, 330.00, 340.00, 350.00],
    ) / constants.liter == pytest.approx(
        [3.6e9, 3.8e9, 4.9e9, 6.0e9, 7.3e9, 8.8e9, 1.0e10], 4e-2
    )

    assert rates.smoluchowski(
        radii,
        reactive_radius=2.6 * constants.angstrom,
        viscosity="water",
        temperature=[298.15, 300.00, 310.00, 320.00, 330.00, 340.00, 350.00],
    ) / constants.liter == pytest.approx(
        [3.6e9, 3.8e9, 4.9e9, 6.0e9, 7.3e9, 8.8e9, 1.0e10], 4e-2
    )


def test_liquid_viscosities_are_correct():
    """Ensure water viscosity values agree with doi:10.1002/9781118131473."""
    temperature = 273.15 + np.array(
        [0, 5, 10, 15, 20, 25, 30, 40, 50, 60, 70, 80, 90, 100]
    )
    viscosity = 1e-3 * np.array(
        [
            1.781,
            1.518,
            1.307,
            1.139,
            1.002,
            0.890,
            0.798,
            0.653,
            0.547,
            0.466,
            0.404,
            0.354,
            0.315,
            0.282,
        ]
    )
    assert rates.liquid_viscosity("water", temperature) == pytest.approx(
        viscosity, 6e-2
    )


def test_conversion_of_rate_constants_work():
    """Ensure converting reaction rate constants work."""
    assert rates.convert_rate_constant(
        12345e5,
        "cm3 mol-1 s-1",
        "l mol-1 s-1",
        molecularity=2,
        temperature=[200, 298.15, 300, 400],
    ) == pytest.approx(12345e8)

    assert rates.convert_rate_constant(
        12345e10,
        "cm3 particle-1 s-1",
        "atm-1 s-1",
        molecularity=2,
        temperature=[200, 298.15, 300, 400],
    ) == pytest.approx(13.63e-23 * 12345e10 * np.array([200, 298.15, 300, 400]), 1e-3)


def test_conversion_rates_know_about_reaction_order():
    """Ensure calculated factors satisfy the reaction order requirements."""
    assert rates.convert_rate_constant(
        1.0, "cm3 particle-1 s-1", "atm-1 s-1"
    ) == pytest.approx(1.0)
    assert (
        rates.convert_rate_constant(
            1.0, "cm3 particle-1 s-1", "atm-1 s-1", molecularity=3
        )
        == rates.convert_rate_constant(
            1.0, "cm3 particle-1 s-1", "atm-1 s-1", molecularity=2
        )
        ** 2
    )
    assert (
        rates.convert_rate_constant(
            1.0, "cm3 particle-1 s-1", "atm-1 s-1", molecularity=4
        )
        == rates.convert_rate_constant(
            1.0, "cm3 particle-1 s-1", "atm-1 s-1", molecularity=2
        )
        ** 3
    )


def test_conversion_rates_match_literature():
    """Ensure calculated factors are the same as in doi:10.1021/ed046p54."""
    # to cm3 mol-1 s-1
    assert rates.convert_rate_constant(
        1.0, "cm3 mol-1 s-1", "cm3 mol-1 s-1", molecularity=2
    ) == pytest.approx(1.0)
    assert rates.convert_rate_constant(
        1.0, "cm3 mol-1 s-1", "l mol-1 s-1", molecularity=2
    ) == pytest.approx(1e3)
    assert rates.convert_rate_constant(
        1.0, "cm3 mol-1 s-1", "m3 mol-1 s-1", molecularity=2
    ) == pytest.approx(1e6)
    assert rates.convert_rate_constant(
        1.0, "cm3 mol-1 s-1", "cm3 particle-1 s-1", molecularity=2
    ) == pytest.approx(6.023e23, 1e-3)
    assert rates.convert_rate_constant(
        1.0, "cm3 mol-1 s-1", "mmHg-1 s-1", molecularity=2
    ) == pytest.approx(62.40e3 * 298.15, 1e-3)
    assert rates.convert_rate_constant(
        1.0, "cm3 mol-1 s-1", "atm-1 s-1", molecularity=2
    ) == pytest.approx(82.10 * 298.15, 1e-3)

    # to l mol-1 s-1
    assert rates.convert_rate_constant(
        1.0, "l mol-1 s-1", "cm3 mol-1 s-1", molecularity=2
    ) == pytest.approx(1e-3)
    assert rates.convert_rate_constant(
        1.0, "l mol-1 s-1", "l mol-1 s-1", molecularity=2
    ) == pytest.approx(1.0)
    assert rates.convert_rate_constant(
        1.0, "l mol-1 s-1", "m3 mol-1 s-1", molecularity=2
    ) == pytest.approx(1e3)
    assert rates.convert_rate_constant(
        1.0, "l mol-1 s-1", "cm3 particle-1 s-1", molecularity=2
    ) == pytest.approx(6.023e20, 1e-3)
    assert rates.convert_rate_constant(
        1.0, "l mol-1 s-1", "mmHg-1 s-1", molecularity=2
    ) == pytest.approx(62.40 * 298.15, 1e-3)
    assert rates.convert_rate_constant(
        1.0, "l mol-1 s-1", "atm-1 s-1", molecularity=2
    ) == pytest.approx(82.10e-3 * 298.15, 1e-3)

    # to m3 mol-1 s-1
    assert rates.convert_rate_constant(
        1.0, "m3 mol-1 s-1", "cm3 mol-1 s-1", molecularity=2
    ) == pytest.approx(1e-6)
    assert rates.convert_rate_constant(
        1.0, "m3 mol-1 s-1", "l mol-1 s-1", molecularity=2
    ) == pytest.approx(1e-3)
    assert rates.convert_rate_constant(
        1.0, "m3 mol-1 s-1", "m3 mol-1 s-1", molecularity=2
    ) == pytest.approx(1.0)
    assert rates.convert_rate_constant(
        1.0, "m3 mol-1 s-1", "cm3 particle-1 s-1", molecularity=2
    ) == pytest.approx(6.023e17, 1e-3)
    assert rates.convert_rate_constant(
        1.0, "m3 mol-1 s-1", "mmHg-1 s-1", molecularity=2
    ) == pytest.approx(62.40e-3 * 298.15, 1e-3)
    assert rates.convert_rate_constant(
        1.0, "m3 mol-1 s-1", "atm-1 s-1", molecularity=2
    ) == pytest.approx(82.10e-6 * 298.15, 1e-3)

    # to cm3 particle-1 s-1
    assert rates.convert_rate_constant(
        1.0, "cm3 particle-1 s-1", "cm3 mol-1 s-1", molecularity=2
    ) == pytest.approx(0.1660e-23)
    assert rates.convert_rate_constant(
        1.0, "cm3 particle-1 s-1", "l mol-1 s-1", molecularity=2
    ) == pytest.approx(0.1660e-20)
    assert rates.convert_rate_constant(
        1.0, "cm3 particle-1 s-1", "m3 mol-1 s-1", molecularity=2
    ) == pytest.approx(0.1660e-17)
    assert rates.convert_rate_constant(
        1.0, "cm3 particle-1 s-1", "cm3 particle-1 s-1", molecularity=2
    ) == pytest.approx(1.0)
    assert rates.convert_rate_constant(
        1.0, "cm3 particle-1 s-1", "mmHg-1 s-1", molecularity=2
    ) == pytest.approx(10.23e-20 * 298.15)
    assert rates.convert_rate_constant(
        1.0, "cm3 particle-1 s-1", "atm-1 s-1", molecularity=2
    ) == pytest.approx(13.63e-23 * 298.15)

    # to mmHg-1 s-1
    assert rates.convert_rate_constant(
        1.0, "mmHg-1 s-1", "cm3 mol-1 s-1", molecularity=2
    ) == pytest.approx(16.03e-6 / 298.15, 1e-3)
    assert rates.convert_rate_constant(
        1.0, "mmHg-1 s-1", "l mol-1 s-1", molecularity=2
    ) == pytest.approx(16.03e-3 / 298.15, 1e-3)
    assert rates.convert_rate_constant(
        1.0, "mmHg-1 s-1", "m3 mol-1 s-1", molecularity=2
    ) == pytest.approx(16.03 / 298.15, 1e-3)
    assert rates.convert_rate_constant(
        1.0, "mmHg-1 s-1", "cm3 particle-1 s-1", molecularity=2
    ) == pytest.approx(96.53e17 / 298.15, 1e-3)
    assert rates.convert_rate_constant(
        1.0, "mmHg-1 s-1", "mmHg-1 s-1", molecularity=2
    ) == pytest.approx(1.0)
    assert rates.convert_rate_constant(
        1.0, "mmHg-1 s-1", "atm-1 s-1", molecularity=2
    ) == pytest.approx(1.316e-3, 1e-3)

    # to atm-1 s-1
    assert rates.convert_rate_constant(
        1.0, "atm-1 s-1", "cm3 mol-1 s-1", molecularity=2
    ) == pytest.approx(12.18e-3 / 298.15, 1e-3)
    assert rates.convert_rate_constant(
        1.0, "atm-1 s-1", "l mol-1 s-1", molecularity=2
    ) == pytest.approx(12.18 / 298.15, 1e-3)
    assert rates.convert_rate_constant(
        1.0, "atm-1 s-1", "m3 mol-1 s-1", molecularity=2
    ) == pytest.approx(12.18e3 / 298.15, 1e-3)
    assert rates.convert_rate_constant(
        1.0, "atm-1 s-1", "cm3 particle-1 s-1", molecularity=2
    ) == pytest.approx(73.36e20 / 298.15, 1e-3)
    assert rates.convert_rate_constant(
        1.0, "atm-1 s-1", "mmHg-1 s-1", molecularity=2
    ) == pytest.approx(760)
    assert rates.convert_rate_constant(
        1.0, "atm-1 s-1", "atm-1 s-1", molecularity=2
    ) == pytest.approx(1.0)
