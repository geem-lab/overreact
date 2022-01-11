#!/usr/bin/env python3

"""Tests for rates module."""

import numpy as np
import pytest

import overreact as rx
from overreact import _constants as constants


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

    scheme = rx.parse_reactions("FH + Cl -> FHCl‡ -> F + HCl")
    delta_freeenergies_H = rx.thermo.get_delta(scheme.B, freeenergies_H)[0]
    delta_freeenergies_D = rx.thermo.get_delta(scheme.B, freeenergies_D)[0]
    assert delta_freeenergies_H == pytest.approx(0.027509)
    assert delta_freeenergies_H * (constants.hartree * constants.N_A) / (
        constants.kcal
    ) == pytest.approx(17.26, 2e-4)
    assert delta_freeenergies_D == pytest.approx(0.030054)
    assert delta_freeenergies_D * (constants.hartree * constants.N_A) / (
        constants.kcal
    ) == pytest.approx(18.86, 5e-5)

    k_H = rx.get_k(
        scheme,
        delta_freeenergies=delta_freeenergies_H * constants.hartree * constants.N_A,
    )
    k_D = rx.get_k(
        scheme,
        delta_freeenergies=delta_freeenergies_D * constants.hartree * constants.N_A,
    )
    assert k_H == pytest.approx(1.38, 4e-4)
    assert k_D == pytest.approx(0.0928, 5e-3)

    kie = k_H / k_D
    assert kie == pytest.approx(1.38 / 0.0928, 4e-3)


def test_eyring_calculates_reaction_barrier():
    """Ensure Eyring rates are correct."""
    barrier1 = 17.26 * constants.kcal
    barrier2 = 18.86 * constants.kcal

    assert rx.rates.eyring(barrier1) == pytest.approx(1.38, 4e-3)
    assert rx.rates.eyring(barrier2) == pytest.approx(0.0928, 3e-3)

    # many temperatures at once
    assert rx.rates.eyring(
        barrier1, temperature=[14.01, 298.15, 1074.0]
    ) == pytest.approx([0.0, 1.38, 6.8808e9], 4e-3)
    assert rx.rates.eyring(
        barrier2, temperature=[14.01, 298.15, 1074.0]
    ) == pytest.approx([0.0, 0.0928, 3.2513e9], 3e-3)

    # many activation barriers at once
    assert rx.rates.eyring([barrier1, barrier2]) == pytest.approx([1.38, 0.0928], 4e-3)

    # many pairs of activation barriers and temperatures
    assert rx.rates.eyring(
        [barrier1, barrier2], temperature=[298.15, 1074.0]
    ) == pytest.approx([1.38, 3.2513e9], 4e-3)


def test_smoluchowski_calculates_diffusion_limited_reaction_rates():
    """Ensure Smoluchowski rates are correct."""
    radii = np.array([2.59, 2.71]) * constants.angstrom
    assert rx.rates.smoluchowski(
        radii,
        reactive_radius=2.6 * constants.angstrom,
        viscosity=[8.90e-4, 8.54e-4, 6.94e-4, 5.77e-4, 4.90e-4, 4.22e-4, 3.69e-4],
        temperature=[298.15, 300.00, 310.00, 320.00, 330.00, 340.00, 350.00],
    ) / constants.liter == pytest.approx(
        [3.6e9, 3.8e9, 4.9e9, 6.0e9, 7.3e9, 8.8e9, 1.0e10], 4e-2
    )

    assert rx.rates.smoluchowski(
        radii,
        reactive_radius=2.6 * constants.angstrom,
        viscosity="water",
        temperature=[298.15, 300.00, 310.00, 320.00, 330.00, 340.00, 350.00],
    ) / constants.liter == pytest.approx(
        [3.6e9, 3.8e9, 4.9e9, 6.0e9, 7.3e9, 8.8e9, 1.0e10], 4e-2
    )


def test_liquid_viscosities_are_correct():
    """Ensure viscosity values agree with the literature.

    The following data were collected from many different sources, whose DOIs
    or ISBNs are to be found in the comments.
    """
    # - pentane (isbn:978-1138561632):
    # TODO(schneiderfelipe): pentane: 144 K -- 308 K
    temperature = 273.15 + np.array([25])
    viscosity = 1e-3 * np.array([2.24e-1])
    assert rx.rates.liquid_viscosity("pentane", temperature) == pytest.approx(
        viscosity, 4e-3
    )

    # - hexane (DOI:10.1063/1.555943):
    # TODO(schneiderfelipe): hexane: 178 K -- 340 K
    temperature = 273.15 + np.array([25])
    viscosity = 1e-3 * np.array([0.295])
    assert rx.rates.liquid_viscosity("hexane", temperature) == pytest.approx(
        viscosity, 5e-3
    )

    # - acetone (DOI:10.1021/je00017a031):
    temperature = 273.15 + np.array([25])
    viscosity = 1e-3 * np.array([0.302])
    assert rx.rates.liquid_viscosity("acetone", temperature) == pytest.approx(
        viscosity, 2e-2
    )

    # - heptane (DOI:10.1063/1.555943):
    # TODO(schneiderfelipe): heptane: 183 K -- 370 K
    temperature = 273.15 + np.array([25])
    viscosity = 1e-3 * np.array([0.389])
    assert rx.rates.liquid_viscosity("heptane", temperature) == pytest.approx(
        viscosity, 9e-5
    )

    # - octane (DOI:10.1063/1.555943):
    # TODO(schneiderfelipe): octane: 217 K -- 398 K
    temperature = 273.15 + np.array([25])
    viscosity = 1e-3 * np.array([0.509])
    assert rx.rates.liquid_viscosity("octane", temperature) == pytest.approx(
        viscosity, 2e-3
    )

    # - benzene (isbn:978-1138561632):
    temperature = 273.15 + np.array([25])
    viscosity = 1e-3 * np.array([0.604])
    assert rx.rates.liquid_viscosity("benzene", temperature) == pytest.approx(
        viscosity, 7e-3
    )

    # - water (isbn:978-1138561632):
    temperature = 273.15 + np.array([10, 20, 25, 30, 50, 70, 90])
    viscosity = 1e-3 * np.array(
        [1.3059, 1.0016, 8.90e-1, 0.79722, 0.54652, 0.40355, 0.31417]
    )
    assert rx.rates.liquid_viscosity("water", temperature) == pytest.approx(
        viscosity, 3e-2
    )

    # - water (DOI:10.1002/9781118131473):
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
    assert rx.rates.liquid_viscosity("water", temperature) == pytest.approx(
        viscosity, 6e-2
    )

    # - water (DOI:10.1002/qua.25686):
    temperature = np.array([298.15, 300, 310, 320, 330, 340, 350])
    viscosity = 1e-4 * np.array([8.90, 8.54, 6.94, 5.77, 4.90, 4.22, 3.69])
    assert rx.rates.liquid_viscosity("water", temperature) == pytest.approx(
        viscosity, 3e-2
    )

    # - ethanol (isbn:978-1138561632):
    temperature = 273.15 + np.array([25])
    viscosity = 1e-3 * np.array([1.074])
    assert rx.rates.liquid_viscosity("ethanol", temperature) == pytest.approx(
        viscosity, 4e-3
    )

    # - 2-propanol (DOI:10.1021/je00058a025):
    temperature = 273.15 + np.array([25])
    viscosity = 1e-3 * np.array([2.052])
    assert rx.rates.liquid_viscosity("2-propanol", temperature) == pytest.approx(
        viscosity, 1e-3
    )


def test_conversion_of_rate_constants_work():
    """Ensure converting reaction rate constants work."""
    assert rx.rates.convert_rate_constant(
        12345e5,
        "cm3 mol-1 s-1",
        "l mol-1 s-1",
        molecularity=2,
        temperature=[200.0, 273.15, 298.15, 300.0, 373.15, 400.0],
    ) == pytest.approx(12345e8)

    assert rx.rates.convert_rate_constant(
        12345e10,
        "cm3 particle-1 s-1",
        "atm-1 s-1",
        molecularity=2,
        temperature=[200.0, 273.15, 298.15, 300.0, 373.15, 400.0],
    ) == pytest.approx(
        13.63e-23 * 12345e10 * np.array([200.0, 273.15, 298.15, 300.0, 373.15, 400.0]),
        3e-4,
    )


def test_conversion_rates_know_about_reaction_order():
    """Ensure calculated factors satisfy the reaction order requirements."""
    assert rx.rates.convert_rate_constant(
        1.0, "cm3 particle-1 s-1", "atm-1 s-1"
    ) == pytest.approx(1.0)
    assert (
        rx.rates.convert_rate_constant(
            1.0, "cm3 particle-1 s-1", "atm-1 s-1", molecularity=3
        )
        == rx.rates.convert_rate_constant(
            1.0, "cm3 particle-1 s-1", "atm-1 s-1", molecularity=2
        )
        ** 2
    )
    assert (
        rx.rates.convert_rate_constant(
            1.0, "cm3 particle-1 s-1", "atm-1 s-1", molecularity=4
        )
        == rx.rates.convert_rate_constant(
            1.0, "cm3 particle-1 s-1", "atm-1 s-1", molecularity=2
        )
        ** 3
    )


def test_second_order_conversion_rate_example():
    """Test a simple example from a set of lecture notes.

    The example can be found at
    <https://cefrc.princeton.edu/sites/cefrc/files/Files/2015%20Lecture%20Notes/Wang/Lecture-4-Bimolecular-Reaction-Rate-Coefficients.pdf>.
    """
    assert rx.rates.convert_rate_constant(
        2e-14, "cm3 mol-1 s-1", "cm3 particle-1 s-1", molecularity=2, temperature=300.0
    ) == pytest.approx(1.2e10, 4e-3)


def test_second_order_conversion_rates_match_literature():
    """Ensure calculated second order factors are correct.

    References are given in the comments."""
    for temperature in [200.0, 273.15, 298.15, 300.0, 373.15, 400.0]:
        # to cm3 mol-1 s-1 (DOI:10.1021/ed046p54)
        assert rx.rates.convert_rate_constant(
            1.0,
            "cm3 mol-1 s-1",
            "cm3 mol-1 s-1",
            molecularity=2,
            temperature=temperature,
        ) == pytest.approx(1.0)
        assert rx.rates.convert_rate_constant(
            1.0, "cm3 mol-1 s-1", "l mol-1 s-1", molecularity=2, temperature=temperature
        ) == pytest.approx(1e3)
        assert rx.rates.convert_rate_constant(
            1.0,
            "cm3 mol-1 s-1",
            "m3 mol-1 s-1",
            molecularity=2,
            temperature=temperature,
        ) == pytest.approx(1e6)
        assert rx.rates.convert_rate_constant(
            1.0,
            "cm3 mol-1 s-1",
            "cm3 particle-1 s-1",
            molecularity=2,
            temperature=temperature,
        ) == pytest.approx(6.023e23, 2e-4)
        assert rx.rates.convert_rate_constant(
            1.0, "cm3 mol-1 s-1", "mmHg-1 s-1", molecularity=2, temperature=temperature
        ) == pytest.approx(62.40e3 * temperature, 6e-4)
        assert rx.rates.convert_rate_constant(
            1.0, "cm3 mol-1 s-1", "atm-1 s-1", molecularity=2, temperature=temperature
        ) == pytest.approx(82.10 * temperature, 6e-4)
        # Next values are from
        # <https://nvlpubs.nist.gov/nistpubs/Legacy/NSRDS/nbsnsrds67.pdf>.
        assert rx.rates.convert_rate_constant(
            1.0, "cm3 mol-1 s-1", "mmHg-1 s-1", molecularity=2, temperature=temperature
        ) == pytest.approx(6.236e4 * temperature, 6e-5)
        assert rx.rates.convert_rate_constant(
            1.0, "cm3 mol-1 s-1", "atm-1 s-1", molecularity=2, temperature=temperature
        ) == pytest.approx(82.06 * temperature, 4e-5)

        # to l mol-1 s-1 (DOI:10.1021/ed046p54)
        assert rx.rates.convert_rate_constant(
            1.0, "l mol-1 s-1", "cm3 mol-1 s-1", molecularity=2, temperature=temperature
        ) == pytest.approx(1e-3)
        assert rx.rates.convert_rate_constant(
            1.0, "l mol-1 s-1", "l mol-1 s-1", molecularity=2, temperature=temperature
        ) == pytest.approx(1.0)
        assert rx.rates.convert_rate_constant(
            1.0, "l mol-1 s-1", "m3 mol-1 s-1", molecularity=2, temperature=temperature
        ) == pytest.approx(1e3)
        assert rx.rates.convert_rate_constant(
            1.0,
            "l mol-1 s-1",
            "cm3 particle-1 s-1",
            molecularity=2,
            temperature=temperature,
        ) == pytest.approx(6.023e20, 2e-4)
        assert rx.rates.convert_rate_constant(
            1.0, "l mol-1 s-1", "mmHg-1 s-1", molecularity=2, temperature=temperature
        ) == pytest.approx(62.40 * temperature, 6e-4)
        assert rx.rates.convert_rate_constant(
            1.0, "l mol-1 s-1", "atm-1 s-1", molecularity=2, temperature=temperature
        ) == pytest.approx(82.10e-3 * temperature, 6e-4)
        # Next values are from
        # <https://nvlpubs.nist.gov/nistpubs/Legacy/NSRDS/nbsnsrds67.pdf>.
        assert rx.rates.convert_rate_constant(
            1.0, "l mol-1 s-1", "mmHg-1 s-1", molecularity=2, temperature=temperature
        ) == pytest.approx(62.36 * temperature, 6e-5)
        assert rx.rates.convert_rate_constant(
            1.0, "l mol-1 s-1", "atm-1 s-1", molecularity=2, temperature=temperature
        ) == pytest.approx(8.206e-2 * temperature, 4e-5)

        # to m3 mol-1 s-1 (DOI:10.1021/ed046p54)
        assert rx.rates.convert_rate_constant(
            1.0,
            "m3 mol-1 s-1",
            "cm3 mol-1 s-1",
            molecularity=2,
            temperature=temperature,
        ) == pytest.approx(1e-6)
        assert rx.rates.convert_rate_constant(
            1.0, "m3 mol-1 s-1", "l mol-1 s-1", molecularity=2, temperature=temperature
        ) == pytest.approx(1e-3)
        assert rx.rates.convert_rate_constant(
            1.0, "m3 mol-1 s-1", "m3 mol-1 s-1", molecularity=2, temperature=temperature
        ) == pytest.approx(1.0)
        assert rx.rates.convert_rate_constant(
            1.0,
            "m3 mol-1 s-1",
            "cm3 particle-1 s-1",
            molecularity=2,
            temperature=temperature,
        ) == pytest.approx(6.023e17, 2e-4)
        assert rx.rates.convert_rate_constant(
            1.0, "m3 mol-1 s-1", "mmHg-1 s-1", molecularity=2, temperature=temperature
        ) == pytest.approx(62.40e-3 * temperature, 6e-4)
        assert rx.rates.convert_rate_constant(
            1.0, "m3 mol-1 s-1", "atm-1 s-1", molecularity=2, temperature=temperature
        ) == pytest.approx(82.10e-6 * temperature, 6e-4)
        # Next values are from
        # <https://nvlpubs.nist.gov/nistpubs/Legacy/NSRDS/nbsnsrds67.pdf>.
        assert rx.rates.convert_rate_constant(
            1.0, "m3 mol-1 s-1", "mmHg-1 s-1", molecularity=2, temperature=temperature
        ) == pytest.approx(6.236e-2 * temperature, 6e-5)
        assert rx.rates.convert_rate_constant(
            1.0, "m3 mol-1 s-1", "atm-1 s-1", molecularity=2, temperature=temperature
        ) == pytest.approx(8.206e-5 * temperature, 4e-5)

        # to cm3 particle-1 s-1 (DOI:10.1021/ed046p54)
        assert rx.rates.convert_rate_constant(
            1.0,
            "cm3 particle-1 s-1",
            "cm3 mol-1 s-1",
            molecularity=2,
            temperature=temperature,
        ) == pytest.approx(0.1660e-23)
        assert rx.rates.convert_rate_constant(
            1.0,
            "cm3 particle-1 s-1",
            "l mol-1 s-1",
            molecularity=2,
            temperature=temperature,
        ) == pytest.approx(0.1660e-20)
        assert rx.rates.convert_rate_constant(
            1.0,
            "cm3 particle-1 s-1",
            "m3 mol-1 s-1",
            molecularity=2,
            temperature=temperature,
        ) == pytest.approx(0.1660e-17)
        assert rx.rates.convert_rate_constant(
            1.0,
            "cm3 particle-1 s-1",
            "cm3 particle-1 s-1",
            molecularity=2,
            temperature=temperature,
        ) == pytest.approx(1.0)
        assert rx.rates.convert_rate_constant(
            1.0,
            "cm3 particle-1 s-1",
            "mmHg-1 s-1",
            molecularity=2,
            temperature=temperature,
        ) == pytest.approx(10.23e-20 * temperature)
        assert rx.rates.convert_rate_constant(
            1.0,
            "cm3 particle-1 s-1",
            "atm-1 s-1",
            molecularity=2,
            temperature=temperature,
        ) == pytest.approx(13.63e-23 * temperature)
        # Next values are from
        # <https://nvlpubs.nist.gov/nistpubs/Legacy/NSRDS/nbsnsrds67.pdf>.
        assert rx.rates.convert_rate_constant(
            1.0,
            "cm3 particle-1 s-1",
            "mmHg-1 s-1",
            molecularity=2,
            temperature=temperature,
        ) == pytest.approx(1.035e-19 * temperature)
        assert rx.rates.convert_rate_constant(
            1.0,
            "cm3 particle-1 s-1",
            "atm-1 s-1",
            molecularity=2,
            temperature=temperature,
        ) == pytest.approx(1.362e-22 * temperature)

        # to mmHg-1 s-1 (DOI:10.1021/ed046p54)
        assert rx.rates.convert_rate_constant(
            1.0, "mmHg-1 s-1", "cm3 mol-1 s-1", molecularity=2, temperature=temperature
        ) == pytest.approx(16.03e-6 / temperature, 4e-4)
        assert rx.rates.convert_rate_constant(
            1.0, "mmHg-1 s-1", "l mol-1 s-1", molecularity=2, temperature=temperature
        ) == pytest.approx(16.03e-3 / temperature, 4e-4)
        assert rx.rates.convert_rate_constant(
            1.0, "mmHg-1 s-1", "m3 mol-1 s-1", molecularity=2, temperature=temperature
        ) == pytest.approx(16.03 / temperature, 4e-4)
        assert rx.rates.convert_rate_constant(
            1.0,
            "mmHg-1 s-1",
            "cm3 particle-1 s-1",
            molecularity=2,
            temperature=temperature,
        ) == pytest.approx(96.53e17 / temperature, 4e-4)
        assert rx.rates.convert_rate_constant(
            1.0, "mmHg-1 s-1", "mmHg-1 s-1", molecularity=2, temperature=temperature
        ) == pytest.approx(1.0)
        assert rx.rates.convert_rate_constant(
            1.0, "mmHg-1 s-1", "atm-1 s-1", molecularity=2, temperature=temperature
        ) == pytest.approx(1.316e-3, 2e-4)
        # Next values are from
        # <https://nvlpubs.nist.gov/nistpubs/Legacy/NSRDS/nbsnsrds67.pdf>.
        assert rx.rates.convert_rate_constant(
            1.0, "mmHg-1 s-1", "cm3 mol-1 s-1", molecularity=2, temperature=temperature
        ) == pytest.approx(1.604e-5 / temperature, 4e-4)
        assert rx.rates.convert_rate_constant(
            1.0, "mmHg-1 s-1", "l mol-1 s-1", molecularity=2, temperature=temperature
        ) == pytest.approx(1.604e-2 / temperature, 4e-4)
        assert rx.rates.convert_rate_constant(
            1.0, "mmHg-1 s-1", "m3 mol-1 s-1", molecularity=2, temperature=temperature
        ) == pytest.approx(16.04 / temperature, 4e-4)
        # the following has a wrong exponent in the reference
        assert rx.rates.convert_rate_constant(
            1.0,
            "mmHg-1 s-1",
            "cm3 particle-1 s-1",
            molecularity=2,
            temperature=temperature,
        ) == pytest.approx(9.658e18 / temperature, 2e-4)

        # to atm-1 s-1 (DOI:10.1021/ed046p54)
        assert rx.rates.convert_rate_constant(
            1.0, "atm-1 s-1", "cm3 mol-1 s-1", molecularity=2, temperature=temperature
        ) == pytest.approx(12.18e-3 / temperature, 6e-4)
        assert rx.rates.convert_rate_constant(
            1.0, "atm-1 s-1", "l mol-1 s-1", molecularity=2, temperature=temperature
        ) == pytest.approx(12.18 / temperature, 6e-4)
        assert rx.rates.convert_rate_constant(
            1.0, "atm-1 s-1", "m3 mol-1 s-1", molecularity=2, temperature=temperature
        ) == pytest.approx(12.18e3 / temperature, 6e-4)
        assert rx.rates.convert_rate_constant(
            1.0,
            "atm-1 s-1",
            "cm3 particle-1 s-1",
            molecularity=2,
            temperature=temperature,
        ) == pytest.approx(73.36e20 / temperature, 5e-4)
        assert rx.rates.convert_rate_constant(
            1.0, "atm-1 s-1", "mmHg-1 s-1", molecularity=2, temperature=temperature
        ) == pytest.approx(760)
        assert rx.rates.convert_rate_constant(
            1.0, "atm-1 s-1", "atm-1 s-1", molecularity=2, temperature=temperature
        ) == pytest.approx(1.0)
        # Next values are from
        # <https://nvlpubs.nist.gov/nistpubs/Legacy/NSRDS/nbsnsrds67.pdf>.
        assert rx.rates.convert_rate_constant(
            1.0, "atm-1 s-1", "cm3 mol-1 s-1", molecularity=2, temperature=temperature
        ) == pytest.approx(1.219e-2 / temperature, 3e-4)
        assert rx.rates.convert_rate_constant(
            1.0, "atm-1 s-1", "l mol-1 s-1", molecularity=2, temperature=temperature
        ) == pytest.approx(12.19 / temperature, 3e-4)
        assert rx.rates.convert_rate_constant(
            1.0, "atm-1 s-1", "m3 mol-1 s-1", molecularity=2, temperature=temperature
        ) == pytest.approx(1.219e4 / temperature, 3e-4)
        assert rx.rates.convert_rate_constant(
            1.0,
            "atm-1 s-1",
            "cm3 particle-1 s-1",
            molecularity=2,
            temperature=temperature,
        ) == pytest.approx(7.34e21 / temperature, 2e-4)


def test_third_order_conversion_rates_match_literature():
    """Ensure calculated third order factors are correct.

    References are given in the comments."""
    for temperature in [200.0, 273.15, 298.15, 300.0, 373.15, 400.0]:
        # to cm3 mol-1 s-1
        # (<https://nvlpubs.nist.gov/nistpubs/Legacy/NSRDS/nbsnsrds67.pdf>)
        assert rx.rates.convert_rate_constant(
            1.0,
            "cm3 mol-1 s-1",
            "cm3 mol-1 s-1",
            molecularity=3,
            temperature=temperature,
        ) == pytest.approx(1.0)
        # the following has a wrong exponent in the reference
        assert rx.rates.convert_rate_constant(
            1.0, "cm3 mol-1 s-1", "l mol-1 s-1", molecularity=3, temperature=temperature
        ) == pytest.approx(1e6)
        assert rx.rates.convert_rate_constant(
            1.0,
            "cm3 mol-1 s-1",
            "m3 mol-1 s-1",
            molecularity=3,
            temperature=temperature,
        ) == pytest.approx(1e12)
        assert rx.rates.convert_rate_constant(
            1.0,
            "cm3 mol-1 s-1",
            "cm3 particle-1 s-1",
            molecularity=3,
            temperature=temperature,
        ) == pytest.approx(3.628e47, 4e-4)
        assert rx.rates.convert_rate_constant(
            1.0, "cm3 mol-1 s-1", "mmHg-1 s-1", molecularity=3, temperature=temperature
        ) == pytest.approx(3.89e9 * temperature ** 2, 3e-4)
        assert rx.rates.convert_rate_constant(
            1.0, "cm3 mol-1 s-1", "atm-1 s-1", molecularity=3, temperature=temperature
        ) == pytest.approx(6.733e3 * temperature ** 2, 7e-5)

        # to l mol-1 s-1
        # (<https://nvlpubs.nist.gov/nistpubs/Legacy/NSRDS/nbsnsrds67.pdf>)
        assert rx.rates.convert_rate_constant(
            1.0, "l mol-1 s-1", "cm3 mol-1 s-1", molecularity=3, temperature=temperature
        ) == pytest.approx(1e-6)
        assert rx.rates.convert_rate_constant(
            1.0, "l mol-1 s-1", "l mol-1 s-1", molecularity=3, temperature=temperature
        ) == pytest.approx(1.0)
        assert rx.rates.convert_rate_constant(
            1.0, "l mol-1 s-1", "m3 mol-1 s-1", molecularity=3, temperature=temperature
        ) == pytest.approx(1e6)
        assert rx.rates.convert_rate_constant(
            1.0,
            "l mol-1 s-1",
            "cm3 particle-1 s-1",
            molecularity=3,
            temperature=temperature,
        ) == pytest.approx(3.628e41, 9e-4)
        assert rx.rates.convert_rate_constant(
            1.0, "l mol-1 s-1", "mmHg-1 s-1", molecularity=3, temperature=temperature
        ) == pytest.approx(3.89e3 * temperature ** 2, 3e-4)
        assert rx.rates.convert_rate_constant(
            1.0, "l mol-1 s-1", "atm-1 s-1", molecularity=3, temperature=temperature
        ) == pytest.approx(6.733e-3 * temperature ** 2, 7e-5)

        # to m3 mol-1 s-1
        # (<https://nvlpubs.nist.gov/nistpubs/Legacy/NSRDS/nbsnsrds67.pdf>)
        assert rx.rates.convert_rate_constant(
            1.0,
            "m3 mol-1 s-1",
            "cm3 mol-1 s-1",
            molecularity=3,
            temperature=temperature,
        ) == pytest.approx(1e-12)
        assert rx.rates.convert_rate_constant(
            1.0, "m3 mol-1 s-1", "l mol-1 s-1", molecularity=3, temperature=temperature
        ) == pytest.approx(1e-6)
        assert rx.rates.convert_rate_constant(
            1.0, "m3 mol-1 s-1", "m3 mol-1 s-1", molecularity=3, temperature=temperature
        ) == pytest.approx(1.0)
        assert rx.rates.convert_rate_constant(
            1.0,
            "m3 mol-1 s-1",
            "cm3 particle-1 s-1",
            molecularity=3,
            temperature=temperature,
        ) == pytest.approx(3.628e35, 4e-4)
        assert rx.rates.convert_rate_constant(
            1.0, "m3 mol-1 s-1", "mmHg-1 s-1", molecularity=3, temperature=temperature
        ) == pytest.approx(3.89e-3 * temperature ** 2, 3e-4)
        assert rx.rates.convert_rate_constant(
            1.0, "m3 mol-1 s-1", "atm-1 s-1", molecularity=3, temperature=temperature
        ) == pytest.approx(6.733e-9 * temperature ** 2, 7e-5)

        # to cm3 particle-1 s-1
        # (<https://nvlpubs.nist.gov/nistpubs/Legacy/NSRDS/nbsnsrds67.pdf>)
        assert rx.rates.convert_rate_constant(
            1.0,
            "cm3 particle-1 s-1",
            "cm3 mol-1 s-1",
            molecularity=3,
            temperature=temperature,
        ) == pytest.approx(2.76e-48)
        assert rx.rates.convert_rate_constant(
            1.0,
            "cm3 particle-1 s-1",
            "l mol-1 s-1",
            molecularity=3,
            temperature=temperature,
        ) == pytest.approx(2.76e-42)
        assert rx.rates.convert_rate_constant(
            1.0,
            "cm3 particle-1 s-1",
            "m3 mol-1 s-1",
            molecularity=3,
            temperature=temperature,
        ) == pytest.approx(2.76e-36)
        assert rx.rates.convert_rate_constant(
            1.0,
            "cm3 particle-1 s-1",
            "cm3 particle-1 s-1",
            molecularity=3,
            temperature=temperature,
        ) == pytest.approx(1.0)
        assert rx.rates.convert_rate_constant(
            1.0,
            "cm3 particle-1 s-1",
            "mmHg-1 s-1",
            molecularity=3,
            temperature=temperature,
        ) == pytest.approx(1.07e-36 * temperature ** 2)
        assert rx.rates.convert_rate_constant(
            1.0,
            "cm3 particle-1 s-1",
            "atm-1 s-1",
            molecularity=3,
            temperature=temperature,
        ) == pytest.approx(1.86e-44 * temperature ** 2)

        # to mmHg-1 s-1
        # (<https://nvlpubs.nist.gov/nistpubs/Legacy/NSRDS/nbsnsrds67.pdf>)
        assert rx.rates.convert_rate_constant(
            1.0, "mmHg-1 s-1", "cm3 mol-1 s-1", molecularity=3, temperature=temperature
        ) == pytest.approx(2.57e-10 / temperature ** 2)
        assert rx.rates.convert_rate_constant(
            1.0, "mmHg-1 s-1", "l mol-1 s-1", molecularity=3, temperature=temperature
        ) == pytest.approx(2.57e-4 / temperature ** 2, 5e-4)
        assert rx.rates.convert_rate_constant(
            1.0, "mmHg-1 s-1", "m3 mol-1 s-1", molecularity=3, temperature=temperature
        ) == pytest.approx(257 / temperature ** 2, 5e-4)
        assert rx.rates.convert_rate_constant(
            1.0,
            "mmHg-1 s-1",
            "cm3 particle-1 s-1",
            molecularity=3,
            temperature=temperature,
        ) == pytest.approx(9.328e37 / temperature ** 2, 4e-4)
        assert rx.rates.convert_rate_constant(
            1.0, "mmHg-1 s-1", "mmHg-1 s-1", molecularity=3, temperature=temperature
        ) == pytest.approx(1.0)
        assert rx.rates.convert_rate_constant(
            1.0, "mmHg-1 s-1", "atm-1 s-1", molecularity=3, temperature=temperature
        ) == pytest.approx(1.73e-6, 8e-4)

        # to atm-1 s-1 (<https://nvlpubs.nist.gov/nistpubs/Legacy/NSRDS/nbsnsrds67.pdf>)
        assert rx.rates.convert_rate_constant(
            1.0, "atm-1 s-1", "cm3 mol-1 s-1", molecularity=3, temperature=temperature
        ) == pytest.approx(1.48e-4 / temperature ** 2, 4e-3)
        # the following has a wrong exponent in the reference
        assert rx.rates.convert_rate_constant(
            1.0, "atm-1 s-1", "l mol-1 s-1", molecularity=3, temperature=temperature
        ) == pytest.approx(1.48e2 / temperature ** 2, 4e-3)
        assert rx.rates.convert_rate_constant(
            1.0, "atm-1 s-1", "m3 mol-1 s-1", molecularity=3, temperature=temperature
        ) == pytest.approx(1.48e8 / temperature ** 2, 4e-3)
        assert rx.rates.convert_rate_constant(
            1.0,
            "atm-1 s-1",
            "cm3 particle-1 s-1",
            molecularity=3,
            temperature=temperature,
        ) == pytest.approx(5.388e43 / temperature ** 2, 4e-4)
        assert rx.rates.convert_rate_constant(
            1.0, "atm-1 s-1", "mmHg-1 s-1", molecularity=3, temperature=temperature
        ) == pytest.approx(5.776e5)
        assert rx.rates.convert_rate_constant(
            1.0, "atm-1 s-1", "atm-1 s-1", molecularity=3, temperature=temperature
        ) == pytest.approx(1.0)
