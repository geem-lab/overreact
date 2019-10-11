#!/usr/bin/env python3

"""Tests for thermo module."""

import numpy as np
import pytest
from scipy.constants import atm
from scipy.constants import bar
from scipy.constants import calorie
from scipy.constants import kilo
from scipy.constants import liter

from overreact.misc import molar_volume
from overreact import thermo


def test_get_delta_freeenergy_works():
    """Ensure safe usage of get_delta_freeenergy."""
    assert thermo.get_delta_freeenergy([-1, 2], [-5.0, 5.0]) == pytest.approx(15.0)
    assert thermo.get_delta_freeenergy(
        [[-1, -1], [2, 1]], [-5.0, 5.0]
    ) == pytest.approx([15.0, 10.0])


def test_equilibrium_constant_works():
    """Ensure equilibrium_constant gives correct numbers."""
    assert thermo.equilibrium_constant(0.0) == 1.0
    assert thermo.equilibrium_constant(1000.0) == pytest.approx(0.668, 1e-4)
    assert thermo.equilibrium_constant(1718.5) == pytest.approx(0.5, 1e-4)
    assert thermo.equilibrium_constant(68497.0) == pytest.approx(0.0)

    assert thermo.equilibrium_constant(0.0, temperature=14.01) == 1.0
    assert thermo.equilibrium_constant(1000.0, temperature=14.01) == pytest.approx(
        1.87e-4, 1e-3
    )
    assert thermo.equilibrium_constant(80.8, temperature=14.01) == pytest.approx(
        0.5, 1e-3
    )
    assert thermo.equilibrium_constant(68497.0, temperature=14.01) == pytest.approx(0.0)

    # reproduction from doi:10.1021/ic202081z:
    delta_freeenergy = (
        kilo
        * (
            np.array([25.2, 15.4, -19.4, -24.3, 40.7])
            + np.array([27.0, 11.6, -16.5, -32.9, 38.6])
        )
        / 2
    )
    assert thermo.equilibrium_constant(
        delta_freeenergy, temperature=298.0
    ) == pytest.approx([2.65e-5, 4.30e-3, 1.39e3, 1.04e5, 1.14e-7], 1e-1)


def test_change_reference_state_works_for_gas_to_liquid_standard_states():
    """Ensure change_reference_state works for gas to liquid state change."""
    # different reference temperatures
    assert thermo.change_reference_state(temperature=200.0) / (
        kilo * calorie
    ) == pytest.approx(1.10, 1e-1)
    assert thermo.change_reference_state() / (kilo * calorie) == pytest.approx(
        1.85, 1e-1
    )
    assert thermo.change_reference_state(temperature=300.0) / (
        kilo * calorie
    ) == pytest.approx(1.90, 1e-2)
    assert thermo.change_reference_state(temperature=400.0) / (
        kilo * calorie
    ) == pytest.approx(2.70, 1e-1)

    # different reference pressures
    temperature = 298.15
    assert thermo.change_reference_state(
        1.0 / liter, 1.0 / molar_volume(temperature, atm), temperature=temperature
    ) / (kilo * calorie) == pytest.approx(1.89, 1e-2)
    assert thermo.change_reference_state(
        1.0 / liter, 1.0 / molar_volume(temperature, bar), temperature=temperature
    ) / (kilo * calorie) == pytest.approx(1.90, 1e-2)

    # volumes instead of concentrations
    temperature = 298.15
    assert thermo.change_reference_state(
        liter, molar_volume(temperature, atm), sign=-1, temperature=temperature
    ) / (kilo * calorie) == pytest.approx(1.89, 1e-2)


def test_change_reference_state_works_for_symmetry():
    """Ensure that change_reference_state works for symmetry contributions."""
    assert thermo.change_reference_state(4, 1, sign=-1, temperature=200.0) / (
        kilo * calorie
    ) == pytest.approx(-0.60, 1e-1)
    assert thermo.change_reference_state(4, 1, sign=-1) / (
        kilo * calorie
    ) == pytest.approx(-0.85, 1e-1)
    assert thermo.change_reference_state(4, 1, sign=-1, temperature=300.0) / (
        kilo * calorie
    ) == pytest.approx(-0.80, 1e-1)
    assert thermo.change_reference_state(4, 1, sign=-1, temperature=400.0) / (
        kilo * calorie
    ) == pytest.approx(-1.10, 1e-2)

    # calculating many symmetry corrections at once
    assert thermo.change_reference_state(
        6, 12, sign=1, temperature=[0, 200, 298.15, 300, 400]
    ) / (kilo * calorie) == pytest.approx([0.0, -0.3, -0.4, -0.4, -0.6], 1e-1)
