#!/usr/bin/env python3

"""Tests for constants module."""

import pytest

import overreact as rx
from overreact import _constants as constants


def test_reference_raw_constants():
    """Ensure raw constants are close to values commonly used by the community.

    Reference values were taken from the ones used by Gaussian 16
    (http://gaussian.com/constants/).
    """
    assert constants.bohr / constants.angstrom == pytest.approx(0.52917721092)
    assert constants.atomic_mass == pytest.approx(1.660538921e-27)
    assert constants.h == pytest.approx(6.62606957e-34)
    assert constants.N_A == pytest.approx(6.02214129e23)
    assert constants.kcal == pytest.approx(4184.0)
    assert constants.hartree == pytest.approx(4.35974434e-18)
    assert constants.c / constants.centi == pytest.approx(2.99792458e10)
    assert constants.k == pytest.approx(1.3806488e-23)
    assert rx.thermo.molar_volume(
        temperature=273.15, pressure=constants.bar
    ) == pytest.approx(0.022710953)


def test_reference_conversion_factors():
    """Ensure conversion factors are close to values commonly used by the community.

    Reference values were taken from the ones used by Gaussian 16
    (http://gaussian.com/constants/).
    """
    assert constants.eV == pytest.approx(1.602176565e-19)
    assert constants.eV * constants.N_A / constants.kcal == pytest.approx(23.06, 3e-5)
    assert constants.hartree * constants.N_A / constants.kcal == pytest.approx(627.5095)
    assert constants.hartree / constants.eV == pytest.approx(27.2114)
    assert constants.hartree * constants.centi / (
        constants.h * constants.c
    ) == pytest.approx(219474.63)
