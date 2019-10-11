#!/usr/bin/env python3

"""Tests for misc module."""

import numpy as np
import pytest
from scipy.constants import atm
from scipy.constants import bar

from overreact import misc


def test_molar_volume_is_precise():
    """Ensure our molar volumes are as precise as possible.

    Values below were taken from
    <https://en.wikipedia.org/wiki/Molar_volume#Ideal_gases>, which were
    calculated to the same precision using the ideal gas constant from 2014
    CODATA.
    """
    assert misc.molar_volume(273.15, bar) == pytest.approx(0.02271098038, 1e-5)
    assert misc.molar_volume(pressure=bar) == pytest.approx(0.02478959842, 1e-5)
    assert misc.molar_volume(273.15) == pytest.approx(0.022414, 1e-5)
    assert misc.molar_volume() == pytest.approx(0.024465, 1e-4)


def test_molar_volume_works_with_sequences():
    """Ensure molar volumes can be calculated for many temperatures at once."""
    assert misc.molar_volume([273.15, 298.15], bar) == pytest.approx(
        [0.02271098038, 0.02478959842], 1e-5
    )
    assert misc.molar_volume([273.15, 298.15], [atm, bar]) == pytest.approx(
        [0.022414, 0.02478959842], 1e-5
    )
    assert misc.molar_volume(273.15, [atm, bar]) == pytest.approx(
        [0.022414, 0.02271098038], 1e-5
    )


def test_broaden_spectrum_works():
    """Ensure we can broad a simple spectrum."""
    x = np.linspace(50, 200, num=15)
    s = misc.broaden_spectrum(x, [150, 100], [2, 1], scale=20.0)
    assert s == pytest.approx(
        [
            0.04316864,
            0.1427911,
            0.35495966,
            0.66562891,
            0.95481719,
            1.09957335,
            1.16005667,
            1.34925384,
            1.72176729,
            2.0,
            1.85988866,
            1.32193171,
            0.70860709,
            0.28544362,
            0.0863263,
        ]
    )
