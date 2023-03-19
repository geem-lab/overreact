#!/usr/bin/env python3

"""Tests for misc module."""

import numpy as np
import pytest

import overreact as rx


def test_broaden_spectrum_works():
    """Ensure we can broad a simple spectrum."""
    x = np.linspace(50, 200, num=15)
    s = rx._misc.broaden_spectrum(x, [150, 100], [2, 1], scale=20.0)
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
        ],
    )
