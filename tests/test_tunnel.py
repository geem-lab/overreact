#!/usr/bin/env python3

"""Tests for tunnel module."""

import pytest

import overreact as rx


def test_wigner_tunneling_corrections_are_correct():
    """Ensure Wigner tunneling values are correct."""
    with pytest.raises(ValueError):
        rx.tunnel.wigner(0.0)

    assert rx.tunnel.wigner(0.1) == pytest.approx(1.0, 1e-8)

    # values below are Wigner corrections at the lower limit for which the
    # Eckart correction is computable
    assert rx.tunnel.wigner(59) == pytest.approx(1.0033776142915123)
    assert rx.tunnel.wigner(158) == pytest.approx(1.0242225691391296)

    assert rx.tunnel.wigner(1218, temperature=[200, 298.15, 300, 400]) == pytest.approx(
        [4.2, 2.4, 2.4, 1.8], 1.7e-2
    )


def test_eckart_tunneling_corrections_are_correct():
    """Ensure Eckart tunneling values are correct."""
    with pytest.raises(ValueError):
        rx.tunnel.eckart(0.0, 15781.6)

    with pytest.raises(ValueError):
        rx.tunnel.eckart(0.0, 56813.61, 94689.35)

    # values below are at the lower limit for which Eckart is computable
    assert rx.tunnel.eckart(59, 15781.6) == pytest.approx(1.01679385)
    assert rx.tunnel.eckart(158, 56813.61, 94689.35) == pytest.approx(1.02392807)

    # a selection for testing higher precision from DOI:10.1021/j100809a040
    assert rx.tunnel.eckart(414.45, 15781.6, 15781.6) == pytest.approx(1.2, 9e-3)
    assert rx.tunnel.eckart(414.45, 1578.16, 1578.16) == pytest.approx(1.32, 2e-3)
    assert rx.tunnel.eckart(1243.35, 9468.94, 28406.81) == pytest.approx(3.39, 2e-4)
    assert rx.tunnel.eckart(3315.6, 12625.25, 126252.50) == pytest.approx(23.3, 3e-4)
    assert rx.tunnel.eckart(3315.6, 12625.25, 12625.25) == pytest.approx(34.0, 2e-4)
    assert rx.tunnel.eckart(2486.7, 56813.61, 94689.35) == pytest.approx(3920, 1e-4)


def test_eckart_is_symmetric():
    """Ensure Eckart has symmetry under interchange of H1 and H2."""
    vibfreq = 3e3
    H1 = 1e4
    H2 = 10 * H1
    assert rx.tunnel.eckart(vibfreq, H1, H2) == rx.tunnel.eckart(vibfreq, H2, H1)


def test_low_level_eckart_against_johnston1962():
    """Reproduce all values from Table 1 of DOI:10.1021/j100809a040."""
    us = [2, 3, 4, 5, 6, 8, 10, 12, 16]  # columns

    # The first examples have very low barriers and comprise the "imaginary
    # region" (4 * alpha1 * alpha2 - pi**2 < 0). The precision of our
    # implementation is a bit reduced in this region, in comparison to the
    # number of decimal places available to compare against. We are
    # nevertheless able to reproduce all decimal places for the majority of the
    # examples.

    gammas = [1.16, 1.25, 1.34, 1.44, 1.55, 1.80, 2.09, 2.42, 3.26]
    for u, gamma in zip(us, gammas):
        assert rx.tunnel._eckart(u, 0.5) == pytest.approx(gamma, 2e-2)

    gammas = [1.13, 1.21, 1.29, 1.38, 1.47, 1.68, 1.93, 2.22, 2.94]
    for u, gamma in zip(us, gammas):
        assert rx.tunnel._eckart(u, 0.5, 1) == pytest.approx(gamma, 2e-2)

    gammas = [1.09, 1.14, 1.20, 1.27, 1.34, 1.51, 1.71, 1.94, 2.53]
    for u, gamma in zip(us, gammas):
        assert rx.tunnel._eckart(u, 0.5, 2) == pytest.approx(gamma, 9e-3)

    gammas = [1.04, 1.07, 1.11, 1.16, 1.22, 1.35, 1.50, 1.69, 2.16]
    for u, gamma in zip(us, gammas):
        assert rx.tunnel._eckart(u, 0.5, 4) == pytest.approx(gamma, 1e-2)

    gammas = [0.99, 1.00, 1.03, 1.06, 1.11, 1.21, 1.34, 1.49, 1.88]
    for u, gamma in zip(us, gammas):
        assert rx.tunnel._eckart(u, 0.5, 8) == pytest.approx(gamma, 9e-3)

    gammas = [0.96, 0.97, 0.99, 1.02, 1.06, 1.15, 1.26, 1.40, 1.76]
    for u, gamma in zip(us, gammas):
        assert rx.tunnel._eckart(u, 0.5, 12) == pytest.approx(gamma, 8e-3)

    gammas = [0.94, 0.95, 0.97, 0.99, 1.02, 1.11, 1.22, 1.35, 1.68]
    for u, gamma in zip(us, gammas):
        assert rx.tunnel._eckart(u, 0.5, 16) == pytest.approx(gamma, 8e-3)

    gammas = [0.93, 0.94, 0.95, 0.97, 1.00, 1.08, 1.19, 1.31, 1.64]
    for u, gamma in zip(us, gammas):
        assert rx.tunnel._eckart(u, 0.5, 20) == pytest.approx(gamma, 1e-2)

    gammas = [1.27, 1.43, 1.62, 1.83, 2.09, 2.72, 3.56, 4.68, 8.19]
    for u, gamma in zip(us, gammas):
        assert rx.tunnel._eckart(u, 1) == pytest.approx(gamma, 2e-2)

    gammas = [1.21, 1.35, 1.51, 1.71, 1.93, 2.50, 3.26, 4.28, 7.48]
    for u, gamma in zip(us, gammas):
        assert rx.tunnel._eckart(u, 1, 2) == pytest.approx(gamma, 6e-3)

    gammas = [1.14, 1.24, 1.37, 1.53, 1.71, 2.16, 2.78, 3.60, 6.16]
    for u, gamma in zip(us, gammas):
        assert rx.tunnel._eckart(u, 1, 4) == pytest.approx(gamma, 5e-3)

    gammas = [1.08, 1.16, 1.26, 1.39, 1.54, 1.92, 2.43, 3.12, 5.25]
    for u, gamma in zip(us, gammas):
        assert rx.tunnel._eckart(u, 1, 8) == pytest.approx(gamma, 5e-3)

    gammas = [1.06, 1.12, 1.21, 1.33, 1.46, 1.81, 2.28, 2.91, 4.88]
    for u, gamma in zip(us, gammas):
        assert rx.tunnel._eckart(u, 1, 12) == pytest.approx(gamma, 9e-3)

    gammas = [1.04, 1.10, 1.18, 1.29, 1.42, 1.75, 2.20, 2.80, 4.66]
    for u, gamma in zip(us, gammas):
        assert rx.tunnel._eckart(u, 1, 16) == pytest.approx(gamma, 5e-3)

    gammas = [1.03, 1.08, 1.16, 1.26, 1.39, 1.70, 2.14, 2.72, 4.52]
    for u, gamma in zip(us, gammas):
        assert rx.tunnel._eckart(u, 1, 20) == pytest.approx(gamma, 7e-3)

    # The next examples are not in the imaginary region anymore and our
    # precision is improved. I believe that here are the most important cases
    # for chemistry. We get precisely all decimal places for all examples in
    # this section.

    gammas = [1.32, 1.58, 1.91, 2.34, 2.90, 4.55, 7.34, 12.1, 34.0]
    for u, gamma in zip(us, gammas):
        assert rx.tunnel._eckart(u, 2) == pytest.approx(gamma, 4e-3)

    gammas = [1.26, 1.47, 1.77, 2.16, 2.66, 4.20, 6.85, 11.4, 33.4]
    for u, gamma in zip(us, gammas):
        assert rx.tunnel._eckart(u, 2, 4) == pytest.approx(gamma, 5e-3)

    gammas = [1.19, 1.36, 1.61, 1.93, 2.36, 3.65, 5.87, 9.69, 28.0]
    for u, gamma in zip(us, gammas):
        assert rx.tunnel._eckart(u, 2, 8) == pytest.approx(gamma, 4e-3)

    gammas = [1.16, 1.32, 1.54, 1.84, 2.23, 3.41, 5.44, 8.94, 25.6]
    for u, gamma in zip(us, gammas):
        assert rx.tunnel._eckart(u, 2, 12) == pytest.approx(gamma, 5e-3)

    gammas = [1.14, 1.29, 1.50, 1.78, 2.15, 3.27, 5.20, 8.51, 24.2]
    for u, gamma in zip(us, gammas):
        assert rx.tunnel._eckart(u, 2, 16) == pytest.approx(gamma, 4e-3)

    gammas = [1.12, 1.27, 1.47, 1.74, 2.10, 3.18, 5.03, 8.22, 23.3]
    for u, gamma in zip(us, gammas):
        assert rx.tunnel._eckart(u, 2, 20) == pytest.approx(gamma, 4e-3)

    gammas = [1.30, 1.58, 2.02, 2.69, 3.69, 7.60, 17.3, 42.4, 304]
    for u, gamma in zip(us, gammas):
        assert rx.tunnel._eckart(u, 4, 4) == pytest.approx(gamma, 8e-3)

    gammas = [1.25, 1.51, 1.93, 2.56, 3.56, 7.57, 18.0, 46.7, 376]
    for u, gamma in zip(us, gammas):
        assert rx.tunnel._eckart(u, 4, 8) == pytest.approx(gamma, 5e-3)

    gammas = [1.22, 1.47, 1.86, 2.46, 3.39, 7.16, 17.0, 44.0, 354]
    for u, gamma in zip(us, gammas):
        assert rx.tunnel._eckart(u, 4, 12) == pytest.approx(gamma, 3e-3)

    gammas = [1.20, 1.44, 1.81, 2.39, 3.28, 6.88, 16.2, 41.9, 335]
    for u, gamma in zip(us, gammas):
        assert rx.tunnel._eckart(u, 4, 16) == pytest.approx(gamma, 3e-3)

    gammas = [1.19, 1.42, 1.78, 2.34, 3.20, 6.68, 15.7, 40.3, 321]
    for u, gamma in zip(us, gammas):
        assert rx.tunnel._eckart(u, 4, 20) == pytest.approx(gamma, 2e-3)

    # The examples below have larger tunneling corrections and the precision of
    # our implementation is again reduced, but we still reproduce all decimal
    # places for most of the examples here.

    gammas = [1.24, 1.56, 2.04, 2.94, 4.54, 13.8, 57.0, 307]
    for u, gamma in zip(us[:-1], gammas):
        assert rx.tunnel._eckart(u, 8, 8) == pytest.approx(gamma, 2e-2)

    gammas = [1.22, 1.54, 2.04, 2.96, 4.68, 15.4, 71.7, 445]
    for u, gamma in zip(us[:-1], gammas):
        assert rx.tunnel._eckart(u, 8, 12) == pytest.approx(gamma, 2e-2)

    gammas = [1.21, 1.53, 2.02, 2.93, 4.65, 15.6, 74.4, 473]
    for u, gamma in zip(us[:-1], gammas):
        assert rx.tunnel._eckart(u, 8, 16) == pytest.approx(gamma, 2e-2)

    gammas = [1.20, 1.51, 2.00, 2.90, 4.61, 15.5, 74.2, 474]
    for u, gamma in zip(us[:-1], gammas):
        assert rx.tunnel._eckart(u, 8, 20) == pytest.approx(gamma, 2e-2)

    gammas = [1.2, 1.5, 2.1, 3.1, 5.2, 22, 162, 1970]
    for u, gamma in zip(us[:-1], gammas):
        assert rx.tunnel._eckart(u, 12, 12) == pytest.approx(gamma, 2e-2)

    gammas = [1.2, 1.5, 2.1, 3.1, 5.4, 25, 220, 3300]
    for u, gamma in zip(us[:-1], gammas):
        assert rx.tunnel._eckart(u, 12, 16) == pytest.approx(gamma, 2e-2)

    gammas = [1.2, 1.5, 2.1, 3.1, 5.4, 26, 246, 3920]
    for u, gamma in zip(us[:-1], gammas):
        assert rx.tunnel._eckart(u, 12, 20) == pytest.approx(gamma, 2e-2)

    gammas = [1.2, 1.5, 2.1, 3.2, 5.7, 32, 437]
    for u, gamma in zip(us[:-2], gammas):
        assert rx.tunnel._eckart(u, 16, 16) == pytest.approx(gamma, 2e-2)

    gammas = [1.2, 1.5, 2.1, 3.2, 5.9, 37, 616]
    for u, gamma in zip(us[:-2], gammas):
        assert rx.tunnel._eckart(u, 16, 20) == pytest.approx(gamma, 2e-2)

    gammas = [1.2, 1.5, 2.1, 3.2, 6.1, 46, 1150]
    for u, gamma in zip(us[:-2], gammas):
        assert rx.tunnel._eckart(u, 20, 20) == pytest.approx(gamma, 4e-2)
