#!/usr/bin/env python3

"""Tests for the application programming interface (API)."""

import numpy as np
import pytest

import overreact as rx
from overreact import _constants as constants
from overreact import coords


def test_get_enthalpies():  # noqa: D103
    model = rx.parse_model("data/hickel1992/UM06-2X/6-311++G(d,p)/model.k")
    assert rx.get_delta(
        model.scheme.B, rx.get_enthalpies(model.compounds, qrrho=False),
    )[0] / (constants.hartree * constants.N_A) == pytest.approx(
        -132.23510843 - (-56.51424787 + -75.72409969), 2e-2,
    )


def test_get_entropies():
    """Ensure get_entropies match some logfiles.

    It is worth mentioning that, currently, ORCA uses QRRHO in entropy
    calculations, but not for enthalpies.
    """
    model = rx.parse_model("data/ethane/B97-3c/model.k")
    assert 298.15 * rx.get_delta(
        model.scheme.B, rx.get_entropies(model.compounds, environment="gas"),
    )[0] / (constants.hartree * constants.N_A) == pytest.approx(
        0.02685478 - 0.00942732 + 0.00773558 - (0.02753672 - 0.00941496 + 0.00772322),
        4e-4,
    )

    model = rx.parse_model("data/tanaka1996/UMP2/cc-pVTZ/model.k")
    assert 298.15 * rx.get_delta(
        model.scheme.B, rx.get_entropies(model.compounds, environment="gas"),
    )[0] / (constants.hartree * constants.N_A) == pytest.approx(
        0.03025523 - 0.01030794 + 0.00927065 - (0.02110620 + 0.00065446 + 0.01740262),
        3e-5,
    )

    model = rx.parse_model("data/hickel1992/UM06-2X/6-311++G(d,p)/model.k")
    sym_correction = 298.15 * rx.change_reference_state(3, 1)
    assert 298.15 * rx.get_delta(
        model.scheme.B, rx.get_entropies(model.compounds, environment="gas"),
    )[0] / (constants.hartree * constants.N_A) == pytest.approx(
        0.03070499
        - ((0.02288418 - 0.00647387 + 0.00543658) + 0.02022750)
        - sym_correction / (constants.hartree * constants.N_A),
        2e-4,
    )


def test_get_freeenergies():  # noqa: D103
    model = rx.parse_model("data/hickel1992/UM06-2X/6-311++G(d,p)/model.k")
    sym_correction = 298.15 * rx.change_reference_state(3, 1)

    # TODO(schneiderfelipe): should qrrho=(False, True) be default?
    assert rx.get_delta(
        model.scheme.B,
        rx.get_freeenergies(model.compounds, environment="gas", qrrho=(False, True)),
    )[0] / (constants.hartree * constants.N_A) == pytest.approx(
        -132.26581342
        - ((-56.53713205 + 0.00647387 - 0.00543658) + -75.74432719)
        + sym_correction / (constants.hartree * constants.N_A),
        3e-3,
    )


def test_compare_calc_star_with_get_star():
    """Ensure the calc_* functions match the get_* functions."""
    model = rx.parse_model("data/hickel1992/UM06-2X/6-311++G(d,p)/model.k")
    sym_correction = 298.15 * rx.change_reference_state(3, 1)

    freeenergies_ref = [
        -56.53713205
        + 0.00647387
        - 0.00543658,  # NH3(w) + correct rot. entropy (ORCA didn't get C3v)
        -75.74432719,
        -132.26581342 + sym_correction / (constants.hartree * constants.N_A),
        -55.87195207
        + 0.00580418
        - 0.00514973,  # NH2·(w) + correct rot. entropy (ORCA didn't get C2v)
        -76.43172074
        + 0.00562807
        - 0.00497361,  # H2O(w) + correct rot. entropy (ORCA didn't get C2v)
        -56.98250956
        + 0.00696968
        - 0.00462348,  # NH4+(w) + correct rot. entropy (ORCA didn't get C2v)
    ]  # ORCA logfiles, Eh

    for bias in np.array([-1, 0, 1]) * constants.kcal:
        for environment in ["gas", "solvent"]:
            for qrrho in [True, False, (False, True)]:
                qrrho_enthalpy, qrrho_entropy = rx.api._check_qrrho(qrrho)
                for temperature in [200, 298.15, 400]:
                    for pressure in [
                        constants.bar / 2,
                        constants.atm,
                        2 * constants.bar,
                    ]:
                        freeenergies_get = rx.get_freeenergies(
                            model.compounds,
                            bias=bias,
                            environment=environment,
                            qrrho=qrrho,
                            temperature=temperature,
                            pressure=pressure,
                        )
                        for i, (compound, data) in enumerate(model.compounds.items()):
                            moments, _, _ = coords.inertia(
                                data.atommasses, data.atomcoords,
                            )
                            symmetry_number = coords.symmetry_number(
                                coords.find_point_group(
                                    data.atommasses, data.atomcoords,
                                ),
                            )

                            enthalpy_calc = rx.thermo.calc_enthalpy(
                                energy=data.energy,
                                degeneracy=data.mult,
                                moments=moments,
                                vibfreqs=data.vibfreqs,
                                qrrho=qrrho_enthalpy,
                                temperature=temperature,
                            )
                            entropy_calc = rx.thermo.calc_entropy(
                                atommasses=data.atommasses,
                                energy=data.energy,
                                degeneracy=data.mult,
                                moments=moments,
                                symmetry_number=symmetry_number,
                                vibfreqs=data.vibfreqs,
                                environment=environment,
                                qrrho=qrrho_entropy,
                                temperature=temperature,
                                pressure=pressure,
                            )
                            if data.symmetry is not None:
                                entropy_calc -= rx.change_reference_state(
                                    data.symmetry,
                                    1,
                                    temperature=temperature,
                                    pressure=pressure,
                                )

                            freeenergy_calc = (
                                enthalpy_calc - temperature * entropy_calc + bias
                            )

                            assert freeenergies_get[i] == pytest.approx(freeenergy_calc)

                            if (
                                bias == 0
                                and environment == "gas"
                                and qrrho == (False, True)
                                and temperature == 298.15
                                and pressure == constants.atm
                                # TODO(schneiderfelipe): do a test for H+(w)
                                and compound != "H+(w)"
                            ):
                                assert freeenergies_get[i] / (
                                    constants.hartree * constants.N_A
                                ) == pytest.approx(
                                    freeenergies_ref[i], 7e-7,
                                )  # ORCA logfile
