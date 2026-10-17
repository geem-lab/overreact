"""Tests for the application programming interface (API)."""

from __future__ import annotations

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

import overreact as rx
from overreact import _constants as constants
from overreact import coords


def test_get_enthalpies() -> None:
    """Ensure we can retrieve enthalpies."""
    model = rx.parse_model("data/hickel1992/UM06-2X/6-311++G(d,p)/model.k")
    assert rx.get_delta(
        model.scheme.B,
        rx.get_enthalpies(model.compounds, qrrho=False),
    )[0] / (constants.hartree * constants.N_A) == pytest.approx(
        -132.23510843 - (-56.51424787 + -75.72409969),
        2e-2,
    )


def test_get_entropies() -> None:
    """Ensure get_entropies match some logfiles.

    It is worth mentioning that, currently, ORCA uses QRRHO in entropy
    calculations, but not for enthalpies.
    """
    model = rx.parse_model("data/ethane/B97-3c/model.k")
    assert 298.15 * rx.get_delta(
        model.scheme.B,
        rx.get_entropies(model.compounds, environment="gas"),
    )[0] / (constants.hartree * constants.N_A) == pytest.approx(
        0.02685478 - 0.00942732 + 0.00773558 - (0.02753672 - 0.00941496 + 0.00772322),
        4e-4,
    )

    model = rx.parse_model("data/tanaka1996/UMP2/cc-pVTZ/model.k")
    assert 298.15 * rx.get_delta(
        model.scheme.B,
        rx.get_entropies(model.compounds, environment="gas"),
    )[0] / (constants.hartree * constants.N_A) == pytest.approx(
        0.03025523 - 0.01030794 + 0.00927065 - (0.02110620 + 0.00065446 + 0.01740262),
        3e-5,
    )

    model = rx.parse_model("data/hickel1992/UM06-2X/6-311++G(d,p)/model.k")
    sym_correction = 298.15 * rx.change_reference_state(3, 1)
    assert 298.15 * rx.get_delta(
        model.scheme.B,
        rx.get_entropies(model.compounds, environment="gas"),
    )[0] / (constants.hartree * constants.N_A) == pytest.approx(
        0.03070499
        - ((0.02288418 - 0.00647387 + 0.00543658) + 0.02022750)
        - sym_correction / (constants.hartree * constants.N_A),
        2e-4,
    )


def test_get_freeenergies() -> None:
    """Ensure we can retrieve free energies."""
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


@settings(deadline=None)
@given(
    st.floats(min_value=-1.0, max_value=1.0),
    st.sampled_from(["gas", "solvent"]),
    st.sampled_from([True, False, (False, True)]),
    st.floats(min_value=200.0, max_value=400.0),
    st.floats(min_value=constants.bar / 2, max_value=2 * constants.bar),
)
def test_compare_calc_star_with_get_star(
    bias, environment, qrrho, temperature, pressure
) -> None:
    """Ensure the calc_* functions match the get_* functions."""
    model = rx.parse_model("data/hickel1992/UM06-2X/6-311++G(d,p)/model.k")
    bias *= constants.kcal
    qrrho_enthalpy, qrrho_entropy = rx.api._check_qrrho(qrrho)
    freeenergies_get = rx.get_freeenergies(
        model.compounds,
        bias=bias,
        environment=environment,
        qrrho=qrrho,
        temperature=temperature,
        pressure=pressure,
    )
    for i, data in enumerate(model.compounds.values()):
        moments, _, _ = coords.inertia(data.atommasses, data.atomcoords)
        symmetry_number = coords.symmetry_number(
            coords.find_point_group(data.atommasses, data.atomcoords),
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

        freeenergy_calc = enthalpy_calc - temperature * entropy_calc + bias
        assert freeenergies_get[i] == pytest.approx(freeenergy_calc)


def test_get_freeenergies_match_orca_logfiles() -> None:
    """Ensure free energies match ORCA logfiles."""
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

    freeenergies = rx.get_freeenergies(
        model.compounds,
        environment="gas",
        qrrho=(False, True),
        temperature=298.15,
        pressure=constants.atm,
    )
    # TODO(schneiderfelipe): do a test for H+(w)
    assert freeenergies[:-1] / (constants.hartree * constants.N_A) == pytest.approx(
        freeenergies_ref,
        7e-7,
    )
