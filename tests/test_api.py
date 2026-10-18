"""Tests for the application programming interface (API)."""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

import overreact as rx
from overreact import _constants as constants


@given(st.floats(-50.0, 50.0))
def test_get_k_satisfies_detailed_balance(delta_freeenergy) -> None:
    """Ensure equilibria rate constants have the ratio exp(-dG/RT)."""
    delta_freeenergy *= constants.kcal
    k = rx.get_k(
        rx.Scheme.parse("A <=> B"),
        delta_freeenergies=[delta_freeenergy, -delta_freeenergy],
    )
    assert k[0] == pytest.approx(
        k[1] * np.exp(-delta_freeenergy / (constants.R * 298.15)),
    )


@given(
    st.floats(-50.0, 50.0),
    st.floats(-1e5, 1e5).filter(lambda mismatch: abs(mismatch) >= 1e-5),
)
def test_get_k_rejects_inconsistent_equilibria(forward, mismatch) -> None:
    """Ensure the reverse half of an equilibrium must undo the forward half."""
    forward *= constants.kcal
    with pytest.raises(ValueError, match="reverse"):
        rx.get_k(
            rx.Scheme.parse("A <=> B"),
            delta_freeenergies=[forward, mismatch - forward],
        )


def test_get_enthalpies() -> None:
    """Ensure we can retrieve enthalpies."""
    model = rx.parse_model("data/hickel1992/UM06-2X/6-311++G(d,p)/model.k")
    assert model.scheme.activation_deltas(
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
    assert 298.15 * model.scheme.activation_deltas(
        rx.get_entropies(model.compounds, environment="gas"),
    )[0] / (constants.hartree * constants.N_A) == pytest.approx(
        0.02685478 - 0.00942732 + 0.00773558 - (0.02753672 - 0.00941496 + 0.00772322),
        4e-4,
    )

    model = rx.parse_model("data/tanaka1996/UMP2/cc-pVTZ/model.k")
    assert 298.15 * model.scheme.activation_deltas(
        rx.get_entropies(model.compounds, environment="gas"),
    )[0] / (constants.hartree * constants.N_A) == pytest.approx(
        0.03025523 - 0.01030794 + 0.00927065 - (0.02110620 + 0.00065446 + 0.01740262),
        3e-5,
    )

    model = rx.parse_model("data/hickel1992/UM06-2X/6-311++G(d,p)/model.k")
    sym_correction = 298.15 * rx.change_reference_state(3, 1)
    assert 298.15 * model.scheme.activation_deltas(
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
    assert model.scheme.activation_deltas(
        rx.get_freeenergies(model.compounds, environment="gas", qrrho=(False, True)),
    )[0] / (constants.hartree * constants.N_A) == pytest.approx(
        -132.26581342
        - ((-56.53713205 + 0.00647387 - 0.00543658) + -75.74432719)
        + sym_correction / (constants.hartree * constants.N_A),
        3e-3,
    )


@pytest.fixture(scope="module")
def hickel1992():
    """Model of NH3(w) + OH·(w) -> NH2·(w) + H2O(w)."""
    return rx.parse_model("data/hickel1992/UM06-2X/6-311++G(d,p)/model.k")


@settings(max_examples=25)
@given(
    st.sampled_from([True, False, (False, True)]),
    st.floats(min_value=200.0, max_value=400.0),
    st.floats(min_value=constants.bar / 2, max_value=2 * constants.bar),
    st.floats(min_value=constants.bar / 2, max_value=2 * constants.bar),
)
def test_gas_freeenergies_follow_ideal_gas_pressure(
    hickel1992, qrrho, temperature, pressure, other_pressure
) -> None:
    """Ensure gas free energies change by RT ln(p2/p1) with pressure."""

    def freeenergies(pressure):
        return rx.get_freeenergies(
            hickel1992.compounds,
            environment="gas",
            qrrho=qrrho,
            temperature=temperature,
            pressure=pressure,
        )

    assert freeenergies(other_pressure) - freeenergies(pressure) == pytest.approx(
        constants.R * temperature * np.log(other_pressure / pressure),
        abs=1e-7,
    )


@pytest.mark.parametrize("tunneling", ["eckart", None])
def test_get_k_matches_compounds_by_name(hickel1992, tunneling) -> None:
    """Ensure rate constants don't depend on the order of compounds."""
    reversed_compounds = dict(reversed(hickel1992.compounds.items()))
    assert rx.get_k(
        hickel1992.scheme, reversed_compounds, tunneling=tunneling
    ) == pytest.approx(
        rx.get_k(hickel1992.scheme, hickel1992.compounds, tunneling=tunneling)
    )


def test_get_freeenergies_match_orca_logfiles(hickel1992) -> None:
    """Ensure free energies match ORCA logfiles."""
    model = hickel1992
    sym_correction = 298.15 * rx.change_reference_state(3, 1)

    freeenergies_ref = {
        "NH3(w)": -56.53713205
        + 0.00647387
        - 0.00543658,  # NH3(w) + correct rot. entropy (ORCA didn't get C3v)
        "OH·(w)": -75.74432719,
        "NH3·OH#(w)": -132.26581342
        + sym_correction / (constants.hartree * constants.N_A),
        "NH2·(w)": -55.87195207
        + 0.00580418
        - 0.00514973,  # NH2·(w) + correct rot. entropy (ORCA didn't get C2v)
        "H2O(w)": -76.43172074
        + 0.00562807
        - 0.00497361,  # H2O(w) + correct rot. entropy (ORCA didn't get C2v)
        "NH4+(w)": -56.98250956
        + 0.00696968
        - 0.00462348,  # NH4+(w) + correct rot. entropy (ORCA didn't get C2v)
    }  # ORCA logfiles, Eh

    freeenergies = dict(
        zip(
            model.compounds,
            rx.get_freeenergies(
                model.compounds,
                environment="gas",
                qrrho=(False, True),
                temperature=298.15,
                pressure=constants.atm,
            ),
            strict=True,
        ),
    )
    # TODO(schneiderfelipe): do a test for H+(w)
    assert [
        freeenergies[name] / (constants.hartree * constants.N_A)
        for name in freeenergies_ref
    ] == pytest.approx(
        list(freeenergies_ref.values()),
        7e-7,
    )
