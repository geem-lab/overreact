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
    st.floats(-1e5, 1e5).filter(lambda mismatch: abs(mismatch) >= 1e-4),
)
def test_get_k_rejects_inconsistent_equilibria(forward, mismatch) -> None:
    """Ensure the reverse half of an equilibrium must undo the forward half."""
    forward *= constants.kcal
    with pytest.raises(ValueError, match="reverse"):
        rx.get_k(
            rx.Scheme.parse("A <=> B"),
            delta_freeenergies=[forward, mismatch - forward],
        )


@given(
    st.floats(-50.0, 50.0),
    st.floats(0.0, 50.0),
    st.floats(100.0, 1000.0),
    st.floats(100.0, 1000.0),
)
def test_get_k_takes_temperatures_per_reaction(
    equilibrium, barrier, equilibrium_temperature, temperature
) -> None:
    """Ensure each rate constant only depends on its own reaction temperature."""
    scheme = rx.Scheme.parse("A <=> B\nB -> C")
    delta_freeenergies = np.array([equilibrium, -equilibrium, barrier]) * constants.kcal
    temperatures = [equilibrium_temperature, equilibrium_temperature, temperature]
    k = rx.get_k(
        scheme, delta_freeenergies=delta_freeenergies, temperature=temperatures
    )
    assert k == pytest.approx(
        [
            rx.get_k(scheme, delta_freeenergies=delta_freeenergies, temperature=t)[i]
            for i, t in enumerate(temperatures)
        ],
    )


def test_get_k_rejects_wrong_number_of_delta_freeenergies() -> None:
    """Ensure there is one delta free energy per reaction."""
    with pytest.raises(ValueError, match="one per reaction"):
        rx.get_k(rx.Scheme.parse("A -> B\nB -> C"), delta_freeenergies=[1.0, 2.0, 3.0])


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
    sym_correction = 298.15 * rx.thermo.change_reference_state(3, 1)
    assert 298.15 * model.scheme.activation_deltas(
        rx.get_entropies(model.compounds, environment="gas"),
    )[0] / (constants.hartree * constants.N_A) == pytest.approx(
        0.03070499
        - ((0.02288418 - 0.00647387 + 0.00543658) + 0.02022750)
        - sym_correction / (constants.hartree * constants.N_A),
        2e-4,
    )


def test_get_entropies_take_each_environment_from_its_name(hickel1992) -> None:
    """Ensure gas and solvated compounds keep their own environments together."""
    gas = rx.parse_model("data/tanaka1996/UMP2/cc-pVTZ/model.k").compounds
    both = {**gas, **hickel1992.compounds}
    assert rx.get_entropies(both) == pytest.approx(
        np.concatenate(
            [rx.get_entropies(gas), rx.get_entropies(hickel1992.compounds)],
        ),
    )


def test_get_freeenergies() -> None:
    """Ensure we can retrieve free energies."""
    model = rx.parse_model("data/hickel1992/UM06-2X/6-311++G(d,p)/model.k")
    sym_correction = 298.15 * rx.thermo.change_reference_state(3, 1)

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


def test_get_freeenergies_broadcast_over_conditions(hickel1992) -> None:
    """Ensure free energies of a batch of conditions are those of each condition."""
    temperatures = [250.0, 350.0]
    pressures = [constants.bar, 2 * constants.bar]
    assert rx.get_freeenergies(
        hickel1992.compounds,
        temperature=np.array(temperatures)[:, np.newaxis],
        pressure=np.array(pressures)[:, np.newaxis],
    ) == pytest.approx(
        np.array(
            [
                rx.get_freeenergies(hickel1992.compounds, temperature=t, pressure=p)
                for t, p in zip(temperatures, pressures, strict=True)
            ],
        ),
    )


@pytest.mark.parametrize("scale", ["l mol-1 s-1", "mmHg-1 s-1"])
def test_get_k_broadcasts_over_temperatures(hickel1992, scale) -> None:
    """Ensure rate constants of a batch of temperatures are those of each temperature."""
    temperatures = [250.0, 350.0]
    assert rx.get_k(
        hickel1992.scheme,
        hickel1992.compounds,
        scale=scale,
        temperature=np.array(temperatures)[:, np.newaxis],
    ) == pytest.approx(
        np.array(
            [
                rx.get_k(
                    hickel1992.scheme, hickel1992.compounds, scale=scale, temperature=t
                )
                for t in temperatures
            ],
        ),
    )


@pytest.mark.parametrize(
    "path",
    [
        "data/tanaka1996/UMP2/cc-pVTZ/model.k",
        "data/hickel1992/UM06-2X/6-311++G(d,p)/model.k",
    ],
)
def test_get_k_from_compounds_is_from_their_free_energies(path) -> None:
    """Ensure rate constants from compounds are those from get_freeenergies."""
    model = rx.parse_model(path)
    assert rx.get_k(model.scheme, model.compounds, tunneling=None) == pytest.approx(
        rx.get_k(
            model.scheme,
            delta_freeenergies=model.scheme.activation_freeenergies(
                rx.get_freeenergies(model.compounds),
            ),
            tunneling=None,
        ),
    )


@pytest.mark.parametrize(
    ("text", "volume", "liters_per_mole"),
    [
        ("A + B -> C", None, constants.R * 298.15 / constants.atm / constants.liter),
        ("A + B -> C", 2.0 * constants.liter, 2.0),
        ("A(w) + B(w) -> C(w)", None, 1.0),
    ],
)
def test_get_k_takes_gas_phase_free_energies_to_molar_rate_constants(
    text, volume, liters_per_mole
) -> None:
    """Ensure k = (k T / h) (V / 1 L) at zero barrier for gases, k T / h in solution."""
    assert rx.get_k(
        rx.Scheme.parse(text), delta_freeenergies=[0.0], volume=volume
    ) == pytest.approx(
        constants.k * 298.15 / constants.h * liters_per_mole,
    )


def test_get_k_rejects_temperatures_per_compound(hickel1992) -> None:
    """Ensure compounds share a temperature when computing rate constants."""
    with pytest.raises(ValueError, match="same for all compounds"):
        rx.get_k(hickel1992.scheme, hickel1992.compounds, temperature=[300.0] * 3)


def test_get_k_rejects_equilibria_at_two_temperatures() -> None:
    """Ensure both halves of an equilibrium share a temperature."""
    with pytest.raises(ValueError, match="temperature"):
        rx.get_k(
            rx.Scheme.parse("A <=> B"),
            delta_freeenergies=[1.0, -1.0],
            temperature=[300.0, 310.0],
        )


@pytest.mark.parametrize("function", [rx.get_k, rx.get_kappa])
@pytest.mark.parametrize("tunneling", ["none", "None", False, "bell"])
def test_tunneling_is_turned_off_only_by_none(hickel1992, function, tunneling) -> None:
    """Ensure unknown tunneling corrections, including spellings of None, raise."""
    with pytest.raises(ValueError, match="tunneling"):
        function(hickel1992.scheme, hickel1992.compounds, tunneling=tunneling)


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
    sym_correction = 298.15 * rx.thermo.change_reference_state(3, 1)

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
