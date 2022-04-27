#!/usr/bin/env python3

"""Regressions against experimental/reference values.

This also tests the high-level application programming interface."""

import numpy as np
import pytest
from scipy import stats

import overreact as rx
from overreact import _constants as constants
from overreact import _datasets as datasets

# TODO(schneiderfelipe): transfer all comparisons with experimental/reference
# values to this file.


def test_basic_example_for_solvation_equilibria():
    """Reproduce literature data for AcOH(g) <=> AcOH(aq).

    Data is as cited in doi:10.1021/jp810292n and doi:10.1063/1.1416902, and
    is experimental except when otherwise indicated in the comments.
    """
    model = rx.parse_model("data/acetate/Orca4/model.k")
    temperature = 298.15
    pK = 4.756  # doi:10.1063/1.1416902

    acid_energy = -constants.R * temperature * np.log(10**-pK) / constants.kcal
    solv_energy = (
        -229.04018997
        - -229.075245654407
        + -228.764256345282
        - (-229.02825429 - -229.064152538732 + -228.749485597775)
    ) * (constants.hartree * constants.N_A / constants.kcal)
    charged_solv_energy = (
        -228.59481510
        - -228.617274320359
        + -228.292486796947
        - (-228.47794098 - -228.500117698893 + -228.169992151890)
    ) * (constants.hartree * constants.N_A / constants.kcal)
    delta_freeenergies_ref = [
        acid_energy,
        -acid_energy,
        solv_energy,
        -solv_energy,
        charged_solv_energy,
        -charged_solv_energy,
    ]

    concentration_correction = -temperature * rx.change_reference_state(
        temperature=temperature
    )

    for qrrho in [False, (False, True), True]:
        # TODO(schneiderfelipe): log the contribution of reaction symmetry
        delta_freeenergies = rx.get_delta(
            model.scheme.A,
            rx.get_freeenergies(model.compounds, temperature=temperature, qrrho=qrrho),
        ) - temperature * rx.get_reaction_entropies(
            model.scheme.A, temperature=temperature
        )
        assert delta_freeenergies / constants.kcal == pytest.approx(
            delta_freeenergies_ref, 7e-3
        )

        # the following tests the solvation free energy from doi:10.1021/jp810292n
        assert delta_freeenergies[2] / constants.kcal == pytest.approx(
            -6.70 + concentration_correction / constants.kcal, 1.5e-1
        )

        # the following tests the reaction free energy from doi:10.1063/1.1416902
        assert delta_freeenergies[0] == pytest.approx(27.147 * constants.kilo, 7e-3)
        assert delta_freeenergies[0] == pytest.approx(
            -constants.R * temperature * np.log(10**-pK), 7e-3
        )

        k = rx.get_k(
            model.scheme, model.compounds, temperature=temperature, qrrho=qrrho
        )
        assert -np.log10(k[0] / k[1]) == pytest.approx(pK, 7e-3)


def test_basic_example_for_solvation_phase_kinetics():
    """Reproduce literature data for NH3(w) + OH·(w) -> NH2·(w) + H2O(w).

    This uses raw data from from doi:10.1002/qua.25686 and no calls from
    overreact.api.
    """
    temperatures = np.array([298.15, 300, 310, 320, 330, 340, 350])
    delta_freeenergies = np.array([10.5, 10.5, 10.8, 11.1, 11.4, 11.7, 11.9])

    # 3-fold symmetry TS
    sym_correction = (
        -temperatures
        * rx.change_reference_state(3, 1, temperature=temperatures)
        / constants.kcal
    )
    assert delta_freeenergies + sym_correction == pytest.approx(
        [9.8, 9.9, 10.1, 10.4, 10.6, 10.9, 11.2], 8e-3
    )

    delta_freeenergies -= (
        temperatures
        * rx.change_reference_state(temperature=temperatures)
        / constants.kcal
    )  # 1 atm to 1 M
    assert delta_freeenergies == pytest.approx(
        [8.6, 8.6, 8.8, 9.0, 9.2, 9.4, 9.6], 6e-3
    )

    # only concentration correction, no symmetry and no tunneling
    k = rx.rates.eyring(delta_freeenergies * constants.kcal, temperature=temperatures)
    assert k == pytest.approx([3.3e6, 3.4e6, 4.0e6, 4.7e6, 5.5e6, 6.4e6, 7.3e6], 8e-2)
    assert np.log10(k) == pytest.approx(
        np.log10([3.3e6, 3.4e6, 4.0e6, 4.7e6, 5.5e6, 6.4e6, 7.3e6]), 6e-3
    )

    # only concentration correction and symmetry, no tunneling
    delta_freeenergies += sym_correction
    assert delta_freeenergies == pytest.approx(
        [7.9, 7.9, 8.1, 8.3, 8.5, 8.7, 8.8], 7e-3
    )
    k = rx.rates.eyring(delta_freeenergies * constants.kcal, temperature=temperatures)
    assert k == pytest.approx([9.8e6, 1.0e7, 1.2e7, 1.4e7, 1.7e7, 1.9e7, 2.2e7], 8e-2)
    assert np.log10(k) == pytest.approx(
        np.log10([9.8e6, 1.0e7, 1.2e7, 1.4e7, 1.7e7, 1.9e7, 2.2e7]), 5e-3
    )

    # concentration correction, symmetry and tunneling included
    kappa = rx.tunnel.eckart(
        986.79,
        3.3 * constants.kcal,
        16.4 * constants.kcal,
        temperature=temperatures,
    )
    assert kappa == pytest.approx([2.3, 2.3, 2.2, 2.1, 2.0, 1.9, 1.9], 9e-2)

    k *= kappa
    assert k == pytest.approx([2.3e7, 2.4e7, 2.7e7, 3.0e7, 3.3e7, 3.7e7, 4.1e7], 1.1e-1)
    assert np.log10(k) == pytest.approx(
        np.log10([2.3e7, 2.4e7, 2.7e7, 3.0e7, 3.3e7, 3.7e7, 4.1e7]), 6e-3
    )


def test_basic_example_for_gas_phase_kinetics():
    """Reproduce literature data for CH4 + Cl⋅ -> CH3· + HCl.

    This uses raw data from from doi:10.1002/qua.25686 and no calls from
    overreact.api.
    """
    temperatures = np.array([200, 298.15, 300, 400])
    delta_freeenergies = np.array([8.0, 10.3, 10.3, 12.6])

    # 4-fold symmetry TS
    sym_correction = (
        -temperatures
        * rx.change_reference_state(4, 1, temperature=temperatures)
        / constants.kcal
    )
    assert delta_freeenergies + sym_correction == pytest.approx(
        [7.4, 9.4, 9.5, 11.5], 9e-3
    )

    delta_freeenergies -= (
        temperatures
        * rx.change_reference_state(temperature=temperatures)
        / constants.kcal
    )  # 1 atm to 1 M
    assert delta_freeenergies == pytest.approx([6.9, 8.4, 8.4, 9.9], 8e-3)

    # only concentration correction, no symmetry and no tunneling
    k = rx.rates.eyring(delta_freeenergies * constants.kcal, temperature=temperatures)
    k = rx.rates.convert_rate_constant(k, "cm3 particle-1 s-1", molecularity=2)
    assert 1e16 * k == pytest.approx(
        1e16 * np.array([2.2e-16, 7.5e-15, 7.9e-15, 5.6e-14]), 7e-2
    )
    assert np.log10(k) == pytest.approx(
        np.log10([2.2e-16, 7.5e-15, 7.9e-15, 5.6e-14]), 2e-3
    )

    # only concentration correction and symmetry, no tunneling
    delta_freeenergies += sym_correction
    assert delta_freeenergies == pytest.approx([6.3, 7.6, 7.6, 8.8], 9e-3)
    k = rx.rates.eyring(delta_freeenergies * constants.kcal, temperature=temperatures)
    k = rx.rates.convert_rate_constant(k, "cm3 particle-1 s-1", molecularity=2)
    assert 1e16 * k == pytest.approx(
        1e16 * np.array([8.8e-16, 3.0e-14, 3.1e-14, 2.2e-13]), 8e-2
    )
    assert np.log10(k) == pytest.approx(
        np.log10([8.8e-16, 3.0e-14, 3.1e-14, 2.2e-13]), 3e-3
    )

    # concentration correction, symmetry and tunneling included
    kappa = rx.tunnel.wigner(1218, temperature=temperatures)
    assert kappa[0] == pytest.approx(4.2, 3e-4)
    assert kappa[2] == pytest.approx(2.4, 1e-2)

    kappa = rx.tunnel.eckart(
        1218,
        4.1 * constants.kcal,
        3.4 * constants.kcal,
        temperature=temperatures,
    )
    assert kappa == pytest.approx([17.1, 4.0, 3.9, 2.3], 2.1e-2)

    k *= kappa
    assert 1e16 * k == pytest.approx(
        1e16 * np.array([1.5e-14, 1.2e-13, 1.2e-13, 5.1e-13]), 7e-2
    )
    assert np.log10(k) == pytest.approx(
        np.log10([1.5e-14, 1.2e-13, 1.2e-13, 5.1e-13]), 3e-3
    )


def test_rate_constants_for_hickel1992():
    """Reproduce literature data for NH3(w) + OH·(w) -> NH2·(w) + H2O(w).

    Data is as cited in doi:10.1002/qua.25686 and is experimental except when
    otherwise indicated in the comments.

    Those tests check for consistency with the literature in terms of
    reaction rate constants.
    """
    theory = "UM06-2X"
    basisset = "6-311++G(d,p)"
    model = rx.parse_model(f"data/hickel1992/{theory}/{basisset}/model.k")

    temperatures = np.array([298.15, 300, 310, 320, 330, 340, 350])
    k_cla_ref = np.array([9.8e6, 1.0e7, 1.2e7, 1.4e7, 1.7e7, 1.9e7, 2.2e7])
    k_eck_ref = np.array([2.3e7, 2.4e7, 2.7e7, 3.0e7, 3.3e7, 3.7e7, 4.1e7])

    k_cla = []
    k_eck = []
    for temperature in temperatures:
        k_cla.append(
            rx.get_k(
                model.scheme,
                model.compounds,
                tunneling=None,
                qrrho=(False, True),
                scale="M-1 s-1",
                temperature=temperature,
            )[0]
        )
        k_eck.append(
            rx.get_k(
                model.scheme,
                model.compounds,
                # tunneling="eckart",  # this is default
                qrrho=(False, True),
                scale="M-1 s-1",
                temperature=temperature,
            )[0]
        )
    k_cla = np.asarray(k_cla).flatten()
    k_eck = np.asarray(k_eck).flatten()
    assert k_eck / k_cla == pytest.approx([2.3, 2.3, 2.2, 2.1, 2.0, 1.9, 1.9], 7e-2)

    assert k_cla == pytest.approx(k_cla_ref, 1.2e-1)
    assert k_eck == pytest.approx(k_eck_ref, 9e-2)
    assert np.log10(k_cla) == pytest.approx(np.log10(k_cla_ref), 8e-3)
    assert np.log10(k_eck) == pytest.approx(np.log10(k_eck_ref), 5e-3)

    for k, k_ref, tols in zip(
        [k_cla, k_eck],
        [k_cla_ref, k_eck_ref],
        [(1.0e-1, 0.62, 2e-3, 5e-8, 3e-2), (1.1e-1, 0.75, 2e-3, 3e-8, 2e-2)],
    ):
        linregress = stats.linregress(np.log10(k), np.log10(k_ref))
        assert linregress.slope == pytest.approx(1.0, tols[0])
        assert linregress.intercept == pytest.approx(0.0, abs=tols[1])

        assert linregress.rvalue**2 == pytest.approx(1.0, tols[2])
        assert linregress.pvalue == pytest.approx(0.0, abs=tols[3])
        assert linregress.pvalue < 0.01
        assert linregress.stderr == pytest.approx(0.0, abs=tols[4])


def test_rate_constants_for_tanaka1996():
    """Reproduce literature data for CH4 + Cl⋅ -> CH3· + HCl.

    Data is as cited in doi:10.1007/BF00058703 and doi:10.1002/qua.25686 and
    is experimental except when otherwise indicated in the comments.

    Those tests check for consistency with the literature in terms of
    reaction rate constants.
    """
    theory = "UMP2"
    basisset = "cc-pVTZ"  # not the basis used in the ref., but close enough
    model = rx.parse_model(f"data/tanaka1996/{theory}/{basisset}/model.k")

    temperatures = np.array(
        [
            200.0,
            # 210.0,
            # 220.0,
            # 230.0,
            # 240.0,
            # 250.0,
            # 260.0,
            # 270.0,
            # 280.0,
            # 290.0,
            298.15,
            300.0,
            400.0,
        ]
    )
    k_cla_ref = np.array([8.8e-16, 3.0e-14, 3.1e-14, 2.2e-13])
    k_eck_ref = np.array([1.5e-14, 1.2e-13, 1.2e-13, 5.1e-13])
    k_exp = np.array(
        [
            1.0e-14,
            # 1.4e-14,
            # 1.9e-14,
            # 2.5e-14,
            # 3.22e-14,
            # 4.07e-14,
            # 5.05e-14,
            # 6.16e-14,
            # 7.41e-14,
            # 8.81e-14,
            10.0e-14,
            10.3e-14,
            # no data for 400K?
        ]
    )

    k_cla = []
    k_wig = []
    k_eck = []
    for temperature in temperatures:
        k_cla.append(
            rx.get_k(
                model.scheme,
                model.compounds,
                tunneling=None,
                qrrho=True,
                scale="cm3 particle-1 s-1",
                temperature=temperature,
            )[0]
        )
        k_wig.append(
            rx.get_k(
                model.scheme,
                model.compounds,
                tunneling="wigner",
                qrrho=True,
                scale="cm3 particle-1 s-1",
                temperature=temperature,
            )[0]
        )
        k_eck.append(
            rx.get_k(
                model.scheme,
                model.compounds,
                # tunneling="eckart",  # this is default
                qrrho=True,
                scale="cm3 particle-1 s-1",
                temperature=temperature,
            )[0]
        )
    k_cla = np.asarray(k_cla).flatten()
    k_wig = np.asarray(k_wig).flatten()
    k_eck = np.asarray(k_eck).flatten()
    assert k_eck / k_cla == pytest.approx([17.1, 4.0, 3.9, 2.3], 1.7e-1)

    assert 1e16 * k_cla == pytest.approx(1e16 * k_cla_ref, 1.9e-1)
    assert 1e16 * k_eck == pytest.approx(1e16 * k_eck_ref, 3.2e-1)
    assert 1e16 * k_eck[:-1] == pytest.approx(1e16 * k_exp, 8e-2)
    assert np.log10(k_cla) == pytest.approx(np.log10(k_cla_ref), 6e-3)
    assert np.log10(k_eck) == pytest.approx(np.log10(k_eck_ref), 2e-2)
    assert np.log10(k_eck[:-1]) == pytest.approx(np.log10(k_exp), 3e-3)

    for k, k_ref, tols in zip(
        [k_cla, k_eck, k_eck[:-1]],
        [k_cla_ref, k_eck_ref, k_exp],
        [
            (2e-2, 0.08, 9e-6, 5e-6, 3e-3),
            (5e-2, 0.52, 4e-4, 2e-4, 2e-2),
            (5e-2, 0.60, 3e-6, 2e-3, 2e-3),
        ],
    ):
        linregress = stats.linregress(np.log10(k), np.log10(k_ref))
        assert linregress.slope == pytest.approx(1.0, tols[0])
        assert linregress.intercept == pytest.approx(0.0, abs=tols[1])

        assert linregress.rvalue**2 == pytest.approx(1.0, tols[2])
        assert linregress.pvalue == pytest.approx(0.0, abs=tols[3])
        assert linregress.pvalue < 0.01
        assert linregress.stderr == pytest.approx(0.0, abs=tols[4])


def test_delta_energies_for_hickel1992():
    """Reproduce literature data for NH3(w) + OH·(w) -> NH2·(w) + H2O(w).

    Data is as cited in doi:10.1002/qua.25686 and is experimental except when
    otherwise indicated in the comments.

    Those tests check for consistency with the literature in terms of
    chemical kinetics and thermochemistry.
    """
    theory = "UM06-2X"
    basisset = "6-311++G(d,p)"
    model = rx.parse_model(f"data/hickel1992/{theory}/{basisset}/model.k")

    temperatures = np.array([298.15, 300, 310, 320, 330, 340, 350])
    delta_freeenergies_ref = [9.8, 9.9, 10.1, 10.4, 10.6, 10.9, 11.2]

    delta_freeenergies = []
    for temperature in temperatures:
        freeenergies = rx.get_freeenergies(
            model.compounds, temperature=temperature, qrrho=(False, True)
        )
        delta_freeenergy = (
            rx.get_delta(model.scheme.B, freeenergies)
            - temperature
            * rx.get_reaction_entropies(model.scheme.B, temperature=temperature)
        )[0]

        delta_freeenergies.append(delta_freeenergy)
    delta_freeenergies = np.asarray(delta_freeenergies)

    assert delta_freeenergies / constants.kcal == pytest.approx(
        delta_freeenergies_ref
        - temperatures
        * rx.change_reference_state(temperature=temperatures)
        / constants.kcal,
        2e-2,
    )  # M06-2X/6-311++G(d,p) from doi:10.1002/qua.25686

    # extra symmetry is required for this reaction since the transition state
    # is nonsymmetric
    assert model.compounds["NH3·OH#(w)"].symmetry == 3

    delta_freeenergies_ref = [7.9, 7.9, 8.1, 8.3, 8.5, 8.7, 8.8]
    assert delta_freeenergies / constants.kcal == pytest.approx(
        delta_freeenergies_ref, 2e-2
    )  # M06-2X/6-311++G(d,p) from doi:10.1002/qua.25686


def test_delta_energies_for_tanaka1996():
    """Reproduce literature data for CH4 + Cl⋅ -> CH3· + HCl.

    Data is as cited in doi:10.1007/BF00058703 and doi:10.1002/qua.25686 and
    is experimental except when otherwise indicated in the comments.

    Those tests check for consistency with the literature in terms of
    chemical kinetics and thermochemistry.
    """
    theory = "UMP2"
    basisset = "6-311G(2d,p)"
    model = rx.parse_model(f"data/tanaka1996/{theory}/{basisset}/model.k")

    temperatures = [0.0]
    delta_freeenergies_ref = [5.98]

    delta_freeenergies = []
    for temperature in temperatures:
        freeenergies = rx.get_freeenergies(model.compounds, temperature=temperature)
        delta_freeenergy = (
            rx.get_delta(model.scheme.B, freeenergies)
            - temperature
            * rx.get_reaction_entropies(model.scheme.B, temperature=temperature)[0]
        )[0]

        delta_freeenergies.append(delta_freeenergy)
    delta_freeenergies = np.asarray(delta_freeenergies)

    assert delta_freeenergies / constants.kcal == pytest.approx(
        delta_freeenergies_ref, 4e-2
    )  # UMP2/6-311G(2d,p) doi:10.1007/BF00058703

    # testing now another level of theory!
    basisset = "cc-pVTZ"  # not the basis used in the ref., but close enough
    model = rx.parse_model(f"data/tanaka1996/{theory}/{basisset}/model.k")

    # no extra symmetry required for this reaction since the transition state
    # is symmetric
    assert model.compounds["H3CHCl‡"].symmetry is None

    temperatures = np.array([200, 298.15, 300, 400])
    delta_freeenergies_ref = [7.4, 9.4, 9.5, 11.5]

    delta_freeenergies = []
    for temperature in temperatures:
        freeenergies = rx.get_freeenergies(model.compounds, temperature=temperature)
        delta_freeenergy = (
            rx.get_delta(model.scheme.B, freeenergies)
            - temperature
            * rx.get_reaction_entropies(model.scheme.B, temperature=temperature)[0]
        )[0]

        delta_freeenergies.append(delta_freeenergy)
    delta_freeenergies = np.asarray(delta_freeenergies)

    assert delta_freeenergies / constants.kcal == pytest.approx(
        delta_freeenergies_ref, 2e-2
    )  # UMP2/6-311G(3d,2p) from doi:10.1002/qua.25686


def test_logfiles_for_hickel1992():
    """Reproduce literature data for NH3(w) + OH·(w) -> NH2·(w) + H2O(w).

    Data is as cited in doi:10.1002/qua.25686 and is experimental except when
    otherwise indicated in the comments.

    Those tests check for details in the logfiles such as point group symmetry
    and frequencies.
    """
    theory = "UM06-2X"
    basisset = "6-311++G(d,p)"

    # NH3(w)
    data = datasets.logfiles["hickel1992"][f"NH3@{theory}/{basisset}"]
    point_group = rx.coords.find_point_group(data.atommasses, data.atomcoords)
    assert rx.coords.symmetry_number(point_group) == 3

    assert data.vibfreqs == pytest.approx(
        [1022, 1691, 1691, 3506, 3577, 3577], 5e-2
    )  # doi:10.1002/qua.25686
    assert data.vibfreqs == pytest.approx(
        [1065.8, 1621.5, 1620.6, 3500.2, 3615.5, 3617.3], 4e-2
    )  # M06-2X/6-311++G(d,p) from doi:10.1002/qua.25686

    # OH·(w)
    data = datasets.logfiles["hickel1992"][f"OH·@{theory}/{basisset}"]
    point_group = rx.coords.find_point_group(data.atommasses, data.atomcoords)
    assert rx.coords.symmetry_number(point_group) == 1

    assert data.vibfreqs == pytest.approx([3737.8], 2e-2)  # doi:10.1002/qua.25686
    assert data.vibfreqs == pytest.approx(
        [3724.3], 2e-2
    )  # M06-2X/6-311++G(d,p) from doi:10.1002/qua.25686

    # NH2·(w)
    data = datasets.logfiles["hickel1992"][f"NH2·@{theory}/{basisset}"]
    point_group = rx.coords.find_point_group(data.atommasses, data.atomcoords)
    assert rx.coords.symmetry_number(point_group) == 2

    assert data.vibfreqs == pytest.approx(
        [1497.3, 3220.0, 3301.1], 7e-2
    )  # doi:10.1002/qua.25686
    assert data.vibfreqs == pytest.approx(
        [1471.2, 3417.6, 3500.8], 9e-3
    )  # M06-2X/6-311++G(d,p) from doi:10.1002/qua.25686

    # H2O(w)
    data = datasets.logfiles["hickel1992"][f"H2O@{theory}/{basisset}"]
    point_group = rx.coords.find_point_group(data.atommasses, data.atomcoords)
    assert rx.coords.symmetry_number(point_group) == 2

    assert data.vibfreqs == pytest.approx(
        [1594.6, 3656.7, 3755.8], 6e-2
    )  # doi:10.1002/qua.25686
    assert data.vibfreqs == pytest.approx(
        [1570.4, 3847.9, 3928.9], 6e-3
    )  # M06-2X/6-311++G(d,p) from doi:10.1002/qua.25686

    # NH3·OH#(w)
    data = datasets.logfiles["hickel1992"][f"NH3·OH@{theory}/{basisset}"]
    point_group = rx.coords.find_point_group(data.atommasses, data.atomcoords)
    assert rx.coords.symmetry_number(point_group) == 1

    # NOTE(schneiderfelipe): I couldn't find any frequency reference data for
    # this transition state.


def test_logfiles_for_tanaka1996():
    """Reproduce literature data for CH4 + Cl⋅ -> CH3· + HCl.

    Data is as cited in doi:10.1007/BF00058703 and doi:10.1002/qua.25686 and
    is experimental except when otherwise indicated in the comments.

    Those tests check for details in the logfiles such as point group symmetry
    and frequencies.
    """
    theory = "UMP2"
    basisset = "cc-pVTZ"  # not the basis used in the ref., but close enough

    # CH4
    data = datasets.logfiles["tanaka1996"][f"methane@{theory}/{basisset}"]
    point_group = rx.coords.find_point_group(data.atommasses, data.atomcoords)
    assert rx.coords.symmetry_number(point_group) == 12

    assert data.vibfreqs == pytest.approx(
        [1306, 1306, 1306, 1534, 1534, 2917, 3019, 3019, 3019], 7e-2
    )  # doi:10.1002/qua.25686
    assert data.vibfreqs == pytest.approx(
        [1367, 1367, 1367, 1598, 1598, 3070, 3203, 3203, 3205], 8e-3
    )  # UMP2/6-311G(3d,2p) from doi:10.1002/qua.25686

    # CH3·
    data = datasets.logfiles["tanaka1996"][f"CH3·@{theory}/{basisset}"]
    point_group = rx.coords.find_point_group(data.atommasses, data.atomcoords)
    assert rx.coords.symmetry_number(point_group) == 6

    assert data.vibfreqs == pytest.approx(
        [580, 1383, 1383, 3002, 3184, 3184], 1.4e-1
    )  # doi:10.1002/qua.25686
    assert data.vibfreqs == pytest.approx(
        [432, 1454, 1454, 3169, 3360, 3360], 1.7e-1
    )  # UMP2/6-311G(3d,2p) from doi:10.1002/qua.25686

    # HCl
    data = datasets.logfiles["tanaka1996"][f"HCl@{theory}/{basisset}"]
    point_group = rx.coords.find_point_group(data.atommasses, data.atomcoords)
    assert rx.coords.symmetry_number(point_group) == 1

    assert data.vibfreqs == pytest.approx([2991], 3e-2)  # doi:10.1002/qua.25686
    assert data.vibfreqs == pytest.approx(
        [3028], 9e-3
    )  # UMP2/6-311G(3d,2p) from doi:10.1002/qua.25686

    # Cl·
    data = datasets.logfiles["tanaka1996"][f"Cl·@{theory}/{basisset}"]
    point_group = rx.coords.find_point_group(data.atommasses, data.atomcoords)
    assert rx.coords.symmetry_number(point_group) == 1

    # CH3-H-Cl
    data = datasets.logfiles["tanaka1996"][f"H3CHCl‡@{theory}/{basisset}"]
    point_group = rx.coords.find_point_group(data.atommasses, data.atomcoords)
    assert rx.coords.symmetry_number(point_group) == 3

    # NOTE(schneiderfelipe): vibrations are from an UMP2/6-311G(3d,2p)
    # calculation, see the reference in doi:10.1007/BF00058703
    assert data.vibfreqs == pytest.approx(
        [
            -1214.38,
            368.97,
            369.97,
            515.68,
            962.03,
            962.03,
            1217.20,
            1459.97,
            1459.97,
            3123.29,
            3293.74,
            3293.74,
        ],
        6e-2,
    )
    assert data.vibfreqs == pytest.approx(
        [-1218, 369, 369, 516, 962, 962, 1217, 1460, 1460, 3123, 3294, 3294],
        7e-2,
    )  # UMP2/6-311G(3d,2p) from doi:10.1002/qua.25686
