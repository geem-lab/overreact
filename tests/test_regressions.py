#!/usr/bin/env python3

"""Regressions against experimental/reference values.

This also tests the high-level application programming interface."""

# TODO(schneiderfelipe): transfer all comparisons with experimental/reference
# values to this file.

import numpy as np
import pytest

from overreact import api
from overreact import constants
from overreact import coords
from overreact import rates
from overreact import tunnel
from overreact import _thermo
from overreact.datasets import logfiles


def test_solubility_of_acetic_acid():
    """Reproduce literature data for AcOH(g) <=> AcOH(aq).

    Data is as cited in doi:10.1021/jp810292n and doi:10.1063/1.1416902, and
    is experimental except when otherwise indicated in the comments.
    """
    model = api.parse_model("data/acetate/model.k")
    temperature = 298.15
    pK = 4.756  # doi:10.1063/1.1416902

    enthalpies = api.get_enthalpies(model.compounds, temperature=temperature)
    entropies = api.get_entropies(model.compounds, temperature=temperature)

    delta_enthalpies = api.get_delta(model.scheme.A, enthalpies)
    # TODO(schneiderfelipe): log the contribution of reaction symmetry
    delta_entropies = api.get_delta(
        model.scheme.A, entropies
    ) + api.get_reaction_entropies(model.scheme.A)
    delta_freeenergies = delta_enthalpies - temperature * delta_entropies

    assert delta_freeenergies / constants.kcal == pytest.approx(
        [6.49, -6.49, -7.94, 7.94, -74.77, 74.77], 5e-4
    )

    concentration_correction = temperature * api.change_reference_state(
        sign=-1.0,
        temperature=temperature,
    )

    # the following tests the solvation free energy from doi:10.1021/jp810292n
    assert delta_freeenergies[2] / constants.kcal == pytest.approx(
        -6.70 + concentration_correction / constants.kcal, 8e-2
    )

    # the following tests the reaction free energy from doi:10.1063/1.1416902
    assert delta_freeenergies[0] == pytest.approx(27.147 * constants.kilo, 8e-4)
    assert delta_freeenergies[0] == pytest.approx(
        -constants.R * temperature * np.log(10 ** -pK), 7e-4
    )

    k = api.get_k(model.scheme, model.compounds, temperature=temperature)
    assert -np.log10(k[0] / k[1]) == pytest.approx(pK, 1e-3)


def test_basic_example_for_gas_phase_kinetics():
    """Ensure we can reproduce a basic gas phase example.

    This uses raw data from from doi:10.1002/qua.25686 and no calls from
    overreact.api.
    """
    temperatures = np.array([200, 298.15, 300, 400])
    delta_freeenergy = np.array([8.0, 10.3, 10.3, 12.6])

    # 4-fold symmetry TS
    sym_correction = (
        temperatures
        * _thermo.change_reference_state(4, 1, sign=-1, temperature=temperatures)
        / constants.kcal
    )
    assert delta_freeenergy + sym_correction == pytest.approx(
        [7.4, 9.4, 9.5, 11.5], 9e-3
    )

    delta_freeenergy -= (
        temperatures
        * _thermo.change_reference_state(temperature=temperatures)
        / constants.kcal
    )  # 1 atm to 1 M
    assert delta_freeenergy == pytest.approx([6.9, 8.4, 8.4, 9.9], 8e-3)

    # only concentration correction, no symmetry and no tunneling
    k = rates.eyring(
        delta_freeenergy * constants.kcal,
        temperature=temperatures,
    )
    k = rates.convert_rate_constant(k, "cm3 particle-1 s-1", molecularity=2)
    assert k == pytest.approx([2.2e-16, 7.5e-15, 7.9e-15, 5.6e-14])
    assert np.log10(k) == pytest.approx(
        np.log10([2.2e-16, 7.5e-15, 7.9e-15, 5.6e-14]), 2e-3
    )

    delta_freeenergy += sym_correction
    assert delta_freeenergy == pytest.approx([6.3, 7.6, 7.6, 8.8], 9e-3)

    # only concentration correction and symmetry, no tunneling
    k = rates.eyring(
        delta_freeenergy * constants.kcal,
        temperature=temperatures,
    )
    k = rates.convert_rate_constant(k, "cm3 particle-1 s-1", molecularity=2)
    assert k == pytest.approx([8.8e-16, 3.0e-14, 3.1e-14, 2.2e-13])
    assert np.log10(k) == pytest.approx(
        np.log10([8.8e-16, 3.0e-14, 3.1e-14, 2.2e-13]), 3e-3
    )

    kappa = tunnel.eckart(
        1218, 4.1 * constants.kcal, 3.4 * constants.kcal, temperature=temperatures
    )
    assert kappa == pytest.approx([17.1, 4.0, 3.9, 2.3], 2.1e-2)

    k *= kappa

    # concentration correction, symmetry and tunneling included
    assert k == pytest.approx([1.5e-14, 1.2e-13, 1.2e-13, 5.1e-13])
    assert np.log10(k) == pytest.approx(
        np.log10([1.5e-14, 1.2e-13, 1.2e-13, 5.1e-13]), 3e-3
    )


def test_basic_example_for_solvation_phase_kinetics():
    """Ensure we can reproduce a basic solvation phase example.

    This uses raw data from from doi:10.1002/qua.25686 and no calls from
    overreact.api.
    """
    temperatures = np.array([298.15, 300, 310, 320, 330, 340, 350])
    delta_freeenergy = np.array([10.5, 10.5, 10.8, 11.1, 11.4, 11.7, 11.9])

    # 3-fold symmetry TS
    sym_correction = (
        temperatures
        * _thermo.change_reference_state(3, 1, sign=-1, temperature=temperatures)
        / constants.kcal
    )
    assert delta_freeenergy + sym_correction == pytest.approx(
        [9.8, 9.9, 10.1, 10.4, 10.6, 10.9, 11.2], 8e-3
    )

    delta_freeenergy -= (
        temperatures
        * _thermo.change_reference_state(temperature=temperatures)
        / constants.kcal
    )  # 1 atm to 1 M
    assert delta_freeenergy == pytest.approx([8.6, 8.6, 8.8, 9.0, 9.2, 9.4, 9.6], 6e-3)

    # only concentration correction, no symmetry and no tunneling
    k = rates.eyring(
        delta_freeenergy * constants.kcal,
        temperature=temperatures,
    )
    assert k == pytest.approx([3.3e6, 3.4e6, 4.0e6, 4.7e6, 5.5e6, 6.4e6, 7.3e6], 8e-2)
    assert np.log10(k) == pytest.approx(
        np.log10([3.3e6, 3.4e6, 4.0e6, 4.7e6, 5.5e6, 6.4e6, 7.3e6]), 6e-3
    )

    delta_freeenergy += sym_correction
    assert delta_freeenergy == pytest.approx([7.9, 7.9, 8.1, 8.3, 8.5, 8.7, 8.8], 7e-3)

    # only concentration correction and symmetry, no tunneling
    k = rates.eyring(
        delta_freeenergy * constants.kcal,
        temperature=temperatures,
    )
    assert k == pytest.approx([9.8e6, 1.0e7, 1.2e7, 1.4e7, 1.7e7, 1.9e7, 2.2e7], 8e-2)
    assert np.log10(k) == pytest.approx(
        np.log10([9.8e6, 1.0e7, 1.2e7, 1.4e7, 1.7e7, 1.9e7, 2.2e7]), 5e-3
    )

    kappa = tunnel.eckart(
        986.79, 3.3 * constants.kcal, 16.4 * constants.kcal, temperature=temperatures
    )
    assert kappa == pytest.approx([2.3, 2.3, 2.2, 2.1, 2.0, 1.9, 1.9], 9e-2)

    k *= kappa

    # concentration correction, symmetry and tunneling included
    assert k == pytest.approx([2.3e7, 2.4e7, 2.7e7, 3.0e7, 3.3e7, 3.7e7, 4.1e7], 1.1e-1)
    assert np.log10(k) == pytest.approx(
        np.log10([2.3e7, 2.4e7, 2.7e7, 3.0e7, 3.3e7, 3.7e7, 4.1e7]), 6e-3
    )


def test_tanaka1996():
    """Reproduce literature data for CH4 + Cl⋅ -> CH3· + HCl.

    Data is as cited in doi:10.1007/BF00058703 and doi:10.1002/qua.25686 and
    is experimental except when otherwise indicated in the comments.

    Those tests also check for small details in the logfiles such as point
    group symmetry and errors in each energy contribution.
    """
    # CH4
    data = logfiles["tanaka1996"]["methane@UMP2/6-311G(2df,2pd)"]
    assert data.vibfreqs == pytest.approx(
        [1306, 1306, 1306, 1534, 1534, 2917, 3019, 3019, 3019], 7e-2
    )
    assert data.vibfreqs == pytest.approx(
        [1367, 1367, 1367, 1598, 1598, 3070, 3205, 3205, 3205], 2e-2
    )  # MP2/6-311G(3d,2p) from doi:10.1002/qua.25686
    point_group = coords.find_point_group(data.atommasses, data.atomcoords)
    assert coords.symmetry_number(point_group) == 12
    zpe = _thermo._gas.calc_vib_energy(data.vibfreqs, temperature=0.0)
    assert zpe / constants.kcal == pytest.approx(28.6, 2e-3)
    assert data.energy / (constants.hartree * constants.N_A) == pytest.approx(
        -40.432993195235, 3e-5
    )

    # CH3·
    data = logfiles["tanaka1996"]["CH3·@UMP2/6-311G(2df,2pd)"]
    assert data.vibfreqs == pytest.approx([580, 1383, 1383, 3002, 3184, 3184], 1.8e-1)
    assert data.vibfreqs == pytest.approx(
        [432, 1454, 1454, 3169, 3360, 3360], 1.1e-1
    )  # MP2/6-311G(3d,2p) from doi:10.1002/qua.25686
    point_group = coords.find_point_group(data.atommasses, data.atomcoords)
    assert coords.symmetry_number(point_group) == 6
    zpe = _thermo._gas.calc_vib_energy(data.vibfreqs, temperature=0.0)
    assert zpe / constants.kcal == pytest.approx(18.2, 5e-2)
    assert data.energy / (constants.hartree * constants.N_A) == pytest.approx(
        -39.755632155859, 3e-5
    )

    # HCl
    data = logfiles["tanaka1996"]["HCl@UMP2/6-311G(2df,2pd)"]
    assert data.vibfreqs == pytest.approx([2991], 2e-2)
    assert data.vibfreqs == pytest.approx(
        [3028], 8e-3
    )  # MP2/6-311G(3d,2p) from doi:10.1002/qua.25686
    point_group = coords.find_point_group(data.atommasses, data.atomcoords)
    assert coords.symmetry_number(point_group) == 1
    zpe = _thermo._gas.calc_vib_energy(data.vibfreqs, temperature=0.0)
    assert zpe / constants.kcal == pytest.approx(4.3, 2e-2)
    assert data.energy / (constants.hartree * constants.N_A) == pytest.approx(
        -460.351060602085, 2e-5
    )

    # CH3-H-Cl
    # vibrations and ZPE are from an MP2/6-311G(3d,2p) calculation, see the
    # reference in doi:10.1007/BF00058703
    data = logfiles["tanaka1996"]["H3CHCl‡@UMP2/6-311G(2df,2pd)"]
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
        8e-2,
    )
    assert data.vibfreqs == pytest.approx(
        [-1218, 369, 369, 516, 962, 962, 1217, 1460, 1460, 3123, 3294, 3294], 8e-2
    )  # MP2/6-311G(3d,2p) from doi:10.1002/qua.25686
    point_group = coords.find_point_group(data.atommasses, data.atomcoords)
    assert coords.symmetry_number(point_group) == 3
    zpe = _thermo._gas.calc_vib_energy(data.vibfreqs, temperature=0.0)
    assert zpe / constants.kcal == pytest.approx(24.34, 8e-3)
    assert data.energy / (constants.hartree * constants.N_A) == pytest.approx(
        -500.105486048554, 2e-5
    )

    # Cl·
    data = logfiles["tanaka1996"]["Cl·@UMP2/6-311G(2df,2pd)"]
    point_group = coords.find_point_group(data.atommasses, data.atomcoords)
    assert coords.symmetry_number(point_group) == 1
    zpe = _thermo._gas.calc_vib_energy(data.vibfreqs, temperature=0.0)
    assert zpe / constants.kcal == pytest.approx(0.0)
    assert data.energy / (constants.hartree * constants.N_A) == pytest.approx(
        -459.680787066187, 3e-5
    )

    # Create model and test it
    model = api.parse_model("data/tanaka1996/UMP2/6-311G(2df,2pd)/model.jk")
    assert model.compounds["H3CHCl‡"].symmetry == 4

    electronic_barrier = api.get_delta(
        model.scheme.B, [data.energy for name, data in model.compounds.items()]
    )
    cryo_barrier = api.get_delta(
        model.scheme.B, api.get_enthalpies(model.compounds, temperature=0.0)
    )
    assert cryo_barrier / constants.kcal == pytest.approx(2.8, 2e-2)
    assert cryo_barrier / constants.kcal == pytest.approx(
        2.12, 3.5e-1
    )  # PMP2/6-311G(3d,2p), see doi:10.1007/BF00058703
    assert cryo_barrier / (constants.hartree * constants.N_A) == pytest.approx(
        0.0045099, 4e-3
    )  # at UMP2/6-311G(2df,2pd)
    assert cryo_barrier / constants.kcal == pytest.approx(
        2.83, 4e-3
    )  # at UMP2/6-311G(2df,2pd)

    assert electronic_barrier / constants.kcal == pytest.approx(
        6.41, 8e-2
    )  # PMP2/6-311G(3d,2p), see doi:10.1007/BF00058703
    assert electronic_barrier / (constants.hartree * constants.N_A) == pytest.approx(
        0.010948, 6e-5
    )  # at UMP2/6-311G(2df,2pd)
    assert electronic_barrier / constants.kcal == pytest.approx(
        6.87, 6e-5
    )  # at UMP2/6-311G(2df,2pd)

    # Reaction energies as calculated from the model
    temperatures = [0.0, 200.0, 298.15, 300.0, 400.0]
    delta_internal_energies = np.array(
        [
            api.get_delta(
                model.scheme.A,
                api.get_internal_energies(model.compounds, temperature=temperature),
            )
            for temperature in temperatures
        ]
    )
    delta_enthalpies = np.array(
        [
            api.get_delta(
                model.scheme.A,
                api.get_enthalpies(model.compounds, temperature=temperature),
            )
            for temperature in temperatures
        ]
    )
    delta_entropies = np.array(
        [
            api.get_delta(
                model.scheme.A,
                api.get_entropies(model.compounds, temperature=temperature),
            )
            for temperature in temperatures
        ]
    )
    delta_freeenergies = np.array(
        [
            api.get_delta(
                model.scheme.A,
                api.get_enthalpies(model.compounds, temperature=temperature)
                - temperature
                * api.get_entropies(model.compounds, temperature=temperature),
            )
            for temperature in temperatures
        ]
    )
    assert delta_internal_energies / constants.kcal == pytest.approx(
        [0.05893551, 0.48739609, 0.77537336, 0.78089114, 1.06380531]
    )  # at UMP2/6-311G(2df,2pd)
    assert delta_enthalpies == pytest.approx(delta_internal_energies)
    assert delta_entropies / constants.calorie == pytest.approx(
        [0.0, 7.55883822, 8.72739303, 8.74584257, 9.56232633]
    )  # at UMP2/6-311G(2df,2pd)
    assert delta_freeenergies / constants.kcal == pytest.approx(
        [0.05893551, -1.02437155, -1.82669887, -1.84286164, -2.76112523]
    )  # at UMP2/6-311G(2df,2pd)

    # Barrier energies as calculated from the model
    temperatures = [200.0, 298.15, 300.0, 400.0]
    delta_internal_energies = np.array(
        [
            api.get_delta(
                model.scheme.B,
                api.get_internal_energies(model.compounds, temperature=temperature),
            )
            for temperature in temperatures
        ]
    )
    delta_enthalpies = np.array(
        [
            api.get_delta(
                model.scheme.B,
                api.get_enthalpies(model.compounds, temperature=temperature),
            )
            for temperature in temperatures
        ]
    )
    delta_entropies = np.array(
        [
            api.get_delta(
                model.scheme.B,
                api.get_entropies(model.compounds, temperature=temperature),
            )
            for temperature in temperatures
        ]
    )
    delta_freeenergies = np.array(
        [
            api.get_delta(
                model.scheme.B,
                api.get_enthalpies(model.compounds, temperature=temperature)
                - temperature
                * api.get_entropies(model.compounds, temperature=temperature),
            )
            for temperature in temperatures
        ]
    )
    assert delta_internal_energies / constants.kcal == pytest.approx(
        [2.43001912, 2.52633752, 2.52978082, 2.77671586]
    )  # at UMP2/6-311G(2df,2pd)
    assert delta_enthalpies / constants.kcal == pytest.approx(
        [2.03257827, 1.93385257, 1.93361955, 1.98183415]
    )  # at UMP2/6-311G(2df,2pd)
    assert delta_entropies / constants.calorie == pytest.approx(
        [-17.79380692, -18.22041152, -18.22119077, -18.08983486]
    )  # at UMP2/6-311G(2df,2pd)
    assert delta_freeenergies / constants.kcal == pytest.approx(
        [5.59133966, 7.36626827, 7.39997678, 9.2177681]
    )  # at UMP2/6-311G(2df,2pd)

    # Reaction rate constants as calculated from the model
    temperatures = [
        200.0,
        210.0,
        220.0,
        230.0,
        240.0,
        250.0,
        260.0,
        270.0,
        280.0,
        290.0,
        298.15,
        300.0,
    ]

    k_cla = []
    k_wig = []
    k_eck = []
    for temperature in temperatures:
        k_cla.append(
            api.get_k(
                model.scheme,
                model.compounds,
                temperature=temperature,
                scale="cm3 particle-1 s-1",
                tunneling=None,
            )
        )
        k_wig.append(
            api.get_k(
                model.scheme,
                model.compounds,
                temperature=temperature,
                scale="cm3 particle-1 s-1",
                tunneling="wigner",
            )
        )
        k_eck.append(
            api.get_k(
                model.scheme,
                model.compounds,
                temperature=temperature,
                scale="cm3 particle-1 s-1",
                # tunneling="eckart",  # this is default
            )
        )
    k_cla = np.asarray(k_cla).flatten()
    k_wig = np.asarray(k_wig).flatten()
    k_eck = np.asarray(k_eck).flatten()

    k_exp = np.array(
        [
            1.0e-14,
            1.4e-14,
            1.9e-14,
            2.5e-14,
            3.22e-14,
            4.07e-14,
            5.05e-14,
            6.16e-14,
            7.41e-14,
            8.81e-14,
            10.0e-14,
            10.3e-14,
        ]
    )

    assert k_eck == pytest.approx(k_exp)
    assert np.log10(k_eck) == pytest.approx(np.log10(k_exp), 4e-2)

    # doi:10.1002/qua.25686
    assert k_eck[-1] == pytest.approx(1.24e-13)
    assert np.log10(k_eck[-1]) == pytest.approx(np.log10(1.24e-13), 6e-3)
    assert (k_wig / k_cla).min() == pytest.approx(2.4, 7e-2)
    assert (k_wig / k_cla).max() == pytest.approx(4.2, 1.0e-1)

    # doi:10.1007/BF00058703
    assert k_eck[-1] == pytest.approx(2.17e-13)
    assert np.log10(k_eck[-1]) == pytest.approx(np.log10(2.17e-13), 3e-2)
