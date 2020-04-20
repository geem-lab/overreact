#!/usr/bin/env python3

"""Tests for the high-level application programming interface."""

import numpy as np
import pytest

from overreact import api
from overreact import constants
from overreact import coords
from overreact.datasets import logfiles
from overreact import _thermo


def test_tanaka1996():
    """Reproduce literature data for CH4 + Cl⋅ -> CH3· + HCl.

    Data is as cited in doi:10.1007/BF00058703 and doi:10.1002/qua.25686 and is
    experimental except when otherwise indicated in the comments.
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
    assert zpe / constants.kcal == pytest.approx(24.34, 1e-2)
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
    k_exp = (
        np.array([1.0, 1.4, 1.9, 2.5, 3.22, 4.07, 5.05, 6.16, 7.41, 8.81, 10.0, 10.3])
        * 1e-14
    )
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
                tunneling="eckart",  # this is default
            )
        )
    k_cla = np.array(k_cla).flatten()
    k_wig = np.array(k_wig).flatten()
    k_eck = np.array(k_eck).flatten()
    assert (k_exp / k_cla).min() == pytest.approx(3.3, 2.3e-1)
    assert (k_exp / k_cla).max() == pytest.approx(11.5, 4.0e-1)
    assert (k_wig / k_cla).min() == pytest.approx(2.4, 6.2e-2)
    assert (k_wig / k_cla).max() == pytest.approx(4.2, 9.2e-2)
    assert np.log10(k_eck) == pytest.approx(np.log10(k_exp), 1.6e-2)
    assert k_eck[-1] / 1.03e-13 == pytest.approx(0.626564857388307)
    assert k_eck[-1] / 1.24e-13 == pytest.approx(
        0.5204530670241583
    )  # doi:10.1002/qua.25686
    assert k_eck[-1] / 2.17e-13 == pytest.approx(
        0.29740175258523327
    )  # doi:10.1007/BF00058703
