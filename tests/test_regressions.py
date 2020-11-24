#!/usr/bin/env python3

"""Regressions against experimental/reference values."""

# TODO(schneiderfelipe): transfer all comparisons with experimental/reference
# values to this file.

import numpy as np
import pytest

from overreact import api
from overreact import constants


def test_solubility_of_acetic_acid():
    """Reproduce literature data for AcOH(g) <=> AcOH(aq).

    Data is as cited in doi:10.1021/jp810292n and is experimental except when
    otherwise indicated in the comments.
    """
    model = api.parse_model("data/acetate/model.k")
    temperature = 298.15
    pK = 4.76  # doi:10.1063/1.1416902

    enthalpies = api.get_enthalpies(model.compounds, temperature=temperature)
    entropies = api.get_entropies(model.compounds, temperature=temperature)

    delta_enthalpies = api.get_delta(model.scheme.A, enthalpies)
    # TODO(schneiderfelipe): log the contribution of reaction symmetry
    delta_entropies = api.get_delta(
        model.scheme.A, entropies
    ) + api.get_reaction_entropies(model.scheme.A)
    delta_freeenergies = delta_enthalpies - temperature * delta_entropies

    assert delta_freeenergies / constants.kcal == pytest.approx(
        [6.50, -6.50, -7.94, 7.94, -74.77, 74.77], 3e-4
    )

    concentration_correction = temperature * api.change_reference_state(
        sign=-1.0,
        temperature=temperature,
    )
    # doi:10.1021/jp810292n
    assert delta_freeenergies[2] / constants.kcal == pytest.approx(
        -6.70 + concentration_correction / constants.kcal, 8e-2
    )
    assert delta_freeenergies[0] == pytest.approx(
        -constants.R * temperature * np.log(10 ** -pK), 7e-4
    )

    k = api.get_k(model.scheme, model.compounds, temperature=temperature)
    assert -np.log10(k[0] / k[1]) == pytest.approx(pK, 5e-2)
