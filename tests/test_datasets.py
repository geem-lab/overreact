#!/usr/bin/env python3

"""Tests for module datasets."""

import os

import numpy as np
import pytest

import overreact as rx
from overreact import _datasets as datasets


def test_logfile_retrieval():
    """Ensure logfiles are properly lazily evaluated."""
    data1 = rx.io.read_logfile(
        os.path.join(
            datasets.data_path, "tanaka1996", "UMP2/6-311G(2df,2pd)", "Cl·.out",
        ),
    )
    data2 = datasets.logfiles["tanaka1996"]["Cl·@UMP2/6-311G(2df,2pd)"]
    for key in set(data1).union(data2):
        if isinstance(data1[key], str):
            assert data1[key] == data2[key]
        else:
            assert np.asarray(data1[key]) == pytest.approx(np.asarray(data2[key]))

    data1 = rx.io.read_logfile(
        os.path.join(datasets.data_path, "symmetries", "ferrocene-staggered.out"),
    )
    data2 = datasets.logfiles["symmetries"]["ferrocene-staggered"]
    for key in set(data1).union(data2):
        if isinstance(data1[key], str):
            assert data1[key] == data2[key]
        else:
            assert np.asarray(data1[key]) == pytest.approx(np.asarray(data2[key]))
