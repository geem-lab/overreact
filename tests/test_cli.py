"""Tests for the command-line interface."""

from __future__ import annotations

import numpy as np
import pytest

import overreact as rx
from overreact import _cli as cli


def test_cli_compiles_source_file(monkeypatch) -> None:
    """Ensure the command-line interface can compile a source file (`.k`)."""
    params = ["overreact", "--compile", "data/ethane/B97-3c/model.k"]
    monkeypatch.setattr("sys.argv", params)
    cli.main()


def test_cli_describes_source_file(monkeypatch) -> None:
    """Ensure the command-line interface can describe a source file (`.k`)."""
    params = ["overreact", "data/ethane/B97-3c/model.k"]
    monkeypatch.setattr("sys.argv", params)
    cli.main()


def test_cli_describes_model_file(monkeypatch) -> None:
    """Ensure the command-line interface can describe a model file (`.jk`)."""
    params = ["overreact", "data/ethane/B97-3c/model.jk"]
    monkeypatch.setattr("sys.argv", params)
    cli.main()


def test_cli_accepts_gaussian_logfiles(monkeypatch) -> None:
    """Ensure the command-line interface is OK with Gaussian logfiles."""
    params = ["overreact", "data/acetate/Gaussian09/wB97XD/6-311++G**/model.k"]
    monkeypatch.setattr("sys.argv", params)
    cli.main()


@pytest.mark.parametrize("concentrations", [["A‡:1.0"], ["C:1.0"], ["C:!1.0"]])
def test_prepare_simulation_rejects_unknown_compounds(concentrations) -> None:
    """Ensure concentrations are given only for compounds of the scheme."""
    with pytest.raises(ValueError, match="not a compound"):
        cli._prepare_simulation(
            rx.Scheme.parse("A -> A‡ -> B"),
            np.ones(1),
            concentrations,
        )
