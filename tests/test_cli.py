#!/usr/bin/env python3

"""Tests for the command-line interface."""

from overreact import _cli as cli


def test_cli_compiles_source_file(monkeypatch):
    """Ensure the command-line interface can compile a source file (.k)."""
    params = ["overreact", "--compile", "data/ethane/B97-3c/model.k"]
    monkeypatch.setattr("sys.argv", params)
    cli.main()


def test_cli_describes_source_file(monkeypatch):
    """Ensure the command-line interface can describe a source file (.k)."""
    params = ["overreact", "data/ethane/B97-3c/model.k"]
    monkeypatch.setattr("sys.argv", params)
    cli.main()


def test_cli_describes_model_file(monkeypatch):
    """Ensure the command-line interface can describe a model file (.jk)."""
    params = ["overreact", "data/ethane/B97-3c/model.jk"]
    monkeypatch.setattr("sys.argv", params)
    cli.main()


def test_cli_accepts_gaussian_logfiles(monkeypatch):
    """Ensure the command-line interface is OK with Gaussian logfiles."""
    params = ["overreact", "data/acetate/Gaussian09/wB97XD/6-311++G**/model.k"]
    monkeypatch.setattr("sys.argv", params)
    cli.main()
