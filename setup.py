#!/usr/bin/env python3

"""Setup script for overreact."""

from setuptools import find_packages
from setuptools import setup

setup(
    name="overreact",
    version="0.1",
    description="A Python package for constructing microkinetic models",
    author="Felipe Silveira de Souza Schneider",
    author_email="schneider.felipe@posgrad.ufsc.br",
    url="https://github.com/schneiderfelipe/overreact",
    packages=find_packages(),
    install_requires=["cclib>=1.6.2", "scipy>=0.19.1"],
    extras_require={"thermo": ["thermo>=0.1.39"]},
    tests_require=["pytest>=5.2.1"],
)
