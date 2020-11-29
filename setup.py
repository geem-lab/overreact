#!/usr/bin/env python3

"""Setup script for overreact."""

from setuptools import find_packages
from setuptools import setup

setup(
    name="overreact",
    version=open("VERSION").read().strip(),
    description="Construct precise chemical microkinetic models from first principles",
    author="Felipe Silveira de Souza Schneider",
    author_email="schneider.felipe@posgrad.ufsc.br",
    url="https://github.com/schneiderfelipe/overreact",
    packages=find_packages(),
    entry_points={"console_scripts": ["overreact = overreact.cli:main"]},
    install_requires=["cclib>=1.6.3", "scipy>=1.4.0"],
    extras_require={
        "cli": ["rich>=9.2.0"],
        "fast": ["jax>=0.2.6", "jaxlib>=0.1.57"],
        "solvents": ["thermo>=0.1.39"],
    },
    tests_require=["matplotlib>=2.1.1", "pytest>=5.2.1"],
)
