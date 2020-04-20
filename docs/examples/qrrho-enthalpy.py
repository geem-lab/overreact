#!/usr/bin/python3

"""Validate Grimme's QRRHO treatment for vibrational enthalpy.

Quasi-Rigid Rotor Harmonic Oscillator (QRRHO) models attempt to improve free
energies for weakly bounded structures.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from overreact import _thermo
from overreact import constants

sns.set(style="white", palette="colorblind")

vibfreqs = np.linspace(0.0001, 400.0, num=400)
vibmoments = _thermo._gas._vibrational_moment(vibfreqs)

fig, ax = plt.subplots()
ax.plot(
    vibfreqs,
    [
        _thermo._gas.calc_vib_energy(vibfreq, qrrho=False) / constants.kcal
        for vibfreq in vibfreqs
    ],
    "-",
    label="RRHO",
)
ax.plot(
    vibfreqs,
    [
        _thermo._gas.calc_vib_energy(vibfreq, qrrho=True) / constants.kcal
        for vibfreq in vibfreqs
    ],
    "-",
    label="Quasi-RRHO",
)

inset = fig.add_axes([0.6, 0.35, 0.25, 0.25])
weights = _thermo._gas._head_gordon_damping(vibfreqs)
inset.plot(vibfreqs, weights)

ax.set_ylabel("V$_i$ [kcal/mol]")
ax.set_xlabel(r"$\nu_i$ [cm$^{-1}$]")
inset.set_ylabel(r"$\omega$")
inset.set_xlabel(r"$\nu_i$ [cm$^{-1}$]")

ax.set_ylim(0.2, 0.8)
ax.set_xlim(0, 400)
inset.set_ylim(0.0, 1.0)
inset.set_xlim(0, 400)

ax.legend()
fig.tight_layout()
plt.show()
