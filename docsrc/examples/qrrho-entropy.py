#!/usr/bin/python3

"""Validate Grimme's QRRHO treatment for vibrational entropy.

Quasi-Rigid Rotor Harmonic Oscillator (QRRHO) models attempt to improve free
energies for weakly bounded structures.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from overreact import _thermo

sns.set(style="white", palette="colorblind")

vibfreqs = np.linspace(0.0001, 350.0, num=400)
vibmoments = _thermo._gas._vibrational_moment(vibfreqs)

fig, ax = plt.subplots()
ax.plot(
    vibfreqs,
    [_thermo._gas.calc_vib_entropy(vibfreq, qrrho=False) for vibfreq in vibfreqs],
    "--",
    label="Harmonic approx.",
)
ax.plot(
    vibfreqs,
    [
        _thermo._gas.calc_rot_entropy(moments=vibmoment, independent=True)
        for vibmoment in vibmoments
    ],
    "-.",
    label="Rotational approx.",
)
ax.plot(
    vibfreqs,
    [_thermo._gas.calc_vib_entropy(vibfreq, qrrho=True) for vibfreq in vibfreqs],
    "-",
    label="Damped average",
)

ax.set_ylabel(r"Entropy [J mol$^{-1}$ K$^{-1}$]")
ax.set_xlabel(r"Mode frequency [cm$^{-1}$]")

ax.set_ylim(0, 60)
ax.set_xlim(0, 350)

ax.legend()
fig.tight_layout()
plt.show()
