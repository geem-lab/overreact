#!/usr/bin/python3

"""Simulate a microkinetics system using calculated reaction rate constants."""

import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from overreact import api

sns.set(style="white", palette="colorblind")

temperature = 298.15

basisset = "6-311G(2df,2pd)"  # 6-311G(2df,2p) best predicts activation enthalpy
if len(sys.argv) > 1:
    basisset = sys.argv[1]

model = api.parse_model(
    os.path.join(api.data_path, f"tanaka1996/UMP2/{basisset}/model.k")
)
k_eck = api.get_k(
    model.scheme, model.compounds, temperature=temperature, scale="M-1 s-1"
)

y0_CH4 = 772e-3 / (np.sum(model.compounds["CH4"].atommasses) * 1e3)
y0_Cl = 1 / (np.sum(model.compounds["Cl·"].atommasses) * 1e3)
y0_HCl = 1 / (np.sum(model.compounds["HCl"].atommasses) * 1e3)
y0 = [y0_CH4, y0_Cl, 0.0, 0.0, y0_HCl]
print(y0)

dydt = api.get_dydt(model.scheme, k_eck)
y, r = api.get_y(dydt, y0=y0, method="Radau")

print(model.scheme.compounds)
print(y(y.t_max))

t = np.linspace(y.t_min, 5e-3)

fig, ax = plt.subplots()
for i, name in enumerate(model.scheme.compounds):
    if not api.is_transition_state(name):
        ax.plot(1e3 * t, 1e3 * y(t)[i], label=f"{name}")

ax.set_ylabel("Concentration [mM]")
ax.set_xlabel("Time [ms]")

ax.legend()
fig.tight_layout()
plt.show()
