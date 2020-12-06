#!/usr/bin/python3

"""Calculate the degree of rate control for a reaction system."""

import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from overreact import api
from overreact import datasets

sns.set(style="white", palette="colorblind")

temperature = 298.15

basisset = "6-311G(2df,2pd)"  # 6-311G(2df,2p) best predicts activation enthalpy
if len(sys.argv) > 1:
    basisset = sys.argv[1]

model = api.parse_model(
    os.path.join(datasets.data_path, f"tanaka1996/UMP2/{basisset}/model.k")
)

y0_CH4 = 772e-3 / (np.sum(model.compounds["CH4"].atommasses) * 1e3)
y0_Cl = 1 / (np.sum(model.compounds["Cl·"].atommasses) * 1e3)
y0_HCl = 1 / (np.sum(model.compounds["HCl"].atommasses) * 1e3)
y0 = [y0_CH4, y0_Cl, 0.0, 0.0, y0_HCl]
print(y0)

t, drc = api.get_drc(
    model.scheme,
    model.compounds,
    y0,
    scale="M-1 s-1",
    num=500,
    dx=2000.0,
    order=5,
)
print(drc)

fig, ax = plt.subplots()
for i, name in enumerate(model.scheme.compounds):
    if not api.is_transition_state(name):
        ax.plot(1e3 * t, drc[i], label=f"{name}")

ax.set_ylabel("DRC")
ax.set_xlabel("Time [ms]")

ax.legend()
fig.tight_layout()
plt.show()
