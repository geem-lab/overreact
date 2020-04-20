Reaction rate constants
=======================

There are functions for calculating reaction rate constants from energy deltas.
But first we need an energy barrier:

>>> import numpy as np
>>> from scipy.constants import kilo, calorie
>>> # TODO(schneiderfelipe): transform all "delta" into "delta_freeenergy"
>>> delta = np.array([17.26, 18.86]) * kilo * calorie
>>> delta  # J/mol
array([72215.8, 78910.2])

Functions in overreact that receive energy values require values in joules per
mole. Above, a conversion from kcal per mole was done automatically. Now we can
calculate our rates:

>>> from overreact import api, core, rates, simulate
>>> k = rates.eyring(delta)
>>> k  # s-1 if first order
array([1.38, 0.093])

(Temperature is 298.15 K by default.)
Simple concurrent first order reactions using the rates above would be:

>>> import matplotlib.pyplot as plt
>>> scheme = core.parse_reactions("""
...     A -> AB‡ -> B
...     A -> AC‡ -> C
... """)
>>> dydt = simulate.get_dydt(scheme, k)
>>> t, y, r = simulate.get_y(dydt, y0=[1.0, 0.0, 0.0, 0.0, 0.0], method="Radau")
>>> plt.clf()
>>> for i, compound in enumerate(scheme.compounds):
...    if not compound.endswith("‡"):
...        plt.plot(t, y[i], label=compound)
[...]
>>> plt.legend()
<...>
>>> plt.xlabel("Time (s)")
Text(...)
>>> plt.ylabel("Concentration (M)")
Text(...)
>>> plt.savefig("docs/_static/first-order.png", transparent=True)

.. figure:: ../_static/first-order.png

   A 5-seconds simulation of two concurrent first order reactions. The reaction
   that produces C is much too slow in comparison with the one producing B.

But I don't have energy deltas
------------------------------

Don't worry. Models parsed by overreact store a matrix :math:`B` that sets the
relationship between reactants and eventually existing transition states. This
is a transformation from absolute state energies to deltas of energy relative
to the reactions in the model:

>>> scheme.B
[[-1., -1.],
 [1.,  0.],
 [0.,  0.],
 [0.,  1.],
 [0.,  0.]]
>>> api.get_delta(scheme.B, [1.92, 19.18, 2.15, 20.78, 3.40])
array([17.26, 18.86])

The returned energies can be used as above.
