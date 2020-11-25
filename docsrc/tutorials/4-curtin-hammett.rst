Curtin-Hammett Principle
========================

Let's do a toy example that addresses the Curtin-Hammett principle:

>>> from overreact import core, api, rates, simulate
>>> from overreact import constants
>>> scheme = core.parse_reactions("""
...     I1 <=> I2
...     I1 -> T1‡ -> P1
...     I2 -> T2‡ -> P2
... """)

The selectivity is defined as the ratio between both products:

.. math::
   \require{mhchem}
   S = \frac{c_{\ce{P_1}}}{c_{\ce{P_2}}}

>>> scheme.compounds
['I1', 'I2', 'T1‡', 'P1', 'T2‡', 'P2']
>>> freeenergy = [0.0, -0.5, 10.0, 1.0, 11.0, 1.0]
>>> k = rates.eyring(constants.kcal * api.get_delta(scheme.B, freeenergy))
>>> k
array([1.445e+13, 6.212e+12, 2.905e+05, 2.310e+04])

>>> y, r = simulate.get_y(simulate.get_dydt(scheme, k), y0=[1.0, 0.0, 0.0, 0.0, 0.0, 0.0], method="BDF")
>>> S = y(y.t_max)[scheme.compounds.index("P1")] / y(y.t_max)[scheme.compounds.index("P2")]
>>> S
5.408
