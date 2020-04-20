Transmission coefficients
=========================

Transmission coefficients (:math:`\kappa`) take quantum tunneling effects into
account in reaction rate constants.

>>> from overreact import api, tunnel
>>> from scipy import constants
>>> tunnel.wigner(266.5144)
1.069
>>> kappa = tunnel.wigner(1000.0)
>>> kappa
1.970

>>> import numpy as np
>>> delta_gibbs = api.get_delta(
...     [[0, 0, 0],
...      [-1, 0, 1],
...      [1, 0, -1],
...      [0, 0, 0],
...      [0, 0, 0]],
...     [0., 5., -10., 5., 7.5])
>>> delta_gibbs
array([-15.,   0.,  15.])

Now comes Eyring's equation

>>> T = 298.15
>>> K = np.exp(- delta_gibbs / (constants.R * T / (constants.kilo * constants.calorie)))
>>> k = kappa * (constants.k * T / constants.h) * K
>>> K, k
(array([9.887e+10, 1.000e+00, 1.011e-11]),
 array([1.210e+24, 1.224e+13, 1.238e+02]))
>>> k / kappa
array([6.143e+23, 6.212e+12, 6.283e+01])
