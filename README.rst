|build|_

.. |build| image:: https://github.com/schneiderfelipe/overreact/workflows/build/badge.svg
.. _build: https://github.com/schneiderfelipe/overreact/actions?query=workflow:build

overreact
=========

.. after-title

**overreact** builds homogeneous microkinetic models from first-principles calculations.

>>> from overreact import api
>>> api.get_k("S -> E‡ -> S", {"S": "data/ethane/B97-3c/staggered.out",
...                            "E‡": "data/ethane/B97-3c/eclipsed.out"})
array([8.15810511e+10])

The two lines above calculate the rate of methyl rotation in ethane at
`B97-3c <https://doi.org/10.1063/1.5012601>`__.
(By the way, the error is around 1.7% when compared to the
`experiment <http://dx.doi.org/10.1126/science.1132178>`__.)

overreact uses precise thermochemical partition funtions and tunneling
corrections.
See the
`installation instructions <https://schneiderfelipe.github.io/overreact/installation.html>`__
and a `short introduction <https://schneiderfelipe.github.io/overreact/quickstart.html>`__.
