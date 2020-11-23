Simple chemical kinetics
========================

This tutorial addresses the most basic features of overreact in terms of
simulation. We will simulate a toy example: a simple first-order reaction with
known unitary reaction rate constant.

.. figure:: ../../_static/simple-first-order.png

   Simple first-order reaction over time.

(We won't read any logfiles nor calculate any energy, but everything given here
also applies to calculated reaction rate constants.)

We are going to simulate the following simple scheme:

.. math::

   \require{mhchem}
   \ce{A ->[k_f] B}

This reaction suggests the following set of equations:

.. math::

   \begin{align*}
     \frac{dA}{dt} &= -k_f A\\
     \frac{dB}{dt} &=  k_f A
   \end{align*}

The above model translates to the following in overreact:

>>> from overreact import api
>>> scheme = api.parse_reactions("A -> B")

(Other tutorials dwelve into the returned object, but let's forget it for now.)

:math:`k_f` is going to be set as unity (inverse seconds).

>>> kf = 1.0

overreact helps us to define the equations and solve the initial value problem.
First, let's define the system of ordinary differential equations:

>>> from overreact import simulate
>>> dydt = simulate.get_dydt(scheme, [kf])

The returned object above is a function of concentrations and time that defines
a set of ordinary differential equations in time:

.. math::

   \frac{dy}{dt} = f(t, y)

We are going to simulate 10 seconds, starting with an initial concentration of
1 molar of A (the concentration units evidently depend on the units of the
reaction rate constant).

>>> y, r = simulate.get_y(dydt, y0=[1., 0.], method="Radau")

>>> import numpy as np
>>> t = np.linspace(y.t_min, 5.0)

The simulation data is stored in `t` (points in time) and `y` (concentrations).
In order to generate the graph shown at the beginning of the present tutorial,
we do the following:

>>> import matplotlib.pyplot as plt
>>> fig, ax = plt.subplots()
>>> ax.plot(t, y(t)[0], label="A")
[...]
>>> ax.plot(t, y(t)[1], label="B")
[...]
>>> ax.legend()
<...>
>>> ax.set_xlabel("Time (s)")
Text(...)
>>> ax.set_ylabel("Concentration (M)")
Text(...)
>>> fig.savefig("docs/_static/simple-first-order.png",
...             transparent=True)

We can see that the reaction went to full completion by checking the final
concentrations:

>>> y(y.t_max)
array([0.0000, 1.0000])
