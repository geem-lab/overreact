Hydrogen abstraction of methane by chlorine atoms
=================================================

We are going to compare the following atmospheric reaction with experimental
results:

.. math::
   \require{mhchem}
   \ce{Cl + CH4 -> HCl + CH3}

NASA reports an accepted value of
:math:`1.0 \times 10^{-13} \text{cm}^3 \text{molecule}^{-1} \text{s}^{-1}`
:cite:`Burkholder_2015`. In fact the following is a plot of the accepted
Arrhenius plot of their accepted regression, together with uncertainty bands:

.. plot:: examples/hydrogen-abstraction-arrhenius.py
   :include-source:

Microkinetic simulation
-----------------------

.. plot:: examples/hydrogen-abstraction-microkinetics.py
   :include-source:
