Notes about thermochemistry
===========================

overreact employs standard statistical thermodynamical partition functions (the
Rigid Rotor Harmonic Oscillator), but also two Quasi-Rigid Rotor Harmonic
Oscillator approximations, one for entropy and one for enthalpy, for when
vibrational frequencies are too small.


A Head-Gordon damping is used for the treatment of QRRHO, which ensures the
standard procedure is used for frequencies well above
:math:`100 \text{cm}^{-1}`.

Below is the treatment for entropy:

.. plot:: examples/qrrho-entropy.py
   :include-source:

Now, the treatment for enthalpy (the inset consists of the damping function
itself):

.. plot:: examples/qrrho-enthalpy.py
   :include-source:
