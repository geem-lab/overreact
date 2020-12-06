Notes about thermochemistry
===========================

overreact employs standard statistical thermodynamical partition functions (the
Rigid Rotor Harmonic Oscillator), but also two Quasi-Rigid Rotor Harmonic
Oscillator approximations, one for entropy and one for enthalpy, for when
vibrational frequencies are too small.


A Head-Gordon damping is used for the treatment of QRRHO, which ensures the
standard procedure is used for frequencies well above
:math:`100 \text{cm}^{-1}`.

See the treatments for entropy and enthalpy in the Jupyter Notebook about
QRRHO (TODO(schneiderfelipe): add link).