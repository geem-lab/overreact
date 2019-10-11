Equilibrium constants
=====================

Calculating equilibrium constants from energy values is easy.

It's known that the stability constant of :math:`\require{mhchem}\ce{Cd(MeNH2)4^{2+}}` is around :math:`10^{6.55}`:

>>> from overreact import core, thermo, simulate
>>> import numpy as np
>>> from scipy import constants
>>> K = thermo.equilibrium_constant(-37.4 * constants.kilo)
>>> np.log10(K)
6.55

So let's check it:

>>> scheme = core.parse("""
...     Cd2p + 4 MeNH2 <=> [Cd(MeNH2)4]2p
... """)
>>> scheme.compounds, scheme.reactions
(['Cd2p', 'MeNH2', '[Cd(MeNH2)4]2p'],
 ['Cd2p + 4 MeNH2 -> [Cd(MeNH2)4]2p', '[Cd(MeNH2)4]2p -> Cd2p + 4 MeNH2'])
>>> dydt = simulate.get_dydt(scheme, [K, 1.])
>>> t, y = simulate.get_y(dydt, y0=[0., 0., 1.])
>>> y[:, -1]
array([0.01609, 0.06435 , 0.98391])
>>> Kobs = y[:, -1][2] / (y[:, -1][0] *  y[:, -1][1]**4)
>>> np.log10(Kobs)
6.55
