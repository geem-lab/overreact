Quickstart
==========

Here is an overview of overreact's capabilities. overreact allows you to build
any thinkable reaction model:

>>> from overreact import api
>>> scheme = api.parse_reactions("S -> E‡ -> S")
>>> scheme
Scheme(compounds=['S', 'E‡'],
       reactions=['S -> S'],
       is_half_equilibrium=[False],
       ...)

The "‡" symbol is used to indicate transition states (but the "#" symbol is
also accepted). Many different reactions can be specified at the same time by
properly giving a list. Equilibria are recognized as having "<=>". Reactions
preserve the order they appeared in the input.

Similarly, compound data is retrieved from logfiles using `parse_compounds`:

>>> compounds = api.parse_compounds({
...     "S": "data/ethane/B97-3c/staggered.out",
...     "E‡": "data/ethane/B97-3c/eclipsed.out"
... })
>>> compounds
{'S': {'logfile': 'data/ethane/B97-3c/staggered.out',
       'energy': -209483812.77142256,
       ...},
 'E‡': {'logfile': 'data/ethane/B97-3c/eclipsed.out',
        'energy': -209472585.3539883,
        ...}}

After both two line above, we can start analyzing our complete model:

>>> api.get_k(scheme, compounds)
array([8.15810511e+10])

Even with a rather simple level of theory (B97-3c :cite:`Brandenburg_2018`),
this result compares well with the experimentally determined value
(:math:`8.3 \times 10^{10} \text{s}^{-1}` :cite:`Zheng_2006`).
This small error of  is due to the very accurate thermochemical and tunneling
corrections employed.

The line above works by calculating internal energies, enthalpies and entropies
for each compound, but you can do this in separate lines as well. In fact, in
any temperature:

>>> api.get_internal_energies(compounds)  # 298.15 K by default
array([-2.09280338e+08, -2.09271131e+08])
>>> api.get_internal_energies(compounds, temperature=400.0)
array([-2.09275396e+08, -2.09266995e+08])

Values are always in joules per mole and honor the original order of compounds,
as they were initially given. The same thing can be done for enthalpies and
entropies (in joules per mole per kelvin):

>>> temperature = 300.0
>>> enthalpies = api.get_enthalpies(compounds, temperature=temperature)
>>> enthalpies
array([-2.092778e+08, -2.092686e+08])
>>> entropies = api.get_entropies(compounds, temperature=temperature)
>>> entropies
array([227.9, 221.9])

Now free energies are easy, we just use full power of Numpy arrays:

>>> freeenergies = enthalpies - temperature * entropies
>>> freeenergies - freeenergies.min()
array([    0.        , 11009.33770153])

In the above, we calculated free energies relative to the minimum.
