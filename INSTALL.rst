Installation
============

overreact depends on:

- `cclib <https://github.com/cclib/cclib/>`_ (parser for computational
  chemistry logfiles).
- `SciPy <https://github.com/scipy/scipy/>`_ (numerical integration,
  optimization, unit conversion and others).

The package, together with the above dependencies, can be installed from
`PyPI <https://pypi.org/project/overreact/>`_, using
`pip <https://pypi.org/project/pip/>`_, with the following command::

   pip install overreact

Optionally, extra functionality is provided by
`thermo <https://github.com/CalebBell/thermo>`_::

    pip install 'overreact[thermo]'

This last line installs thermo  as well. thermo is used to calculate the
dynamic viscosity of solvents in the context of the
:doc:`tutorials/collins-kimball` for diffusion-limited reactions.
