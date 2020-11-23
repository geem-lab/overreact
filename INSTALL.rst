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

Optionally, extra functionality is provided such as a command-line interface
and solvent properties::

    pip install 'overreact[cli,solvents]'

This last line installs `Rich <https://github.com/willmcgugan/rich>`_
and `thermo <https://github.com/CalebBell/thermo>`_ as well.
Rich is used in the command-line interface, and thermo is used
to calculate the dynamic viscosity of solvents in the context of the
:doc:`tutorials/collins-kimball` for diffusion-limited reactions.
