#!/usr/bin/env python3

"""Module dedicated to the time simulation of reaction models.

Here are functions that calculate reaction rates as well, which is needed for
the time simulations.
"""

import logging

import numpy as np
from scipy.integrate import solve_ivp as _solve_ivp

from overreact import core as _core
from overreact import misc as _misc

logger = logging.getLogger(__name__)

_found_jax = _misc._find_package("jax")
if _found_jax:
    import jax.numpy as jnp
    from jax import jacfwd
    from jax.config import config

    config.update("jax_enable_x64", True)
else:
    jnp = np


def get_y(dydt, y0, t_span=None, method="Radau", rtol=1e-5, atol=1e-9):
    """Simulate a reaction scheme from its rate function.

    This uses scipy's ``solve_ivp`` under the hood.

    Parameters
    ----------
    dydt : callable
        Right-hand side of the system.
    y0 : array-like
        Initial state.
    t_span : array-like, optional
        Interval of integration (t0, tf). The solver starts with t=t0 and
        integrates until it reaches t=tf. If not given, a conservative value
        is chosen based on the system at hand (the method of choice works for
        any zeroth-, first- or second-order reactions).
    method : str, optional
        Integration method to use. See `scipy.integrade.solve_ivp` for details.
        Kinetics problems are very often stiff and, as such, "RK45" is
        normally unsuited. "Radau", "BDF" or "LSODA" are good choices.
    rtol, atol : array-like
        See `scipy.integrade.solve_ivp` for details.

    Returns
    -------
    y, r : callable
        Concentrations and reaction rates as functions of time. The y object
        is an OdeSolution and stores attributes t_min and t_max.


    Examples
    --------
    >>> import numpy as np
    >>> from overreact import core

    A toy simulation can be performed in just two lines:

    >>> scheme = core.parse_reactions("A <=> B")
    >>> y, r = get_y(get_dydt(scheme, [1, 1]), y0=[1, 0])

    The `y` object stores information about the simulation time, which can be
    used to produce a suitable vector of timepoints for, e.g., plotting:

    >>> y.t_min, y.t_max
    (0.0, 10.0)
    >>> t = np.linspace(y.t_min, y.t_max)
    >>> t
    array([ 0.        ,  0.20408163,  ...,  9.79591837, 10.        ])

    Both `y` and `r` can be used to check concentrations and rates in any
    point in time. In particular, both are vectorized:

    >>> y(t)  # doctest: +SKIP
    array([[1.        , 0.83244929, ..., 0.49999842, 0.49999888],
           [0.        , 0.16755071, ..., 0.50000158, 0.50000112]])
    >>> r(t)  # doctest: +SKIP
    array([[-1.00000000e+00, ..., -1.01639971e-07],
           [ 1.00000000e+00, ...,  1.01639971e-07]])
    """
    # TODO(schneiderfelipe): raise a meaningful error when y0 has the wrong shape.
    y0 = np.asanyarray(y0)

    if t_span is None:
        n_halflives = 10.0

        halflife_estimate = 1.0
        if hasattr(dydt, "k"):
            halflife_estimate = (
                np.max(
                    [
                        np.max(y0) / 2.0,  # zeroth-order half-life
                        np.log(2.0),  # first-order half-life
                        1.0 / np.min(y0[np.nonzero(y0)]),  # second-order half-life
                    ]
                )
                / np.min(dydt.k)
            )

        t_span = [
            0.0,
            n_halflives * halflife_estimate,
        ]
        logger.info(f"simulation time span = {t_span} s")

    jac = None
    if hasattr(dydt, "jac"):
        jac = dydt.jac

    # TODO(schneiderfelipe): log solve_ivp stuff.
    res = _solve_ivp(
        dydt,
        t_span,
        y0,
        method=method,
        dense_output=True,
        rtol=rtol,
        atol=atol,
        jac=jac,
    )
    y = res.sol

    def r(t):
        # TODO(schneiderfelipe): this is probably not the best way to
        # vectorize a function!
        try:
            return np.array([dydt(_t, _y) for _t, _y in zip(t, y(t).T)]).T
        except TypeError:
            return dydt(t, y(t))

    # TODO(schneiderfelipe): use a flag such as full_output to indicate we
    # want everything, not just y.
    return y, r


def get_dydt(scheme, k, ef=1.0e3):
    """Generate a rate function that models a reaction scheme.

    Parameters
    ----------
    scheme : Scheme
    k : array-like
        Reaction rate constant(s). Units match the concentration units given to
        the returned function ``dydt``.
    ef : float, optional

    Returns
    -------
    dydt : callable
        Reaction rate function. The actual reaction rate constants employed
        are stored in the attribute `k` of the returned function. If JAX is
        available, the attribute `jac` will hold the Jacobian function of
        `dydt`.

    Warns
    -----
    RuntimeWarning
        If the slowest half equilibrium is slower than the fastest non half
        equilibrium.

    Notes
    -----
    The returned function is suited to be used by ODE solvers such as
    `scipy.integrate.solve_ivp` or the older `scipy.integrate.ode` (see
    examples below). This is actually what the function `get_y` from the
    current module does.

    Examples
    --------
    >>> import numpy as np
    >>> from overreact import core
    >>> scheme = core.parse_reactions("A <=> B")
    >>> dydt = get_dydt(scheme, [1, 1])
    >>> dydt(0.0, np.array([1., 1.]))  # doctest: +SKIP
    array([0., 0.])

    If available, JAX is used for JIT compilation. This will make `dydt`
    complain if given lists instead of numpy arrays. So stick to the safer,
    faster side as above.

    The actually used reaction rate constants can be inspected with the `k`
    attribute of `dydt`:

    >>> dydt.k
    array([1, 1])

    If JAX is available, the Jacobian function will be available as
    `dydt.jac`:

    >>> dydt.jac(0.0, np.array([1., 1.]))  # doctest: +SKIP
    DeviceArray([[-1.,  1.],
                 [ 1., -1.]], dtype=float64)

    """
    scheme = _core._check_scheme(scheme)
    is_half_equilibrium = np.asanyarray(scheme.is_half_equilibrium)
    k_adj = np.asanyarray(k).copy()
    A = np.asanyarray(scheme.A)

    # TODO(schneiderfelipe): this test for equilibria should go to get_k since
    # equilibria must obey the Collins-Kimball maximum reaction rate rule as
    # well.
    # TODO(schneiderfelipe): check whether we should filter RuntimeWarning.
    # TODO(schneiderfelipe): if there's only equilibria, should I want the
    # smallest one to be equal to one?
    if np.any(is_half_equilibrium) and np.any(~is_half_equilibrium):
        # TODO(schneiderfelipe): test those conditions
        k_adj[is_half_equilibrium] *= ef * (
            k_adj[~is_half_equilibrium].max() / k_adj[is_half_equilibrium].min()
        )

    M = np.where(A > 0, 0, -A).T

    def _dydt(t, y, k=k_adj, M=M):
        r = k * jnp.prod(jnp.power(y, M), axis=1)
        return jnp.dot(A, r)

    if _found_jax:
        _dydt = _dydt

        def _jac(t, y, k=k_adj, M=M):
            # _jac(t, y)[i, j] == d f_i / d y_j
            # shape is (n_compounds, n_compounds)
            return jacfwd(lambda _y: _dydt(t, _y, k, M))(y)

        _dydt.jac = _jac

    _dydt.k = k_adj
    return _dydt
