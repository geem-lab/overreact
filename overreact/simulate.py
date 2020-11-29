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
    from jax import jit
    from jax.config import config

    config.update("jax_enable_x64", True)
else:
    jnp = np


def get_y(
    dydt, y0, t_span=None, method="Radau", rtol=1e-5, atol=1e-11, max_time=5 * 60
):
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
    max_time : float, optional
        If `t_span` is not given, an interval will be estimated, but it can't
        be larger than this parameter.

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
    >>> y, r = get_y(get_dydt(scheme, np.array([1, 1])), y0=[1, 0])

    The `y` object stores information about the simulation time, which can be
    used to produce a suitable vector of timepoints for, e.g., plotting:

    >>> y.t_min, y.t_max
    (0.0, 5.0)
    >>> t = np.linspace(y.t_min, y.t_max)
    >>> t
    array([..., 0.20408163, ..., 4.89795918, ...])

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
    y0 = np.asarray(y0)

    if t_span is None:
        n_halflives = 5.0  # ensure < 5% remaining material in the worst case

        halflife_estimate = 1.0
        if hasattr(dydt, "k"):
            halflife_estimate = (
                np.max(
                    [
                        np.max(y0) / 2.0,  # zeroth-order halflife
                        np.log(2.0),  # first-order halflife
                        1.0 / np.min(y0[np.nonzero(y0)]),  # second-order halflife
                    ]
                )
                / np.min(dydt.k)
            )
            logger.info(f"largest halflife guess = {halflife_estimate} s")

        t_span = [
            0.0,
            min(n_halflives * halflife_estimate, max_time),
        ]
        logger.info(f"simulation time span   = {t_span} s")

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


def get_dydt(scheme, k, ef=1e3):
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

    >>> dydt.k  # doctest: +SKIP
    array([1., 1.])

    If JAX is available, the Jacobian function will be available as
    `dydt.jac`:

    >>> dydt.jac(0.0, np.array([1., 1.]))  # doctest: +SKIP
    DeviceArray([[-1.,  1.],
                 [ 1., -1.]], dtype=float64)

    """
    scheme = _core._check_scheme(scheme)
    A = jnp.asarray(scheme.A)
    M = jnp.where(A > 0, 0, -A).T
    k_adj = _adjust_k(scheme, k, ef=ef)

    def _dydt(t, y):
        r = k * jnp.prod(jnp.power(y, M), axis=1)
        return jnp.dot(A, r)

    if _found_jax:
        _dydt = jit(_dydt)

        def _jac(t, y):
            # _jac(t, y)[i, j] == d f_i / d y_j
            # shape is (n_compounds, n_compounds)
            res = jacfwd(lambda _y: _dydt(t, _y))(y)
            return res

        _dydt.jac = _jac

    _dydt.k = k_adj
    return _dydt


def _adjust_k(scheme, k, ef=1e3):
    """Adjust reaction rate constants so that equilibria are equilibria.

    Parameters
    ----------
    scheme : Scheme
    k : array-like
        Reaction rate constant(s). Units match the concentration units given to
        the returned function ``dydt``.
    ef : float, optional

    Returns
    -------
    k : array-like
        Adjusted constants.

    Examples
    --------
    >>> from overreact import api, core

    >>> scheme = core.parse_reactions("A <=> B")
    >>> _adjust_k(scheme, [1, 1])  # doctest: +SKIP
    array([1., 1.])

    >>> model = api.parse_model("data/ethane/B97-3c/model.k")
    >>> _adjust_k(model.scheme,
    ...           api.get_k(model.scheme, model.compounds))  # doctest: +SKIP
    array([8.15810511e+10])

    >>> model = api.parse_model("data/acetate/model.k")
    >>> _adjust_k(model.scheme, api.get_k(model.scheme, model.compounds))  # doctest: +SKIP
    array([1.00000000e+00, 3.43865350e+04, 6.58693442e+05,
           1.00000000e+00, 6.36388893e+54, 1.00000000e+00])

    >>> model = api.parse_model(
    ...     "data/perez-soto2020/RI/BLYP-D4/def2-TZVP/model.k"
    ... )  # doctest: +SKIP
    >>> _adjust_k(model.scheme,
    ...           api.get_k(model.scheme, model.compounds))  # doctest: +SKIP
    array([1.02300196e+11, 3.08436461e+15, 1.02300196e+11, 1.25048767e+20,
           2.50281559e+12, 3.08378146e+19, 2.50281559e+12, 2.49786052e+22,
           2.50281559e+12, 6.76606575e+18, 2.99483252e-08, 1.31433415e-09,
           3.20122447e+01, 5.43065970e+01, 3.36730955e+03, 2.06802748e+04,
           1.63458719e+04, 1.02300196e+08, 3.92788711e+12, 1.02300196e+11,
           2.65574047e+17, 2.50281559e+12, 2.00892034e+14, 1.02300196e+11,
           8.69343596e+17, 2.50281559e+12, 3.31477037e+15, 1.02300196e+11])

    """
    scheme = _core._check_scheme(scheme)
    is_half_equilibrium = np.asarray(scheme.is_half_equilibrium)
    k = np.asarray(k).copy()

    # TODO(schneiderfelipe): this test for equilibria should go to get_k since
    # equilibria must obey the Collins-Kimball maximum reaction rate rule as
    # well.

    if np.any(is_half_equilibrium):
        # at least one equilibrium
        if np.any(~is_half_equilibrium):
            # at least one true reaction

            k_slowest_equil = k[is_half_equilibrium].min()
            k_fastest_react = k[~is_half_equilibrium].max()
            adjustment = ef * (k_fastest_react / k_slowest_equil)

            k[is_half_equilibrium] *= adjustment
            logger.warning(f"equilibria adjustment = {adjustment}")
        else:
            # only equilibria

            # set the smallest one to be equal to one
            k = k / k.min()
    # else:
    #     # only zero or more true reactions (no equilibria)
    #     pass

    return jnp.asarray(k)
