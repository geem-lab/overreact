#!/usr/bin/env python3

"""Module dedicated to the time simulation of reaction models.

Here are functions that calculate reaction rates as well, which is needed for
the time simulations.
"""

import numpy as _np
from scipy.integrate import solve_ivp as _solve_ivp


def get_y(dydt, y0, t_span=(0.0, 10.0), method="RK45"):
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
        integrates until it reaches t=tf.
    method : str, optional
        Integration method to use. See `scipy.integrade.solve_ivp` for details.
        If not sure, first try to run "RK45" (default). If it makes unusually
        many iterations, diverges, or fails, your problem is likely to be stiff
        and you should use "BDF" or "Radau".

    Returns
    -------
    t, y : array-like

    Examples
    --------
    >>> from overreact import core
    >>> scheme = core.parse("A <=> B")
    >>> t, y = get_y(get_dydt(scheme, [1, 1]), y0=[1, 0])
    >>> y
    array([[1.000, 0.999, 0.989, 0.901, 0.695, 0.580, 0.528, 0.508, 0.502,
            0.500, 0.500, 0.500, 0.501, 0.500],
           [0.000, 0.001, 0.011, 0.099, 0.305, 0.420, 0.472, 0.492, 0.498,
            0.500, 0.500, 0.500, 0.499, 0.500]])
    >>> t
    array([0.00000000e+00, 9.99000500e-04, 1.09890055e-02, ...,
           5.08550239e+00, 6.82280908e+00, 8.99291649e+00, 1.00000000e+01])

    """
    res = _solve_ivp(dydt, t_span, y0, method)
    return res.t, res.y


def get_dydt(scheme, k, ef=1.0e3):
    """Generate a rate function that models a reaction scheme.

    Parameters
    ----------
    scheme : Scheme
    k : array-like
    ef : float, optional

    Returns
    -------
    dydt : callable
        Reaction rate function.

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
    >>> from overreact import core
    >>> scheme = core.parse("A <=> B")
    >>> dydt = get_dydt(scheme, [1, 1])
    >>> dydt(0.0, [1., 1.])
    array([0., 0.])

    """
    k_adj = _np.asanyarray(k).copy()

    # TODO(schneiderfelipe): if there's only equilibria, I want the smallest
    # one to be equal to one!
    if _np.any(scheme.is_half_equilibrium) and _np.any(~scheme.is_half_equilibrium):
        k_adj[scheme.is_half_equilibrium] *= ef * (
            k_adj[~scheme.is_half_equilibrium].max()
            / k_adj[scheme.is_half_equilibrium].min()
        )

    def _dydt(t, y, k=k_adj, A=scheme.A):
        r = k * _np.prod(_np.power(y, _np.where(A > 0, 0, -A).T), axis=1)
        return _np.dot(A, r)

    return _dydt
