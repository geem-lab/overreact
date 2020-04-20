#!/usr/bin/env python3

"""Module dedicated to the time simulation of reaction models.

Here are functions that calculate reaction rates as well, which is needed for
the time simulations.
"""

import numpy as np
from scipy.integrate import solve_ivp as _solve_ivp

from overreact import core as _core


# TODO(schneiderfelipe): return dense output
# TODO(schneiderfelipe): make t_span required
def get_y(dydt, y0, t_span=(0.0, 10.0), method="Radau", num=50):
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
        If not sure, first try to run "RK45". If it makes unusually many
        iterations, diverges, or fails, your problem is likely to be stiff and
        you should use "BDF" or "Radau" (default).
    num : int, optional
        Number of time samples to generate. Must be non-negative.

    Returns
    -------
    t, y, r : array-like

    Examples
    --------
    >>> from overreact import core
    >>> scheme = core.parse_reactions("A <=> B")
    >>> t, y, r = get_y(get_dydt(scheme, [1, 1]), y0=[1, 0])
    >>> y
    array([[1.        , 0.83243215, 0.72104241, 0.64695335, 0.59768619,
            0.56497874, 0.54317395,        ..., 0.50000039, 0.50000037,
            0.50000026, 0.50000017, 0.50000011, 0.50000008, 0.50000005],
           [0.        , 0.16756785, 0.27895759, 0.35304665, 0.40231381,
            0.43502126, 0.45682605,        ..., 0.49999961, 0.49999963,
            0.49999974, 0.49999983, 0.49999989, 0.49999992, 0.49999995]])
    >>> t
    array([ 0.        ,  0.20408163,  0.40816327,  0.6122449 ,  0.81632653,
            1.02040816,  1.2244898 ,         ...,  8.7755102 ,  8.97959184,
            9.18367347,  9.3877551 ,  9.59183673,  9.79591837, 10.        ])
    >>> r
    array([[-1.00000000e+00, -6.64864300e-01, -4.42084817e-01,
            -2.93906694e-01,             ..., -3.43875637e-07,
            -2.26132930e-07, -1.50799945e-07, -1.01639971e-07],
           [ 1.00000000e+00,  6.64864300e-01,  4.42084817e-01,
             2.93906694e-01,             ...,  3.43875637e-07,
             2.26132930e-07,  1.50799945e-07,  1.01639971e-07]])
    """
    # TODO(schneiderfelipe): raise a meaningful error when y0 has the wrong shape.
    res = _solve_ivp(
        dydt,
        t_span,
        y0,
        method=method,
        t_eval=np.linspace(t_span[0], t_span[1], num=num),
    )
    return res.t, res.y, np.array([dydt(t, y) for t, y in zip(res.t, res.y.T)]).T


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
    >>> scheme = core.parse_reactions("A <=> B")
    >>> dydt = get_dydt(scheme, [1, 1])
    >>> dydt(0.0, [1., 1.])
    array([0., 0.])

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

    def _dydt(t, y, k=k_adj, A=A):
        r = k * np.prod(np.power(y, np.where(A > 0, 0, -A).T), axis=1)
        return np.dot(A, r)

    return _dydt
