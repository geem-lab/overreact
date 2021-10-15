#!/usr/bin/env python3

"""Module dedicated to the time simulation of reaction models.

Here are functions that calculate reaction rates as well, which is needed for
the time simulations.
"""

import logging

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import minimize_scalar

import overreact as rx
from overreact import constants
from overreact.misc import _found_jax

if _found_jax:
    import jax.numpy as jnp
    from jax import jacfwd
    from jax import jit
    from jax.config import config

    config.update("jax_enable_x64", True)
else:
    jnp = np

logger = logging.getLogger(__name__)


# TODO(schneiderfelipe): allow y0 to be a dict-like object.
def get_y(
    dydt, y0, t_span=None, method="Radau", rtol=1e-5, atol=1e-11, max_time=24 * 60 * 60
):
    """Simulate a reaction scheme from its rate function.

    This function provides two functions that calculate the concentrations and
    the rates of formation at any point in time for any compound. It does that
    by solving an initial value problem (IVP) through scipy's ``solve_ivp``
    under the hood.

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
        Integration method to use. See `scipy.integrate.solve_ivp` for details.
        Kinetics problems are very often stiff and, as such, "RK45" is
        normally unsuited. "Radau", "BDF" or "LSODA" are good choices.
    rtol, atol : array-like
        See `scipy.integrate.solve_ivp` for details.
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
    >>> import overreact as rx

    A toy simulation can be performed in just two lines:

    >>> scheme = rx.parse_reactions("A <=> B")
    >>> y, r = get_y(get_dydt(scheme, np.array([1, 1])), y0=[1, 0])

    The `y` object stores information about the simulation time, which can be
    used to produce a suitable vector of timepoints for, e.g., plotting:

    >>> y.t_min, y.t_max
    (0.0, 14.0)
    >>> t = np.linspace(y.t_min, y.t_max)
    >>> t
    array([ 0. , 0.28571429, ..., 13.71428571, 14. ])

    Both `y` and `r` can be used to check concentrations and rates in any
    point in time. In particular, both are vectorized:

    >>> y(t)
    array([[1. , ..., 0.5 ]])
    >>> r(t)  # doctest: +SKIP
    array([[-1.00000000e+00, ..., -1.39544265e-10],
           [ 1.00000000e+00, ...,  1.39544265e-10]])
    """
    # TODO(schneiderfelipe): raise a meaningful error when y0 has the wrong shape.
    y0 = np.asarray(y0)

    if t_span is None:
        # We defined alpha such that 1.0 - alpha is an estimate of the extend
        # to which the reaction is simulated.
        alpha = 1e-4
        n_halflives = np.ceil(-np.log(alpha) / np.log(2))

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
    res = solve_ivp(
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

    return y, r


def get_dydt(scheme, k, ef=1e4):
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
    >>> import overreact as rx

    >>> scheme = rx.parse_reactions("A <=> B")
    >>> dydt = get_dydt(scheme, np.array([1, 1]))
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
    scheme = rx.core._check_scheme(scheme)
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
            return jacfwd(lambda _y: _dydt(t, _y))(y)

        _dydt.jac = _jac

    _dydt.k = k_adj
    return _dydt


def _adjust_k(scheme, k, ef=1e4):
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
    >>> import overreact as rx

    >>> scheme = rx.parse_reactions("A <=> B")
    >>> _adjust_k(scheme, [1, 1])  # doctest: +SKIP
    array([1., 1.])

    >>> model = rx.parse_model("data/ethane/B97-3c/model.k")
    >>> _adjust_k(model.scheme,
    ...           rx.get_k(model.scheme, model.compounds))  # doctest: +SKIP
    array([8.16880917e+10])

    >>> model = rx.parse_model("data/acetate/model.k")
    >>> _adjust_k(model.scheme,
    ...           rx.get_k(model.scheme, model.compounds))  # doctest: +SKIP
    array([1.00000000e+00, 5.74491548e+04, 1.61152010e+07,
           1.00000000e+00, 1.55695112e+56, 1.00000000e+00])

    >>> model = rx.parse_model(
    ...     "data/perez-soto2020/RI/BLYP-D4/def2-TZVP/model.k"
    ... )
    >>> _adjust_k(model.scheme,
    ...           rx.get_k(model.scheme, model.compounds))  # doctest: +SKIP
    array([1.02320357e+12, ..., 1.02320357e+12])

    """
    scheme = rx.core._check_scheme(scheme)
    is_half_equilibrium = np.asarray(scheme.is_half_equilibrium)
    k = np.asarray(k, dtype=float).copy()

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


def get_fixed_scheme(scheme, k, fixed_y0):
    """Generate an alternative scheme with some concentrations fixed.

    This function returns data that allow the microkinetic simulation of a
    reaction network under constraints, namely when some compounds have fixed
    concentrations. This works by 1. removing all references to the fixed
    compounds and by 2. properly multiplying the reaction rate constants by
    the respective concentrations.

    Parameters
    ----------
    scheme : Scheme
    k : array-like
        Reaction rate constant(s). Units match the concentration units given to
        the returned function ``dydt``.
    fixed_y0 : dict-like
        Fixed initial state. Units match the concentration units given to
        the returned function ``dydt``.

    Returns
    -------
    scheme : Scheme
        Associated reaction scheme with all references to fixed compounds
        removed.
    k : array-like
        Associated (effective) reaction rate constants that model the fixed
        concentrations.

    Notes
    -----
    Keep in mind that when a compound get its concentration fixed, the
    reaction scheme no longer conserves matter. You can think of it as
    reacting close to an infinite source of the compound, but it accumulates
    in the milleu at the given concentration.

    Examples
    --------
    >>> import numpy as np
    >>> import overreact as rx

    Equilibria under a specific pH can be easily modeled:

    >>> pH = 7
    >>> scheme = rx.parse_reactions("AH <=> A- + H+")
    >>> k = np.array([1, 1])
    >>> scheme, k = rx.get_fixed_scheme(scheme, k, {"H+": 10**-pH})
    >>> scheme
    Scheme(compounds=('AH', 'A-'),
           reactions=('AH -> A-',
                      'A- -> AH'),
           is_half_equilibrium=(True, True),
           A=((-1.0, 1.0),
              (1.0, -1.0)),
           B=((-1.0, 0.0),
              (1.0, 0.0)))
    >>> k
    array([1.e+00, 1.e-07])

    It is also possible to model the fixed activity of a solvent, for
    instance:

    >>> scheme = rx.parse_reactions("A + 2H2O -> B")
    >>> k = np.array([1.0])
    >>> scheme, k = rx.get_fixed_scheme(scheme, k, {"H2O": 55.6})
    >>> scheme
    Scheme(compounds=('A', 'B'),
           reactions=('A -> B',),
           is_half_equilibrium=(False,),
           A=((-1.0,),
              (1.0,)),
           B=((-1.0,),
              (1.0,)))
    >>> k
    array([3091.36])

    Multiple reactions work fine, see both examples below:

    >>> pH = 12
    >>> scheme = rx.parse_reactions("B <- AH <=> A- + H+")
    >>> k = np.array([10.0, 1, 1])
    >>> scheme, k = rx.get_fixed_scheme(scheme, k, {"H+": 10**-pH})
    >>> scheme
    Scheme(compounds=('AH', 'B', 'A-'),
           reactions=('AH -> B',
                      'AH -> A-',
                      'A- -> AH'),
           is_half_equilibrium=(False, True, True),
           A=((-1.0, -1.0, 1.0),
              (1.0, 0.0, 0.0),
              (0.0, 1.0, -1.0)),
           B=((-1.0, -1.0, 0.0),
              (1.0, 0.0, 0.0),
              (0.0, 1.0, 0.0)))
    >>> k
    array([1.e+01, 1.e+00, 1.e-12])

    >>> pH = 2
    >>> scheme = rx.parse_reactions(["AH <=> A- + H+", "B- + H+ <=> BH"])
    >>> k = np.array([1, 1, 2, 2])
    >>> scheme, k = rx.get_fixed_scheme(scheme, k, {"H+": 10**-pH})
    >>> scheme
    Scheme(compounds=('AH', 'A-', 'B-', 'BH'),
           reactions=('AH -> A-',
                      'A- -> AH',
                      'B- -> BH',
                      'BH -> B-'),
           is_half_equilibrium=(True, True, True, True),
           A=((-1.0, 1.0, 0.0, 0.0),
              (1.0, -1.0, 0.0, 0.0),
              (0.0, 0.0, -1.0, 1.0),
              (0.0, 0.0, 1.0, -1.0)),
           B=((-1.0, 0.0, 0.0, 0.0),
              (1.0, 0.0, 0.0, 0.0),
              (0.0, 0.0, -1.0, 0.0),
              (0.0, 0.0, 1.0, 0.0)))
    >>> k
    array([1. , 0.01, 0.02, 2. ])

    Multiple fixed compounds also work fine:

    >>> pH = 6
    >>> scheme = rx.parse_reactions("A + H2O -> B <=> B- + H+")
    >>> k = np.array([1.0, 100.0, 2.0])
    >>> scheme, k = rx.get_fixed_scheme(scheme, k, {"H+": 10**-pH, "H2O": 55.6})
    >>> scheme
    Scheme(compounds=('A', 'B', 'B-'),
           reactions=('A -> B',
                      'B -> B-',
                      'B- -> B'),
           is_half_equilibrium=(False, True, True),
           A=((-1.0, 0.0, 0.0),
              (1.0, -1.0, 1.0),
              (0.0, 1.0, -1.0)),
           B=((-1.0, 0.0, 0.0),
              (1.0, -1.0, 0.0),
              (0.0, 1.0, 0.0)))
    >>> k
    array([5.56e+01, 1.00e+02, 2.00e-06])

    This function is a no-op if `fixed_y0` is empty, which is very important
    for overall code consistency:

    >>> scheme = rx.parse_reactions(["AH <=> A- + H+", "B- + H+ <=> BH"])
    >>> k = np.array([1, 1, 2, 2])
    >>> new_scheme, new_k = rx.get_fixed_scheme(scheme, k, {})
    >>> new_scheme == scheme
    True
    >>> np.allclose(new_k, k)
    True

    """
    new_k = np.asarray(k, dtype=float).copy()
    new_reactions = []
    for i, (reaction, is_half_equilibrium) in enumerate(
        zip(scheme.reactions, scheme.is_half_equilibrium)
    ):
        for reactants, products, _ in rx.core._parse_reactions(reaction):
            new_reactants = tuple(
                (coeff, compound)
                for (coeff, compound) in reactants
                if compound not in fixed_y0
            )
            new_products = tuple(
                (coeff, compound)
                for (coeff, compound) in products
                if compound not in fixed_y0
            )

            for fixed_compound in fixed_y0:
                for coeff, compound in reactants:
                    if fixed_compound == compound:
                        new_k[i] *= fixed_y0[fixed_compound] ** coeff

            new_reactions.append((new_reactants, new_products, is_half_equilibrium))

    new_reactions = tuple(r for r in rx.core._unparse_reactions(new_reactions))
    new_is_half_equilibrium = scheme.is_half_equilibrium

    new_A = []
    new_B = []
    new_compounds = []
    for i, (compound, row_A, row_B) in enumerate(
        zip(scheme.compounds, scheme.A, scheme.B)
    ):
        if compound not in fixed_y0:
            new_compounds.append(compound)
            new_A.append(row_A)
            new_B.append(row_B)

    new_compounds = tuple(new_compounds)
    new_A = tuple(new_A)
    new_B = tuple(new_B)

    return (
        rx.core.Scheme(
            compounds=new_compounds,
            reactions=new_reactions,
            is_half_equilibrium=new_is_half_equilibrium,
            A=new_A,
            B=new_B,
        ),
        new_k,
    )


# TODO(schneiderfelipe): this is probably not ready yet
def get_bias(
    scheme,
    compounds,
    data,
    y0,
    tunneling="eckart",
    qrrho=True,
    temperature=298.15,
    pressure=constants.atm,
    method="Radau",
    rtol=1e-5,
    atol=1e-11,
):
    r"""Estimate a energy bias for a given set of reference data points.

    Parameters
    ----------
    scheme : Scheme
    compounds : dict-like, optional
    data : dict-like of array-like
    y0: array-like
    tunneling : str or None, optional
        Choose between "eckart", "wigner" or None (or "none").
    qrrho : bool or tuple-like, optional
        Apply both the quasi-rigid rotor harmonic oscillator (QRRHO)
        approximations of M. Head-Gordon (enthalpy correction, see
        doi:10.1021/jp509921r) and S. Grimme (entropy correction, see
        doi:10.1002/chem.201200497) on top of the classical RRHO.
    temperature : array-like, optional
        Absolute temperature in Kelvin.
    pressure : array-like, optional
        Reference gas pressure.
    delta_freeenergies : array-like, optional
    molecularity : array-like, optional
        Reaction order, i.e., number of molecules that come together to react.
        If set, this is used to calculate `delta_moles` for
        `equilibrium_constant`, which effectively calculates a solution
        equilibrium constant between reactants and the transition state for
        gas phase data. You should set this to `None` if your free energies
        were already adjusted for solution Gibbs free energies.
    volume : float, optional
        Molar volume.

    Returns
    -------
    array-like

    Examples
    --------
    >>> model = rx.parse_model("data/tanaka1996/UMP2/cc-pVTZ/model.jk")

    The following are some estimates on actual atmospheric concentrations:

    >>> y0 = [4.8120675684099e-5,
    ...       2.8206357713029e-5,
    ...       0.0,
    ...       0.0,
    ...       2.7426565371219e-5]
    >>> data = {"t": [1.276472128376942246e-6,
    ...               1.446535794555581743e-4,
    ...               1.717069678525567564e-2],
    ...         "CH3·": [9.694916853338366211e-9,
    ...                  1.066033349343709026e-6,
    ...                  2.632179124780495175e-5]}
    >>> get_bias(model.scheme, model.compounds, data, y0) / constants.kcal
    -1.364171
    """
    max_time = np.max(data["t"])

    def f(bias):
        k = rx.get_k(
            scheme,
            compounds,
            bias=bias,
            tunneling=tunneling,
            qrrho=qrrho,
            temperature=temperature,
            pressure=pressure,
        )

        # TODO(schneiderfelipe): support schemes with fixed concentrations
        dydt = rx.get_dydt(scheme, k)
        y, _ = rx.get_y(
            dydt,
            y0=y0,
            method=method,
            rtol=rtol,
            atol=atol,
            max_time=max_time,
        )

        yhat = y(data["t"])
        return np.sum(
            [
                (yhat[i] - data[name]) ** 2
                for (i, name) in enumerate(compounds)
                if name in data
            ]
        )

    res = minimize_scalar(f)
    return res.x
