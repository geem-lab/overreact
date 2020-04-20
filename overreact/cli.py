#!/usr/bin/env python3

"""Command-line interface."""

import argparse
import logging
import os
import re
import sys

import numpy as np

from overreact import api
from overreact import constants
from overreact import core
from overreact import io

levels = [logging.WARNING, logging.INFO, logging.DEBUG]


def summarize_model(
    model, quantities=None, savepath=None, plot=False, qrrho=True, temperature=298.15
):
    """Produce a string describing a model.

    The returned string contains a final line break.

    Parameters
    ----------
    model : dict-like
    qrrho : bool, optional
        Apply both the quasi-rigid rotor harmonic oscilator (QRRHO)
        approximations of M. Head-Gordon (enthalpy correction, see
        doi:10.1021/jp509921r) and S. Grimme (entropy correction, see
        doi:10.1002/chem.201200497) on top of the classical RRHO.
    temperature : array-like, optional
        Absolute temperature in Kelvin.

    Returns
    -------
    str

    Examples
    --------
    >>> model = api.parse_model("data/ethane/B97-3c/model.jk")
    >>> print(summarize_model(model))
    <BLANKLINE>
                                   (READ) REACTIONS
    ------------------------------------------------------------------------------
                                     S -> E‡ -> S
    <BLANKLINE>
                                                              (PARSED) REACTIONS
    --------------------------------------------------------------------------------------------------------------------------------------
    no                      reactants                          via‡                           products                      half equilib.?
    -- --------------------------------------------------- ------------ --------------------------------------------------- --------------
     0                          S                               E‡                               S
    <BLANKLINE>
                                                            COMPOUNDS
    -------------------------------------------------------------------------------------------------------------------------
    no      compound        elec. energy   spin mult.     smallest vibfreqs                    original logfile
                                [Eh]                            [cm⁻¹]
    -- ----------------- ----------------- ---------- ------------------------- ---------------------------------------------
     0         S          -79.788170457691     1        307.6,   825.4,   826.1           data/ethane/B97-3c/staggered.out
     1         E‡         -79.783894160233     1       -298.9,   902.2,   902.5            data/ethane/B97-3c/eclipsed.out
    <BLANKLINE>
                           CALCULATED THERMOCHEMISTRY (COMPOUNDS)
    -----------------------------------------------------------------------------------
    no      compound      mass  Gcorr(298.15K) Ucorr(298.15K) Hcorr(298.15K) S(298.15K)
                         [amu]    [kcal/mol]     [kcal/mol]     [kcal/mol]   [cal/mol·K]
    -- ----------------- ------ -------------- -------------- -------------- ----------
     0         S          30.07          33.01          48.63          49.22      54.40
     1         E‡         30.07          32.95          48.15          48.74      52.96
    <BLANKLINE>
                                                                                   CALCULATED THERMOCHEMISTRY (REACTIONS)
    ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    no                               reaction                              Δmass‡    ΔG‡        ΔE‡        ΔU‡        ΔH‡         ΔS‡     Δmass°    ΔG°        ΔE°        ΔU°        ΔH°         ΔS°
                                                                           [amu]  [kcal/mol] [kcal/mol] [kcal/mol] [kcal/mol] [cal/mol·K] [amu]  [kcal/mol] [kcal/mol] [kcal/mol] [kcal/mol] [cal/mol·K]
    -- ------------------------------------------------------------------- ------ ---------- ---------- ---------- ---------- ----------- ------ ---------- ---------- ---------- ---------- -----------
     0                                S -> S                                 0.00       2.63       2.68       2.20       2.20       -1.44   0.00       0.00       0.00       0.00       0.00        0.00
    <BLANKLINE>
                                                              CALCULATED KINETICS
    ---------------------------------------------------------------------------------------------------------------------------------------
    no                               reaction                              half equilib.?      k                 k                  k
                                                                                          [M⁻ⁿ⁺¹·s⁻¹] [(cm³/particle)ⁿ⁻¹·s⁻¹] [atm⁻ⁿ⁺¹·s⁻¹]
    -- ------------------------------------------------------------------- -------------- ----------- ----------------------- -------------
     0                                S -> S                                                8.2e+10           8.2e+10            8.2e+10
    """
    sections = []
    sections.append(_summarize_scheme(model.scheme))
    sections.append(_summarize_compounds(model.compounds))
    sections.append(
        _summarize_thermochemistry(
            model.scheme, model.compounds, qrrho=qrrho, temperature=temperature
        )
    )
    sections.append(
        _summarize_kinetics(
            model.scheme,
            model.compounds,
            quantities=quantities,
            savepath=savepath,
            plot=plot,
            qrrho=qrrho,
            temperature=temperature,
        )
    )
    return "".join(sections)


def _summarize_scheme(scheme):
    """Produce a string describing a reaction scheme.

    This is meant to be used from within `summarize_model`. The returned string
    contains a final line break.

    Parameters
    ----------
    scheme : Scheme

    Returns
    -------
    str
    """
    scheme = core._check_scheme(scheme)
    reactions = _format_table(
        [[row] for row in core.unparse_reactions(scheme).split("\n")],
        title="(read) reactions",
        length=[78],
    )

    transition_states = core.get_transition_states(
        scheme.A, scheme.B, scheme.is_half_equilibrium
    )

    reaction_rows = [
        [
            "no",
            "reactants".center(51),
            "via‡".center(12),
            "products".center(51),
            "half equilib.?",
        ]
    ]
    reaction_rows.append(["-" * len(field) for field in reaction_rows[0]])
    for i, reaction in enumerate(scheme.reactions):
        reactants, _, products = re.split(r"\s*(->|<=>|<-)\s*", reaction)
        row = [f"{i:2d}", reactants, None, products, None]
        if transition_states[i] is not None:
            row[2] = scheme.compounds[transition_states[i]]
        elif scheme.is_half_equilibrium[i]:
            row[4] = True
        reaction_rows.append(row)
    parsed_reactions = _format_table(reaction_rows, title="(parsed) reactions")

    return reactions + parsed_reactions


def _summarize_compounds(compounds):
    """Produce a string describing compounds.

    This is meant to be used from within `summarize_model`. The returned string
    contains a final line break.

    Parameters
    ----------
    compounds : dict-like

    Returns
    -------
    str

    Raises
    ------
    ValueError
        If at least one compound has no data defined.
    """
    undefined_compounds = []
    for name in compounds:
        if not compounds[name]:
            undefined_compounds.append(name)
    if undefined_compounds:
        raise ValueError(f"undefined compounds: {', '.join(undefined_compounds)}")

    compound_rows = [
        [
            "no",
            "compound".center(17),
            "elec. energy".center(17),
            "spin mult.",
            "smallest vibfreqs".center(25),
            "original logfile".center(45),
        ],
        [None, None, "[Eh]", None, "[cm⁻¹]", None],
    ]
    compound_rows.append(["-" * len(field) for field in compound_rows[0]])
    for i, (name, data) in enumerate(compounds.items()):
        compound_rows.append(
            [
                f"{i:2d}",
                name,
                f"{data.energy / (constants.hartree * constants.N_A):17.12f}",
                data.mult,
                ", ".join([f"{vibfreq:7.1f}" for vibfreq in data.vibfreqs[:3]]),
                # TODO(schneiderfelipe): show only the file name and inform
                # the absolute path to folder (as a bash variable) somewhere
                # else.
                data.logfile,
            ]
        )

    return _format_table(compound_rows, title="compounds")


def _summarize_thermochemistry(scheme, compounds, qrrho=True, temperature=298.15):
    """Produce a string describing the thermochemistry of a reaction scheme.

    This is meant to be used from within `summarize_model`. The returned string
    contains a final line break.

    Parameters
    ----------
    scheme : Scheme
    compounds : dict-like
    qrrho : bool, optional
        Apply both the quasi-rigid rotor harmonic oscilator (QRRHO)
        approximations of M. Head-Gordon (enthalpy correction, see
        doi:10.1021/jp509921r) and S. Grimme (entropy correction, see
        doi:10.1002/chem.201200497) on top of the classical RRHO.
    temperature : array-like, optional
        Absolute temperature in Kelvin.

    Returns
    -------
    str
    """
    scheme = core._check_scheme(scheme)

    molecular_masses = np.array(
        [np.sum(data.atommasses) for name, data in compounds.items()]
    )
    energies = np.array([data.energy for name, data in compounds.items()])
    internal_energies = api.get_internal_energies(
        compounds, qrrho=qrrho, temperature=temperature
    )
    enthalpies = api.get_enthalpies(compounds, qrrho=qrrho, temperature=temperature)
    entropies = api.get_entropies(compounds, qrrho=qrrho, temperature=temperature)
    freeenergies = enthalpies - temperature * entropies

    compound_rows = [
        [
            "no",
            "compound".center(17),
            "mass".center(6),
            f"Gcorr({temperature}K)",
            f"Ucorr({temperature}K)",
            f"Hcorr({temperature}K)",
            f"S({temperature}K)",
        ],
        [None, None, "[amu]", "[kcal/mol]", "[kcal/mol]", "[kcal/mol]", "[cal/mol·K]"],
    ]
    compound_rows.append(["-" * len(field) for field in compound_rows[0]])
    for i, (name, data) in enumerate(compounds.items()):
        compound_rows.append(
            [
                f"{i:2d}",
                name,
                f"{molecular_masses[i]:6.2f}",
                f"{(freeenergies[i] - data.energy) / constants.kcal:14.2f}",
                f"{(internal_energies[i] - data.energy) / constants.kcal:14.2f}",
                f"{(enthalpies[i] - data.energy) / constants.kcal:14.2f}",
                f"{entropies[i] / constants.calorie:10.2f}",
            ]
        )
    compounds_table = _format_table(
        compound_rows, title="calculated thermochemistry (compounds)"
    )

    delta_mass = api.get_delta(scheme.A, molecular_masses)
    delta_energies = api.get_delta(scheme.A, energies)
    delta_internal_energies = api.get_delta(scheme.A, internal_energies)
    delta_enthalpies = api.get_delta(scheme.A, enthalpies)
    delta_entropies = api.get_delta(scheme.A, entropies)
    delta_freeenergies = api.get_delta(scheme.A, freeenergies)

    delta_activation_mass = api.get_delta(scheme.B, molecular_masses)
    delta_activation_energies = api.get_delta(scheme.B, energies)
    delta_activation_internal_energies = api.get_delta(scheme.B, internal_energies)
    delta_activation_enthalpies = api.get_delta(scheme.B, enthalpies)
    delta_activation_entropies = api.get_delta(scheme.B, entropies)
    delta_activation_freeenergies = api.get_delta(scheme.B, freeenergies)

    reaction_rows = [
        [
            "no",
            "reaction".center(67),
            "Δmass‡",
            "ΔG‡".center(10),
            "ΔE‡".center(10),
            "ΔU‡".center(10),
            "ΔH‡".center(10),
            "ΔS‡".center(11),
            "Δmass°",
            "ΔG°".center(10),
            "ΔE°".center(10),
            "ΔU°".center(10),
            "ΔH°".center(10),
            "ΔS°".center(11),
        ],
        [
            None,
            None,
            "[amu]",
            "[kcal/mol]",
            "[kcal/mol]",
            "[kcal/mol]",
            "[kcal/mol]",
            "[cal/mol·K]",
            "[amu]",
            "[kcal/mol]",
            "[kcal/mol]",
            "[kcal/mol]",
            "[kcal/mol]",
            "[cal/mol·K]",
        ],
    ]
    reaction_rows.append(["-" * len(field) for field in reaction_rows[0]])
    for i, reaction in enumerate(scheme.reactions):
        if scheme.is_half_equilibrium[i]:
            row = [
                f"{i:2d}",
                reaction,
                None,
                None,
                None,
                None,
                None,
                None,
                f"{delta_mass[i]:6.2f}",
                f"{delta_freeenergies[i] / constants.kcal:10.2f}",
                f"{delta_energies[i] / constants.kcal:10.2f}",
                f"{delta_internal_energies[i] / constants.kcal:10.2f}",
                f"{delta_enthalpies[i] / constants.kcal:10.2f}",
                f"{delta_entropies[i] / constants.calorie:11.2f}",
            ]
        else:
            row = [
                f"{i:2d}",
                reaction,
                f"{delta_activation_mass[i]:6.2f}",
                f"{delta_activation_freeenergies[i] / constants.kcal:10.2f}",
                f"{delta_activation_energies[i] / constants.kcal:10.2f}",
                f"{delta_activation_internal_energies[i] / constants.kcal:10.2f}",
                f"{delta_activation_enthalpies[i] / constants.kcal:10.2f}",
                f"{delta_activation_entropies[i] / constants.calorie:11.2f}",
                f"{delta_mass[i]:6.2f}",
                f"{delta_freeenergies[i] / constants.kcal:10.2f}",
                f"{delta_energies[i] / constants.kcal:10.2f}",
                f"{delta_internal_energies[i] / constants.kcal:10.2f}",
                f"{delta_enthalpies[i] / constants.kcal:10.2f}",
                f"{delta_entropies[i] / constants.calorie:11.2f}",
            ]

        reaction_rows.append(row)
    reactions_table = _format_table(
        reaction_rows, title="calculated thermochemistry (REACTIONS)"
    )

    return compounds_table + reactions_table


def _summarize_kinetics(
    scheme,
    compounds,
    quantities=None,
    savepath=None,
    plot=False,
    qrrho=True,
    temperature=298.15,
):
    # TODO(schneiderfelipe): apply other corrections to k
    k = {
        "M⁻ⁿ⁺¹·s⁻¹": api.get_k(
            scheme, compounds, qrrho=qrrho, temperature=temperature, scale="l mol-1 s-1"
        ),
        "(cm³/particle)ⁿ⁻¹·s⁻¹": api.get_k(
            scheme,
            compounds,
            qrrho=qrrho,
            temperature=temperature,
            scale="cm3 particle-1 s-1",
        ),
        "atm⁻ⁿ⁺¹·s⁻¹": api.get_k(
            scheme, compounds, qrrho=qrrho, temperature=temperature, scale="atm-1 s-1"
        ),
    }

    reaction_rows = [
        ["no", "reaction".center(67), "half equilib.?"]
        + ["k".center(len(scale) + 2) for scale in k],
        [None, None, None] + [f"[{scale}]" for scale in k],
    ]
    reaction_rows.append(["-" * len(field) for field in reaction_rows[0]])
    for i, reaction in enumerate(scheme.reactions):
        row = [f"{i:2d}", reaction, None] + [f"{k[scale][i]:7.2g}" for scale in k]
        if scheme.is_half_equilibrium[i]:
            row[2] = True

        reaction_rows.append(row)
    reactions_table = _format_table(reaction_rows, title="calculated kinetics")

    if quantities is not None and quantities:
        # TODO(schneiderfelipe): apply post-processing to scheme, k (with functions
        # that receive a scheme, k and return a scheme, k). One that solves the pH
        # problem is welcome: get a scheme, k and, for each reaction in it, remove
        # the H+ and multiplies the reaction rate constants by the proper
        # concentration if there is H+ in the reactants.
        # TODO(schneiderfelipe): encapsulate everything in a function that depends
        # on the freeenergies as first parameter
        dydt = api.get_dydt(scheme, k)

        y0 = np.zeros(len(scheme.compounds))
        for spec in quantities:
            fields = spec.split(":", 1)
            name = fields[0]
            try:
                quantity = float(fields[1])
            except (IndexError, ValueError):
                raise ValueError(
                    f"badly formatted quantities: '{' '.join(quantities)}'"
                )

            # TODO(schneiderfelipe): the following is inefficient but probably OK
            y0[scheme.compounds.index(name)] = quantity

        t, y, r = api.get_y(dydt, y0=y0, method="Radau")
        if plot:
            import matplotlib.pyplot as plt

            for i, name in enumerate(scheme.compounds):
                if not core.is_transition_state(name):
                    plt.plot(t, y[i], label=name)

            plt.legend()
            plt.xlabel("Time (s)")
            plt.ylabel("Concentration (M)")
            plt.show()

        if savepath is not None:
            np.savetxt(
                savepath,
                np.block([t[:, np.newaxis], y.T]),
                header=f"t,{','.join(scheme.compounds)}",
                delimiter=",",
            )

        # TODO(schneiderfelipe): implement the degree of rate control

    return reactions_table


def _create_banner(text, title_char="-", width=78):
    banner = f"\n\n{text.center(width)}"
    return banner + f"\n{title_char * width}\n"


def _format_table(rows, length=None, sep=" ", title=None):
    """Format a table for printing.

    The width of each column is taken from the length of the first row. Nones
    are omitted.

    Parameters
    ----------
    rows : sequence of sequence of str
    length : sequence of int, optional
    sep : str, optional

    Returns
    -------
    str

    Examples
    --------
    >>> print(_format_table([["one  ", "two"], [1, 2], ["hello", "world"]]))
    one    two
    1      2
    hello  world
    """
    default_length = [len(field) for field in rows[0]]
    if length is None:
        length = default_length
    else:
        length = [a if a is not None else b for a, b in zip(length, default_length)]

    lines = []
    for row in rows:
        line = [
            str(field).center(length[i]) if field is not None else " " * length[i]
            for i, field in enumerate(row)
        ]
        lines.append(sep.join(line))

    table = "\n".join(lines)
    if title is not None:
        return (
            _create_banner(title.upper(), width=np.sum(length) + len(length) - 1)
            + table
        )
    return table + "\n"


def main():
    """Command-line interface."""
    # TODO(schneiderfelipe): test and docs
    parser = argparse.ArgumentParser(
        description="Interface for building and modifying models."
    )
    parser.add_argument("path", help="path to a source (.k) or model file (.jk)")
    parser.add_argument(
        "quantities",
        help=(
            "optional initial compound concentrations as 'name:quantity' for "
            "a microkinetic simulation"
        ),
        nargs="*",
    )
    parser.add_argument("-T", "--temperature", type=float, default=298.15)
    # TODO(schneiderfelipe): support pressure specification!
    parser.add_argument("-p", "--pressure", type=float, default=constants.atm)
    parser.add_argument(
        "--no-qrrho",
        dest="qrrho",
        help=(
            "disable the quasi-rigid rotor harmonic oscilator (QRRHO) "
            "approximations to enthalpies and entropies "
            "(see doi:10.1021/jp509921r and doi:10.1002/chem.201200497)"
        ),
        action="store_false",
    )
    parser.add_argument(
        "--plot",
        help=(
            "plot concentrations as a function of time in a microkinetics simulation"
        ),
        action="store_true",
    )
    parser.add_argument(
        "-c",
        "--compile",
        help="only compile a source file (.k) into a model file (.jk)",
        action="store_true",
    )
    parser.add_argument(
        "-v", "--verbose", help="increase output verbosity", action="count", default=0
    )
    # TODO(schneiderfelipe): --dry-run|-n for testing purposes (useful
    # usage together with --compile|-c --- or should we consider --compile|-c
    # always as a do-nothing (no analysis)?).
    # TODO(schneiderfelipe): some commands for concatenating/summing .k/.jk
    # files. This might be useful for some of the more complex operations I
    # want to be able to do in the future.
    args = parser.parse_args()
    logging.basicConfig(
        level=levels[min(len(levels) - 1, args.verbose)], stream=sys.stdout
    )
    for handler in logging.root.handlers:
        handler.setFormatter(io.InterfaceFormatter("%(message)s"))

    model = io.parse_model(args.path, force_compile=args.compile)
    print(
        summarize_model(
            model,
            quantities=args.quantities,
            savepath=os.path.splitext(args.path)[0] + ".csv",
            plot=args.plot,
            qrrho=args.qrrho,
            temperature=args.temperature,
        )
    )


if __name__ == "__main__":
    main()
