#!/usr/bin/env python3

"""Command-line interface."""

import shutil
import argparse
import logging
import os
import re
import sys

import numpy as np
from rich.table import Table
from rich.table import Column
from rich.console import Console
from rich.markdown import Markdown
from rich import box

from overreact import __version__
from overreact import api
from overreact import constants
from overreact import core
from overreact import io


# TODO(schneiderfelipe): test this class
class Report:
    """Produce a report object based on a model.

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

    Examples
    --------
    >>> model = api.parse_model("data/ethane/B97-3c/model.jk")
    >>> Report(model)  # doctest: +SKIP
    ╔══════════════════════════════════════════════════════════════════════════════╗
    ║                                overreact 1.0                                 ║
    ╚══════════════════════════════════════════════════════════════════════════════╝

    Construct precise chemical microkinetic models from first principles


                                Input description

                                    (read) reactions

                                    S -> E‡ -> S

                                (parsed) reactions
                    ╷          ╷      ╷         ╷
                    no │ reactant │ via‡ │ product │ half equilib.?
                ╶────┼──────────┼──────┼─────────┼────────────────╴
                    0 │    S     │  E‡  │    S    │
                    ╵          ╵      ╵         ╵
                                        logfiles
                    ╷          ╷
                    no │ compound │               path
                ╶────┼──────────┼──────────────────────────────────╴
                    0 │    S     │ data/ethane/B97-3c/staggered.out
                    1 │    E‡    │ data/ethane/B97-3c/eclipsed.out
                    ╵          ╵
                                    compounds
        ╷          ╷                   ╷            ╷
        no │ compound │   elec. energy    │ spin mult. │   smallest vibfreqs
        │          │       [Eₕ]        │            │         [cm⁻¹]
    ╶────┼──────────┼───────────────────┼────────────┼────────────────────────╴
        0 │    S     │  -79.788170457691 │     1      │ +307.6, +825.4, +826.1
        1 │    E‡    │  -79.783894160233 │     1      │ -298.9, +902.2, +902.5
        ╵          ╵                   ╵            ╵
    Temperature = 298.15 K

                                    Output section

                        calculated thermochemistry (compounds)
        ╷          ╷        ╷             ╷            ╷             ╷
    no │ compound │  mass  │    Gᶜᵒʳʳ    │   Uᶜᵒʳʳ    │    Hᶜᵒʳʳ    │     S
        │          │ [amu]  │ [kcal/mol]  │ [kcal/mol] │ [kcal/mol]  │ [cal/mol·…
    ╶────┼──────────┼────────┼─────────────┼────────────┼─────────────┼────────────╴
    0 │    S     │  30.07 │          3… │          … │          4… │      54.40
    1 │    E‡    │  30.07 │          3… │          … │          4… │      52.96
        ╵          ╵        ╵             ╵            ╵             ╵
                        calculated thermochemistry (reactions°)
        ╷          ╷        ╷          ╷          ╷          ╷          ╷
    no │ reaction │ Δmass° │   ΔG°    │   ΔE°    │   ΔU°    │   ΔH°    │   ΔS°
        │          │ [amu]  │ [kcal/m… │ [kcal/m… │ [kcal/m… │ [kcal/m… │ [cal/m…
    ╶────┼──────────┼────────┼──────────┼──────────┼──────────┼──────────┼─────────╴
    0 │  S -> S  │   0.00 │       0… │       0… │       0… │       0… │       …
        ╵          ╵        ╵          ╵          ╵          ╵          ╵
                        calculated thermochemistry (reactions‡)
        ╷          ╷        ╷          ╷          ╷          ╷          ╷
    no │ reaction │ Δmass‡ │   ΔG‡    │   ΔE‡    │   ΔU‡    │   ΔH‡    │   ΔS‡
        │          │ [amu]  │ [kcal/m… │ [kcal/m… │ [kcal/m… │ [kcal/m… │ [cal/m…
    ╶────┼──────────┼────────┼──────────┼──────────┼──────────┼──────────┼─────────╴
    0 │  S -> S  │   0.00 │       2… │       2… │       2… │       2… │       …
        ╵          ╵        ╵          ╵          ╵          ╵          ╵
                                calculated kinetics
        ╷          ╷                ╷             ╷               ╷
    no │ reaction │ half equilib.? │      k      │       k       │       k
        │          │                │ [M⁻ⁿ⁺¹·s⁻¹] │ [(cm³/partic… │ [atm⁻ⁿ⁺¹·s⁻¹]
    ╶────┼──────────┼────────────────┼─────────────┼───────────────┼───────────────╴
    0 │  S -> S  │                │   8.2e+10   │    8.2e+10    │    8.2e+10
        ╵          ╵                ╵             ╵               ╵
    """

    def __init__(
        self,
        model,
        quantities=None,
        savepath=None,
        plot=False,  # TODO(schneiderfelipe): change to do_plot
        qrrho=True,  # TODO(schneiderfelipe): change to use_qrrho
        temperature=298.15,
        box_style=box.SIMPLE,
    ):
        self.model = model
        self.quantities = quantities
        self.savepath = savepath
        self.plot = plot
        self.qrrho = qrrho
        self.temperature = temperature
        self.box_style = box_style

    def __rich_console__(self, console, options):
        """
        Implement Rich Console protocol.

        This works by yielding from generators.

        Yields
        ------
        renderable
        """
        yield from self._yield_scheme()
        yield from self._yield_compounds()
        yield from self._yield_thermochemistry()
        yield from self._yield_kinetics()

    def _yield_scheme(self):
        """Produce a renderables describing the reaction scheme.

        This is meant to be used from within `__rich_console__`.

        Yields
        ------
        renderable
        """
        scheme = core._check_scheme(self.model.scheme)

        raw_table = Table(
            title="(read) reactions", box=self.box_style, show_header=False
        )
        raw_table.add_column(justify="center")
        for r in core.unparse_reactions(scheme).split("\n"):
            raw_table.add_row(r)
        yield raw_table

        transition_states = core.get_transition_states(
            scheme.A, scheme.B, scheme.is_half_equilibrium
        )

        parsed_table = Table(
            Column("no", justify="center"),
            Column("reactant", justify="center"),
            Column("via‡", justify="center"),
            Column("product", justify="center"),
            Column("half equilib.?", justify="center"),
            title="(parsed) reactions",
            box=self.box_style,
        )
        for i, reaction in enumerate(scheme.reactions):
            reactants, _, products = re.split(r"\s*(->|<=>|<-)\s*", reaction)
            # TODO(schneiderfelipe): should we use "No" instead of None for
            # "half-equilib.?"?
            row = [f"{i:2d}", reactants, None, products, None]
            if transition_states[i] is not None:
                row[2] = scheme.compounds[transition_states[i]]
            elif scheme.is_half_equilibrium[i]:
                row[4] = True
            parsed_table.add_row(*row)
        yield parsed_table

    def _yield_compounds(self):
        """Produce a renderables describing the compounds.

        This is meant to be used from within `__rich_console__`.

        Yields
        ------
        renderable

        Raises
        ------
        ValueError
            If at least one compound has no data defined.
        """
        undefined_compounds = []
        for name in self.model.compounds:
            if not self.model.compounds[name]:
                undefined_compounds.append(name)
        if undefined_compounds:
            raise ValueError(f"undefined compounds: {', '.join(undefined_compounds)}")

        logfiles_table = Table(
            Column("no", justify="center"),
            Column("compound", justify="center"),
            Column("path", justify="center"),
            title="logfiles",
            box=self.box_style,
        )
        compounds_table = Table(
            Column("no", justify="center"),
            Column("compound", justify="center"),
            Column("elec. energy\n\[Eₕ]", justify="center"),
            Column("spin mult.", justify="center"),
            Column("smallest vibfreqs\n\[cm⁻¹]", justify="center"),
            title="compounds",
            box=self.box_style,
        )
        for i, (name, data) in enumerate(self.model.compounds.items()):
            logfiles_table.add_row(
                f"{i:2d}",
                name,
                # TODO(schneiderfelipe): show only the file name and inform
                # the absolute path to folder (as a bash variable) somewhere
                # else.
                data.logfile,
            )
            compounds_table.add_row(
                f"{i:2d}",
                name,
                f"{data.energy / (constants.hartree * constants.N_A):17.12f}",
                f"{data.mult}",
                ", ".join([f"{vibfreq:+.1f}" for vibfreq in data.vibfreqs[:3]]),
            )
        yield logfiles_table
        yield compounds_table

    def _yield_thermochemistry(self):
        """Produce a renderables describing the thermochemistry of the reaction scheme.

        This is meant to be used from within `__rich_console__`.

        Yields
        ------
        renderable
        """
        scheme = core._check_scheme(self.model.scheme)

        molecular_masses = np.array(
            [np.sum(data.atommasses) for name, data in self.model.compounds.items()]
        )
        energies = np.array(
            [data.energy for name, data in self.model.compounds.items()]
        )
        internal_energies = api.get_internal_energies(
            self.model.compounds, qrrho=self.qrrho, temperature=self.temperature
        )
        enthalpies = api.get_enthalpies(
            self.model.compounds, qrrho=self.qrrho, temperature=self.temperature
        )
        entropies = api.get_entropies(
            self.model.compounds, qrrho=self.qrrho, temperature=self.temperature
        )
        freeenergies = enthalpies - self.temperature * entropies

        compounds_table = Table(
            Column("no", justify="center"),
            Column("compound", justify="center"),
            Column("mass\n\[amu]", justify="center"),
            Column("Gᶜᵒʳʳ\n\[kcal/mol]", justify="center", style="green"),
            Column("Uᶜᵒʳʳ\n\[kcal/mol]", justify="center"),
            Column("Hᶜᵒʳʳ\n\[kcal/mol]", justify="center"),
            Column("S\n\[cal/mol·K]", justify="center"),
            title="calculated thermochemistry (compounds)",
            box=self.box_style,
        )
        for i, (name, data) in enumerate(self.model.compounds.items()):
            compounds_table.add_row(
                f"{i:2d}",
                name,
                f"{molecular_masses[i]:6.2f}",
                f"{(freeenergies[i] - data.energy) / constants.kcal:14.2f}",
                f"{(internal_energies[i] - data.energy) / constants.kcal:14.2f}",
                f"{(enthalpies[i] - data.energy) / constants.kcal:14.2f}",
                f"{entropies[i] / constants.calorie:10.2f}",
            )
        yield compounds_table

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

        circ_table = Table(
            Column("no", justify="center"),
            Column("reaction", justify="center"),
            Column("Δmass°\n\[amu]", justify="center"),
            Column("ΔG°\n\[kcal/mol]", justify="center", style="green"),
            Column("ΔE°\n\[kcal/mol]", justify="center"),
            Column("ΔU°\n\[kcal/mol]", justify="center"),
            Column("ΔH°\n\[kcal/mol]", justify="center"),
            Column("ΔS°\n\[cal/mol·K]", justify="center"),
            title="calculated thermochemistry (reactions°)",
            box=self.box_style,
        )
        dagger_table = Table(
            Column("no", justify="center"),
            Column("reaction", justify="center"),
            Column("Δmass‡\n\[amu]", justify="center"),
            Column("ΔG‡\n\[kcal/mol]", justify="center", style="green"),
            Column("ΔE‡\n\[kcal/mol]", justify="center"),
            Column("ΔU‡\n\[kcal/mol]", justify="center"),
            Column("ΔH‡\n\[kcal/mol]", justify="center"),
            Column("ΔS‡\n\[cal/mol·K]", justify="center"),
            title="calculated thermochemistry (reactions‡)",
            box=self.box_style,
        )
        for i, reaction in enumerate(scheme.reactions):
            if scheme.is_half_equilibrium[i]:
                circ_row = [
                    f"{i:2d}",
                    reaction,
                    f"{delta_mass[i]:6.2f}",
                    f"{delta_freeenergies[i] / constants.kcal:10.2f}",
                    f"{delta_energies[i] / constants.kcal:10.2f}",
                    f"{delta_internal_energies[i] / constants.kcal:10.2f}",
                    f"{delta_enthalpies[i] / constants.kcal:10.2f}",
                    f"{delta_entropies[i] / constants.calorie:11.2f}",
                ]
                dagger_row = [
                    f"{i:2d}",
                    reaction,
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                ]
            else:
                circ_row = [
                    f"{i:2d}",
                    reaction,
                    f"{delta_mass[i]:6.2f}",
                    f"{delta_freeenergies[i] / constants.kcal:10.2f}",
                    f"{delta_energies[i] / constants.kcal:10.2f}",
                    f"{delta_internal_energies[i] / constants.kcal:10.2f}",
                    f"{delta_enthalpies[i] / constants.kcal:10.2f}",
                    f"{delta_entropies[i] / constants.calorie:11.2f}",
                ]
                dagger_row = [
                    f"{i:2d}",
                    reaction,
                    f"{delta_activation_mass[i]:6.2f}",
                    f"{delta_activation_freeenergies[i] / constants.kcal:10.2f}",
                    f"{delta_activation_energies[i] / constants.kcal:10.2f}",
                    f"{delta_activation_internal_energies[i] / constants.kcal:10.2f}",
                    f"{delta_activation_enthalpies[i] / constants.kcal:10.2f}",
                    f"{delta_activation_entropies[i] / constants.calorie:11.2f}",
                ]

            circ_table.add_row(*circ_row)
            dagger_table.add_row(*dagger_row)
        yield circ_table
        yield dagger_table

    def _yield_kinetics(self):
        """Produce a renderables describing the kinetics of the system.

        This is meant to be used from within `__rich_console__`.

        Yields
        ------
        renderable
        """
        # TODO(schneiderfelipe): apply other corrections to k (such as
        # diffusion control)
        k = {
            "M⁻ⁿ⁺¹·s⁻¹": api.get_k(
                self.model.scheme,
                self.model.compounds,
                qrrho=self.qrrho,
                temperature=self.temperature,
                scale="l mol-1 s-1",
            ),
            "(cm³/particle)ⁿ⁻¹·s⁻¹": api.get_k(
                self.model.scheme,
                self.model.compounds,
                qrrho=self.qrrho,
                temperature=self.temperature,
                scale="cm3 particle-1 s-1",
            ),
            "atm⁻ⁿ⁺¹·s⁻¹": api.get_k(
                self.model.scheme,
                self.model.compounds,
                qrrho=self.qrrho,
                temperature=self.temperature,
                scale="atm-1 s-1",
            ),
        }

        kinetics_table = Table(
            *(
                [
                    Column("no", justify="center"),
                    Column("reaction", justify="center"),
                    Column("half equilib.?", justify="center"),
                ]
                + [Column(f"k\n\[{scale}]", justify="center") for scale in k]
            ),
            title="calculated kinetics",
            box=self.box_style,
        )
        for i, reaction in enumerate(self.model.scheme.reactions):
            row = [f"{i:2d}", reaction, None] + [f"{k[scale][i]:7.2g}" for scale in k]
            if self.model.scheme.is_half_equilibrium[i]:
                row[2] = True

            kinetics_table.add_row(*row)
        yield kinetics_table

        if self.quantities is not None and self.quantities:
            scale = "M⁻ⁿ⁺¹·s⁻¹"

            # TODO(schneiderfelipe): apply post-processing to scheme, k (with functions
            # that receive a scheme, k and return a scheme, k). One that solves the pH
            # problem is welcome: get a scheme, k and, for each reaction in it, remove
            # the H+ and multiplies the reaction rate constants by the proper
            # concentration if there is H+ in the reactants.
            # TODO(schneiderfelipe): encapsulate everything in a function that depends
            # on the freeenergies as first parameter
            dydt = api.get_dydt(self.model.scheme, k[scale])

            y0 = np.zeros(len(self.model.scheme.compounds))
            for spec in self.quantities:
                fields = spec.split(":", 1)
                name = fields[0]
                try:
                    quantity = float(fields[1])
                except (IndexError, ValueError):
                    raise ValueError(
                        f"badly formatted quantities: '{' '.join(self.quantities)}'"
                    )

                # TODO(schneiderfelipe): the following is inefficient but probably OK
                y0[self.model.scheme.compounds.index(name)] = quantity

            y, _ = api.get_y(dydt, y0=y0, method="Radau")
            # TODO(schneiderfelipe): we can get a max time now based on the
            # changes through time: stop when the graph gets boring.
            t = np.linspace(y.t_min, y.t_max)
            if self.plot:
                import matplotlib.pyplot as plt

                for i, name in enumerate(self.model.scheme.compounds):
                    if not core.is_transition_state(name):
                        plt.plot(t, y(t)[i], label=name)

                plt.legend()
                plt.xlabel("Time (s)")
                plt.ylabel("Concentration (M)")
                plt.show()

            if self.savepath is not None:
                np.savetxt(
                    self.savepath,
                    np.block([t[:, np.newaxis], y(t).T]),
                    header=f"t,{','.join(self.model.scheme.compounds)}",
                    delimiter=",",
                )
                yield f"CSV file saved to {self.savepath}"


def main():
    """Command-line interface."""
    console = Console(width=max(105, shutil.get_terminal_size()[0]))
    levels = [logging.WARNING, logging.INFO, logging.DEBUG]

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

    console.print(
        Markdown(
            f"""
# overreact {__version__}
Construct precise chemical microkinetic models from first principles

Inputs:
- Path          = {args.path}
- Quantities    = {args.quantities}
- Verbose level = {args.verbose}
- Compile?      = {args.compile}
- Plot?         = {args.plot}
- QRRHO?        = {args.qrrho}
- Temperature   = {args.temperature} K
- Pressure      = {args.pressure} Pa

Parsing and calculating…
            """
        ),
        justify="center",
    )

    logging.basicConfig(
        level=levels[min(len(levels) - 1, args.verbose)], stream=sys.stdout
    )
    for handler in logging.root.handlers:
        handler.setFormatter(io.InterfaceFormatter("%(message)s"))

    model = io.parse_model(args.path, force_compile=args.compile)
    report = Report(
        model,
        quantities=args.quantities,
        savepath=os.path.splitext(args.path)[0] + ".csv",
        plot=args.plot,
        qrrho=args.qrrho,
        temperature=args.temperature,
    )
    console.print(report, justify="center")


if __name__ == "__main__":
    main()
