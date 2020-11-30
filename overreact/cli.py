#!/usr/bin/env python3

"""Command-line interface."""

import shutil
import argparse
import logging
import os
import re
import sys

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize_scalar
from rich import box
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.table import Table
from rich.table import Column
from rich.text import Text

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
    >>> from rich import print
    >>> model = api.parse_model("data/ethane/B97-3c/model.jk")
    >>> print(Report(model))
    ╭──────────────────╮
    │ (read) reactions │
    │                  │
    │   S -> E‡ -> S   │
    │                  │
    ╰──────────────────╯
                       (parsed) reactions
    <BLANKLINE>
      no   reactant(s)   via‡   product(s)   half equilib.?
     ───────────────────────────────────────────────────────
       0   S             E‡     S                  No
    <BLANKLINE>
    ...
    <BLANKLINE>
    Only in the table above, all Gibbs free energies were biased by 0.0 J/mol.
    For half-equilibria, only ratios make sense.
    """

    def __init__(
        self,
        model,
        concentrations=None,
        savepath=None,
        plot=None,
        qrrho=True,
        temperature=298.15,
        bias=0.0,
        tunneling="eckart",
        method="Radau",
        max_time=24 * 60 * 60,
        rtol=1e-5,
        atol=1e-11,
        box_style=box.SIMPLE,
    ):
        self.model = model
        self.concentrations = concentrations
        self.savepath = savepath
        self.plot = plot
        self.qrrho = qrrho
        self.temperature = temperature
        self.bias = bias
        self.tunneling = tunneling
        self.method = method
        self.max_time = max_time
        self.rtol = rtol
        self.atol = atol
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
        raw_table.add_column(justify="left")
        for r in core.unparse_reactions(scheme).split("\n"):
            raw_table.add_row(r)
        yield Panel(raw_table, expand=False)

        transition_states = core.get_transition_states(
            scheme.A, scheme.B, scheme.is_half_equilibrium
        )

        parsed_table = Table(
            Column("no", justify="right"),
            Column("reactant(s)", justify="left"),
            Column("via‡", justify="left"),
            Column("product(s)", justify="left"),
            Column("half equilib.?", justify="center"),
            title="(parsed) reactions",
            box=self.box_style,
        )
        for i, reaction in enumerate(scheme.reactions):
            reactants, _, products = re.split(r"\s*(->|<=>|<-)\s*", reaction)
            row = [f"{i:d}", reactants, None, products, "No"]
            if transition_states[i] is not None:
                row[2] = scheme.compounds[transition_states[i]]
            elif scheme.is_half_equilibrium[i]:
                row[4] = "Yes"
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
            Column("no", justify="right"),
            Column("compound", justify="left"),
            Column("path", justify="left"),
            title="logfiles",
            box=self.box_style,
        )
        compounds_table = Table(
            Column("no", justify="right"),
            Column("compound", justify="left"),
            Column("elec. energy\n\[Eₕ]", justify="center"),
            Column("spin mult.", justify="center"),
            Column("smallest vibfreqs\n\[cm⁻¹]", justify="center"),
            title="compounds",
            box=self.box_style,
        )
        for i, (name, data) in enumerate(self.model.compounds.items()):
            path_text = None
            if data.logfile is not None:
                path_text = Text(data.logfile)
                path_text.highlight_regex(r"[^\/]+$", "bright_blue")
            logfiles_table.add_row(
                f"{i:d}",
                name,
                path_text,
            )

            vibfreqs_text = None
            if data.vibfreqs is not None:
                vibfreqs_text = Text(
                    ", ".join([f"{vibfreq:+7.1f}" for vibfreq in data.vibfreqs[:3]])
                )
                vibfreqs_text.highlight_regex(r"-\d+\.\d", "bright_yellow")
            compounds_table.add_row(
                f"{i:d}",
                name,
                f"{data.energy / (constants.hartree * constants.N_A):17.12f}",
                f"{data.mult}",
                vibfreqs_text,
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
        assert np.allclose(
            freeenergies,
            api.get_freeenergies(
                self.model.compounds,
                qrrho=self.qrrho,
                temperature=self.temperature,
                # pressure=pressure,
            ),
        )

        compounds_table = Table(
            Column("no", justify="right"),
            Column("compound", justify="left"),
            Column("mass\n\[amu]", justify="center"),
            Column("Gᶜᵒʳʳ\n\[kcal/mol]", justify="center", style="bright_green"),
            Column("Uᶜᵒʳʳ\n\[kcal/mol]", justify="center"),
            Column("Hᶜᵒʳʳ\n\[kcal/mol]", justify="center"),
            Column("S\n\[cal/mol·K]", justify="center"),
            title="calculated thermochemistry (compounds)",
            box=self.box_style,
        )
        for i, (name, data) in enumerate(self.model.compounds.items()):
            compounds_table.add_row(
                f"{i:d}",
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
        # TODO(schneiderfelipe): log the contribution of reaction symmetry
        delta_entropies = api.get_delta(
            scheme.A, entropies
        ) + api.get_reaction_entropies(scheme.A)
        delta_freeenergies = delta_enthalpies - self.temperature * delta_entropies
        assert np.allclose(
            delta_freeenergies,
            api.get_delta(scheme.A, freeenergies)
            - self.temperature * api.get_reaction_entropies(scheme.A),
        )

        delta_activation_mass = api.get_delta(scheme.B, molecular_masses)
        delta_activation_energies = api.get_delta(scheme.B, energies)
        delta_activation_internal_energies = api.get_delta(scheme.B, internal_energies)
        delta_activation_enthalpies = api.get_delta(scheme.B, enthalpies)
        # TODO(schneiderfelipe): log the contribution of reaction symmetry
        delta_activation_entropies = api.get_delta(
            scheme.B, entropies
        ) + api.get_reaction_entropies(scheme.B)
        delta_activation_freeenergies = (
            delta_activation_enthalpies - self.temperature * delta_activation_entropies
        )
        assert np.allclose(
            delta_activation_freeenergies,
            api.get_delta(scheme.B, freeenergies)
            - self.temperature * api.get_reaction_entropies(scheme.B),
        )

        circ_table = Table(
            Column("no", justify="right"),
            Column("reaction", justify="left"),
            Column("Δmass°\n\[amu]", justify="center"),
            Column("ΔG°\n\[kcal/mol]", justify="center", style="bright_green"),
            Column("ΔE°\n\[kcal/mol]", justify="center"),
            Column("ΔU°\n\[kcal/mol]", justify="center"),
            Column("ΔH°\n\[kcal/mol]", justify="center"),
            Column("ΔS°\n\[cal/mol·K]", justify="center"),
            title="calculated (reaction°) thermochemistry",
            box=self.box_style,
        )
        dagger_table = Table(
            Column("no", justify="right"),
            Column("reaction", justify="left"),
            Column("Δmass‡\n\[amu]", justify="center"),
            Column("ΔG‡\n\[kcal/mol]", justify="center", style="bright_green"),
            Column("ΔE‡\n\[kcal/mol]", justify="center"),
            Column("ΔU‡\n\[kcal/mol]", justify="center"),
            Column("ΔH‡\n\[kcal/mol]", justify="center"),
            Column("ΔS‡\n\[cal/mol·K]", justify="center"),
            title="calculated (activation‡) thermochemistry",
            box=self.box_style,
        )
        for i, reaction in enumerate(scheme.reactions):
            if scheme.is_half_equilibrium[i]:
                circ_row = [
                    f"{i:d}",
                    reaction,
                    f"{delta_mass[i]:6.2f}",
                    f"{delta_freeenergies[i] / constants.kcal:10.2f}",
                    f"{delta_energies[i] / constants.kcal:10.2f}",
                    f"{delta_internal_energies[i] / constants.kcal:10.2f}",
                    f"{delta_enthalpies[i] / constants.kcal:10.2f}",
                    f"{delta_entropies[i] / constants.calorie:11.2f}",
                ]
                dagger_row = [
                    f"{i:d}",
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
                    f"{i:d}",
                    reaction,
                    f"{delta_mass[i]:6.2f}",
                    f"{delta_freeenergies[i] / constants.kcal:10.2f}",
                    f"{delta_energies[i] / constants.kcal:10.2f}",
                    f"{delta_internal_energies[i] / constants.kcal:10.2f}",
                    f"{delta_enthalpies[i] / constants.kcal:10.2f}",
                    f"{delta_entropies[i] / constants.calorie:11.2f}",
                ]
                dagger_row = [
                    f"{i:d}",
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
        # diffusion control).
        # TODO(schneiderfelipe): use pressure.
        k = {
            "M⁻ⁿ⁺¹·s⁻¹": api.get_k(
                self.model.scheme,
                self.model.compounds,
                bias=self.bias,
                tunneling=self.tunneling,
                qrrho=self.qrrho,
                temperature=self.temperature,
                scale="l mol-1 s-1",
            ),
            "(cm³/particle)ⁿ⁻¹·s⁻¹": api.get_k(
                self.model.scheme,
                self.model.compounds,
                bias=self.bias,
                tunneling=self.tunneling,
                qrrho=self.qrrho,
                temperature=self.temperature,
                scale="cm3 particle-1 s-1",
            ),
            "atm⁻ⁿ⁺¹·s⁻¹": api.get_k(
                self.model.scheme,
                self.model.compounds,
                bias=self.bias,
                tunneling=self.tunneling,
                qrrho=self.qrrho,
                temperature=self.temperature,
                scale="atm-1 s-1",
            ),
        }

        kinetics_table = Table(
            *(
                [
                    Column("no", justify="right"),
                    Column("reaction", justify="left"),
                    Column("half equilib.?", justify="center"),
                ]
                + [Column(f"k\n\[{scale}]", justify="center") for scale in k]
            ),
            title="calculated reaction rate constants",
            box=self.box_style,
        )
        for i, reaction in enumerate(self.model.scheme.reactions):
            row = [f"{i:d}", reaction, "No"] + [f"{k[scale][i]:.3g}" for scale in k]
            if self.model.scheme.is_half_equilibrium[i]:
                row[2] = "Yes"

            kinetics_table.add_row(*row)
        yield kinetics_table
        yield Markdown(
            "Only in the table above, all Gibbs free energies were biased by "
            f"{self.bias} J/mol."
        )
        yield Markdown("For **half-equilibria**, only ratios make sense.")

        if self.concentrations is not None and self.concentrations:
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
            for spec in self.concentrations:
                fields = spec.split(":", 1)
                name = fields[0]
                try:
                    quantity = float(fields[1])
                except (IndexError, ValueError):
                    raise ValueError(
                        "badly formatted concentrations: "
                        f"'{' '.join(self.concentrations)}'"
                    )

                y0[self.model.scheme.compounds.index(name)] = quantity

            y, r = api.get_y(
                dydt,
                y0=y0,
                method=self.method,
                rtol=self.rtol,
                atol=self.atol,
                max_time=self.max_time,
            )
            conc_table = Table(
                Column("no", justify="right"),
                Column("compound", justify="left"),
                Column(f"t = {y.t_min:.1g} s", justify="right"),
                Column(f"t = {y.t_max:.1g} s", justify="right", style="bright_green"),
                title="initial and final concentrations\n\[M]",
                box=self.box_style,
            )
            for i, (name, _) in enumerate(self.model.compounds.items()):
                conc_table.add_row(
                    f"{i:d}",
                    name,
                    f"{y(y.t_min)[i]:.3f}",
                    f"{y(y.t_max)[i]:.3f}",
                )
            yield conc_table

            active = ~np.isclose(y(y.t_min), y(y.t_max), rtol=1e-2)
            if self.plot == "all" or not np.any(active):
                active = np.array([True for _ in self.model.compounds])

            factor = y(y.t_max)[active].max()
            reference = y(y.t_max)[active] / factor

            alpha = 0.9
            n_max = np.log(1e-8) / np.log(alpha)

            t_max, i = y.t_max, 0
            while i < n_max and np.allclose(
                y(t_max)[active] / factor, reference, atol=1e-2
            ):
                t_max = alpha * t_max
                i += 1

            num = 100
            t = set(np.linspace(y.t_min, t_max, num=num))
            for i, name in enumerate(self.model.scheme.compounds):
                if not core.is_transition_state(name):
                    res = minimize_scalar(
                        lambda t: -r(t)[i],
                        bounds=(y.t_min, (t_max + y.t_max) / 2),
                        method="bounded",
                    )
                    if y.t_min < res.x < t_max:
                        t.update(np.linspace(y.t_min, res.x, num=num // 2))
                        t.update(np.linspace(res.x, t_max, num=num // 2))
                        active[i] = True

            t.update(np.geomspace(np.min([_t for _t in t if _t > 0.0]), t_max, num=num))
            t = np.array(sorted(t))
            if self.plot not in {"none", None}:
                if self.plot not in {"all", "active"}:
                    name = self.plot
                    plt.plot(
                        t, y(t)[self.model.scheme.compounds.index(name)], label=name
                    )
                else:
                    for i, name in enumerate(self.model.scheme.compounds):
                        if active[i] and not core.is_transition_state(name):
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
                yield Markdown(f"Simulation data was saved to **{self.savepath}**")


def main():
    """Command-line interface."""
    console = Console(width=max(105, shutil.get_terminal_size()[0]))
    levels = [logging.WARNING, logging.INFO, logging.DEBUG]

    parser = argparse.ArgumentParser(
        description="Interface for building and modifying models."
    )
    parser.add_argument("path", help="path to a source (.k) or model file (.jk)")
    parser.add_argument(
        "concentrations",
        help=(
            "optional initial compound concentrations as 'name:quantity' for "
            "a microkinetic simulation"
        ),
        nargs="*",
    )
    parser.add_argument("-b", "--bias", type=float, default=0.0)
    parser.add_argument(
        "--tunneling",
        help="tunneling method",
        choices=["eckart", "wigner", "none"],
        default="eckart",
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
        "--method",
        help="integrator",
        choices=["BDF", "LSODA", "Radau"],
        default="Radau",
    )
    parser.add_argument("--max-time", type=float, default=24 * 60 * 60)
    parser.add_argument("--rtol", type=float, default=1e-5)
    parser.add_argument("--atol", type=float, default=1e-11)
    parser.add_argument(
        "--plot",
        help=(
            "plot concentrations as a function of time in a microkinetics "
            "simulation for 'none', 'all', 'active' species only (i.e., the "
            "ones that actually change concentration) or a compound name for "
            "a single compound (e.g. 'NH3(w)')"
        ),
        # TODO(schneiderfelipe): validate inputs to avoid "ValueError:
        # tuple.index(x): x not in tuple"
        # choices=["active", "all", "none"],
        default="none",
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
    # TODO(schneiderfelipe): should we consider --compile|-c always as a
    # do-nothing (no analysis)?
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
- Path           = {args.path}
- Concentrations = {args.concentrations}
- Verbose level  = {args.verbose}
- Compile?       = {args.compile}
- Plot?          = {args.plot}
- QRRHO?         = {args.qrrho}
- Temperature    = {args.temperature} K
- Pressure       = {args.pressure} Pa
- Integrator     = {args.method}
- Max. Time      = {args.max_time}
- Rel. Tol.      = {args.rtol}
- Abs. Tol.      = {args.atol}
- Bias           = {args.bias / constants.kcal} kcal/mol
- Tunneling      = {args.tunneling}

Parsing and calculating…
            """
        ),
        justify="left",
    )

    logging.basicConfig(
        level=levels[min(len(levels) - 1, args.verbose)], stream=sys.stdout
    )
    for handler in logging.root.handlers:
        handler.setFormatter(io.InterfaceFormatter("%(message)s"))

    model = io.parse_model(args.path, force_compile=args.compile)
    report = Report(
        model,
        concentrations=args.concentrations,
        savepath=os.path.splitext(args.path)[0] + ".csv",
        plot=args.plot,
        qrrho=args.qrrho,
        temperature=args.temperature,
        bias=args.bias,
        tunneling=args.tunneling,
        method=args.method,
        max_time=args.max_time,
        rtol=args.rtol,
        atol=args.atol,
    )
    # TODO(schneiderfelipe): use a progress bar to inform about the
    # simulation and show the time it took to simulate.
    console.print(report, justify="left")


if __name__ == "__main__":
    main()
