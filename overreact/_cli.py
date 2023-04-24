#!/usr/bin/env python3  # noqa: EXE001

"""Command-line interface."""


from __future__ import annotations

import argparse
import logging
import os
import re
import shutil
import sys

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize_scalar

import overreact as rx
from overreact import _constants as constants
from overreact import coords
from overreact._misc import _check_package, _found_rich, _found_seaborn

logger = logging.getLogger(__name__)


if _found_seaborn:
    import seaborn as sns

    sns.set(style="white", palette="colorblind")
else:
    logger.warning("Install seaborn to get nicer plots: pip install seaborn")


if _found_rich:
    from rich import box, traceback
    from rich.console import Console
    from rich.markdown import Markdown
    from rich.panel import Panel
    from rich.table import Column, Table
    from rich.text import Text

    traceback.install(show_locals=True)
else:
    _check_package("rich", _found_rich, "cli")


class Report:
    """Produce a report object based on a model.

    Parameters
    ----------
    model : dict-like
    qrrho : bool, optional
        Apply both the quasi-rigid rotor harmonic oscillator (QRRHO)
        approximations of M. Head-Gordon and others (enthalpy correction, see
        [*J. Phys. Chem. C* **2015**, 119, 4, 1840-1850](http://dx.doi.org/10.1021/jp509921r))
        and S. Grimme (entropy correction, see
        [*Theory. Chem. Eur. J.*, **2012**, 18: 9955-9964](https://doi.org/10.1002/chem.201200497))
        on top of the classical RRHO.
    temperature : array-like, optional
        Absolute temperature in Kelvin.

    Examples
    --------
    >>> from rich import print
    >>> model = rx.parse_model("data/ethane/B97-3c/model.jk")
    >>> print(Report(model))  # doctest: +SKIP
    ────────────────────────────────────────────────────────────────────────────────
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
                        logfiles
    <BLANKLINE>
    no   compound   path
    ──────────────────────────────────────────────────
    0   S          data/ethane/B97-3c/staggered.out
    1   E‡         data/ethane/B97-3c/eclipsed.out
    <BLANKLINE>
                                    compounds
    <BLANKLINE>
    no   compound    elec. energy     spin mult.      smallest       point group
                        〈Eₕ〉                        vibfreqs
                                                        〈cm⁻¹〉
    ──────────────────────────────────────────────────────────────────────────────
    0   S           -79.788170457…       1            +307.6,           D3d
                                                    +825.4,  +826.1
    1   E‡          -79.783894160…       1            -298.9,           D3h
                                                    +902.2,  +902.5
    <BLANKLINE>
                        estimated thermochemistry (compounds)
    <BLANKLINE>
    no   compound    mass       Gᶜᵒʳʳ        Uᶜᵒʳʳ         Hᶜᵒʳʳ          S
                    〈amu〉   〈kcal/mo…   〈kcal/mo…   〈kcal/mol…   〈cal/mol…
    ──────────────────────────────────────────────────────────────────────────────
    0   S           30.07             …            …            4…        54.40
    1   E‡          30.07             …            …            4…        52.96
    <BLANKLINE>
                        estimated (reaction°) thermochemistry
    <BLANKLINE>
    no   reaction   Δmass°      ΔG°        ΔE°        ΔU°       ΔH°        ΔS°
                    〈amu〉   〈kcal/…   〈kcal/…   〈kcal…   〈kcal/…   〈cal/…
    ──────────────────────────────────────────────────────────────────────────────
    0   S -> S       0.00          0…         0…         …         0…         …
    <BLANKLINE>
                        estimated (activation‡) thermochemistry
    <BLANKLINE>
    no   reaction   Δmass‡      ΔG‡        ΔE‡        ΔU‡       ΔH‡        ΔS‡
                    〈amu〉   〈kcal/…   〈kcal/…   〈kcal…   〈kcal/…   〈cal/…
    ──────────────────────────────────────────────────────────────────────────────
    0   S -> S       0.00          2…         2…         …         2…         …
    <BLANKLINE>
                        estimated reaction rate constants
    <BLANKLINE>
    no   reaction      half            k             k             k         κ
                    equilib.?    〈M⁻ⁿ⁺¹·s⁻…   〈(cm³/par…   〈atm⁻ⁿ⁺¹·…
    ──────────────────────────────────────────────────────────────────────────────
    0   S -> S         No         8.17e+10      8.17e+10      8.17e+10     1.11
    <BLANKLINE>
    Only in the table above, all Gibbs free energies were biased by 0.0 J/mol.
    For half-equilibria, only ratios make sense: in simulations, equilibria will be
    adjusted to be faster than all other reactions.
    ────────────────────────────────────────────────────────────────────────────────
    """

    def __init__(  # noqa: PLR0913
        self,  # noqa: ANN101
        model,
        concentrations=None,
        savepath=None,
        plot=None,
        qrrho_descriptor="both",
        temperature=298.15,
        pressure=constants.atm,
        bias=0.0,
        tunneling="eckart",
        method="RK23",
        max_time=1 * 60 * 60,
        rtol=1e-3,
        atol=1e-6,
        box_style=box.SIMPLE,
    ) -> None:
        """Initialize a Report object."""
        self.model = model
        self.concentrations = concentrations
        self.savepath = savepath
        self.plot = plot
        self.qrrho = {
            "both": (True, True),
            "enthalpy": (True, False),
            "entropy": (False, True),
            "none": (False, False),
        }[qrrho_descriptor]
        self.qrrho_enthalpy, self.qrrho_entropy = self.qrrho

        self.temperature = temperature
        # TODO(schneiderfelipe): use pressure throughout
        self.pressure = pressure
        self.bias = bias
        self.tunneling = tunneling
        self.method = method
        self.max_time = max_time
        self.rtol = rtol
        self.atol = atol
        self.box_style = box_style

    def __rich_console__(self, console, options):  # noqa: ANN101, ANN204
        """
        Implement Rich Console protocol.

        This works by yielding from generators.

        Yields
        ------
        renderable
        """
        yield Markdown("---")
        yield from self._yield_scheme()
        yield from self._yield_compounds()
        yield from self._yield_thermochemistry()
        yield from self._yield_kinetics()
        yield Markdown("---")

    def _yield_scheme(self):  # noqa: ANN101
        """Produce a renderables describing the reaction scheme.

        This is meant to be used from within `__rich_console__`.

        Yields
        ------
        renderable
        """
        scheme = rx.core._check_scheme(self.model.scheme)  # noqa: SLF001

        raw_table = Table(
            title="(read) reactions",
            box=self.box_style,
            show_header=False,
        )
        raw_table.add_column(justify="left")
        for r in rx.core.unparse_reactions(scheme).split("\n"):
            raw_table.add_row(r)
        yield Panel(raw_table, expand=False)

        transition_states = rx.core.get_transition_states(
            scheme.A,
            scheme.B,
            scheme.is_half_equilibrium,
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

    def _yield_compounds(self):  # noqa: ANN101
        """Produce a renderables describing the compounds.

        This is meant to be used from within `__rich_console__`.

        Yields
        ------
        renderable

        Raises
        ------
        ValueError
            If at least one compound has undefined data.
        """
        undefined_compounds = []
        for name in self.model.compounds:
            if not self.model.compounds[name]:
                undefined_compounds.append(name)
        if undefined_compounds:
            raise ValueError(  # noqa: TRY003
                f"undefined compounds: {', '.join(undefined_compounds)}",  # noqa: EM102
            )  # noqa: RUF100

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
            Column("elec. energy\n〈Eₕ〉", justify="center"),
            Column("spin mult.", justify="center"),
            Column("smallest vibfreqs\n〈cm⁻¹〉", justify="center"),
            Column("point group", justify="center"),
            title="compounds",
            box=self.box_style,
        )
        for i, (name, data) in enumerate(self.model.compounds.items()):
            path_text = None
            if data.logfile is not None:
                path_text = Text(data.logfile)
                path_text.highlight_regex(r"[^\/]+$", "bright_blue")
            logfiles_table.add_row(f"{i:d}", name, path_text)

            vibfreqs_text = None
            if data.vibfreqs is not None:
                vibfreqs_text = Text(
                    ", ".join([f"{vibfreq:+7.1f}" for vibfreq in data.vibfreqs[:3]]),
                )
                vibfreqs_text.highlight_regex(r"-\d+\.\d", "bright_yellow")

            point_group = coords.find_point_group(
                atommasses=data.atommasses,
                atomcoords=data.atomcoords,
            )
            compounds_table.add_row(
                f"{i:d}",
                name,
                f"{data.energy / (constants.hartree * constants.N_A):17.12f}",
                f"{data.mult}",
                vibfreqs_text,
                point_group,
            )
        yield logfiles_table
        yield compounds_table

    def _yield_thermochemistry(self):  # noqa: ANN101
        """Produce a renderables describing the thermochemistry of the reaction scheme.

        This is meant to be used from within `__rich_console__`.

        Yields
        ------
        renderable
        """
        scheme = rx.core._check_scheme(self.model.scheme)  # noqa: SLF001

        molecular_masses = np.array(
            [np.sum(data.atommasses) for name, data in self.model.compounds.items()],
        )
        energies = np.array(
            [data.energy for name, data in self.model.compounds.items()],
        )
        internal_energies = rx.get_internal_energies(
            self.model.compounds,
            qrrho=self.qrrho,
            temperature=self.temperature,
        )
        enthalpies = rx.get_enthalpies(
            self.model.compounds,
            qrrho=self.qrrho,
            temperature=self.temperature,
        )
        entropies = rx.get_entropies(
            self.model.compounds,
            qrrho=self.qrrho,
            temperature=self.temperature,
        )
        freeenergies = enthalpies - self.temperature * entropies
        assert np.allclose(
            freeenergies,
            rx.get_freeenergies(
                self.model.compounds,
                qrrho=self.qrrho,
                temperature=self.temperature,
                pressure=self.pressure,
            ),
        ), "free energies do not match enthalpies and entropies"

        compounds_table = Table(
            Column("no", justify="right"),
            Column("compound", justify="left"),
            Column("mass\n〈amu〉", justify="center"),
            Column("Gᶜᵒʳʳ\n〈kcal/mol〉", justify="center", style="bright_green"),
            Column("Uᶜᵒʳʳ\n〈kcal/mol〉", justify="center"),
            Column("Hᶜᵒʳʳ\n〈kcal/mol〉", justify="center"),
            Column("S\n〈cal/mol·K〉", justify="center"),
            title="estimated thermochemistry (compounds)",
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

        delta_mass = rx.get_delta(scheme.A, molecular_masses)
        delta_energies = rx.get_delta(scheme.A, energies)
        delta_internal_energies = rx.get_delta(scheme.A, internal_energies)
        delta_enthalpies = rx.get_delta(scheme.A, enthalpies)
        # TODO(schneiderfelipe): log the contribution of reaction symmetry
        delta_entropies = rx.get_delta(scheme.A, entropies) + rx.get_reaction_entropies(
            scheme.A,
            temperature=self.temperature,
            pressure=self.pressure,
        )
        delta_freeenergies = delta_enthalpies - self.temperature * delta_entropies
        assert np.allclose(
            delta_freeenergies,
            rx.get_delta(scheme.A, freeenergies)
            - self.temperature
            * rx.get_reaction_entropies(
                scheme.A,
                temperature=self.temperature,
                pressure=self.pressure,
            ),
        ), "reaction free energies do not match reaction enthalpies and reaction entropies"  # noqa: E501

        delta_activation_mass = rx.get_delta(scheme.B, molecular_masses)
        delta_activation_energies = rx.get_delta(scheme.B, energies)
        delta_activation_internal_energies = rx.get_delta(scheme.B, internal_energies)
        delta_activation_enthalpies = rx.get_delta(scheme.B, enthalpies)
        # TODO(schneiderfelipe): log the contribution of reaction symmetry
        delta_activation_entropies = rx.get_delta(
            scheme.B,
            entropies,
        ) + rx.get_reaction_entropies(
            scheme.B,
            temperature=self.temperature,
            pressure=self.pressure,
        )
        delta_activation_freeenergies = (
            delta_activation_enthalpies - self.temperature * delta_activation_entropies
        )
        assert np.allclose(
            delta_activation_freeenergies,
            rx.get_delta(scheme.B, freeenergies)
            - self.temperature
            * rx.get_reaction_entropies(
                scheme.B,
                temperature=self.temperature,
                pressure=self.pressure,
            ),
        ), "activation free energies do not match activation enthalpies and activation entropies"  # noqa: E501

        circ_table = Table(
            Column("no", justify="right"),
            Column("reaction", justify="left"),
            Column("Δmass°\n〈amu〉", justify="center"),
            Column("ΔG°\n〈kcal/mol〉", justify="center", style="bright_green"),
            Column("ΔE°\n〈kcal/mol〉", justify="center"),
            Column("ΔU°\n〈kcal/mol〉", justify="center"),
            Column("ΔH°\n〈kcal/mol〉", justify="center"),
            Column("ΔS°\n〈cal/mol·K〉", justify="center"),
            title="estimated (reaction°) thermochemistry",
            box=self.box_style,
        )
        dagger_table = Table(
            Column("no", justify="right"),
            Column("reaction", justify="left"),
            Column("Δmass‡\n〈amu〉", justify="center"),
            Column("ΔG‡\n〈kcal/mol〉", justify="center", style="bright_green"),
            Column("ΔE‡\n〈kcal/mol〉", justify="center"),
            Column("ΔU‡\n〈kcal/mol〉", justify="center"),
            Column("ΔH‡\n〈kcal/mol〉", justify="center"),
            Column("ΔS‡\n〈cal/mol·K〉", justify="center"),
            title="estimated (activation‡) thermochemistry",
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

    def _yield_kinetics(self):  # noqa: ANN101, C901, PLR0912, PLR0915
        """Produce a renderables describing the kinetics of the system.

        This is meant to be used from within `__rich_console__`.

        Yields
        ------
        renderable
        """
        if isinstance(self.bias, str):
            data = np.genfromtxt(self.bias, names=True, delimiter=",")
            data = {name: data[name] for name in data.dtype.names}

            scheme, _, y0 = _prepare_simulation(
                self.model.scheme,
                rx.get_k(
                    self.model.scheme,
                    self.model.compounds,
                    bias=0.0,
                    tunneling=self.tunneling,
                    qrrho=self.qrrho,
                    scale="l mol-1 s-1",
                    temperature=self.temperature,
                    pressure=self.pressure,
                ),
                self.concentrations,
            )
            self.bias = rx.get_bias(
                scheme,
                self.model.compounds,
                data,
                y0,
                tunneling=self.tunneling,
                qrrho=self.qrrho,
                temperature=self.temperature,
                pressure=self.pressure,
                method=self.method,
                rtol=self.rtol,
                atol=self.atol,
            )

        # TODO(schneiderfelipe): apply other corrections to k (such as
        # diffusion control).
        # TODO(schneiderfelipe): use pressure.
        k = {
            "M⁻ⁿ⁺¹·s⁻¹": rx.get_k(
                self.model.scheme,
                self.model.compounds,
                bias=self.bias,
                tunneling=self.tunneling,
                qrrho=self.qrrho,
                scale="l mol-1 s-1",
                temperature=self.temperature,
                pressure=self.pressure,
            ),
            "(cm³/particle)ⁿ⁻¹·s⁻¹": rx.get_k(
                self.model.scheme,
                self.model.compounds,
                bias=self.bias,
                tunneling=self.tunneling,
                qrrho=self.qrrho,
                scale="cm3 particle-1 s-1",
                temperature=self.temperature,
                pressure=self.pressure,
            ),
            "atm⁻ⁿ⁺¹·s⁻¹": rx.get_k(
                self.model.scheme,
                self.model.compounds,
                bias=self.bias,
                tunneling=self.tunneling,
                qrrho=self.qrrho,
                scale="atm-1 s-1",
                temperature=self.temperature,
                pressure=self.pressure,
            ),
        }
        kappa = rx.get_kappa(
            self.model.scheme,
            self.model.compounds,
            method=self.tunneling,
            qrrho=self.qrrho,
            temperature=self.temperature,
        )

        kinetics_table = Table(
            *(
                [
                    Column("no", justify="right"),
                    Column("reaction", justify="left"),
                    Column("half equilib.?", justify="center"),
                ]
                + [Column(f"k\n〈{scale}〉", justify="center") for scale in k]
                + [Column("κ", justify="center")]
            ),
            title="estimated reaction rate constants",
            box=self.box_style,
        )
        for i, reaction in enumerate(self.model.scheme.reactions):
            row = (
                [f"{i:d}", reaction, "No"]
                + [f"{k[scale][i]:.3g}" for scale in k]
                + [f"{kappa[i]:.3g}"]
            )
            if self.model.scheme.is_half_equilibrium[i]:
                row[2] = "Yes"
                row[-1] = None  # hide transmission coefficient

            kinetics_table.add_row(*row)
        yield kinetics_table
        yield Markdown(
            "Only in the table above, all Gibbs free energies were biased by "
            f"{self.bias} J/mol.",
        )
        yield Markdown(
            "For **half-equilibria**, only ratios make sense: in simulations, **equilibria will be adjusted to be faster than all other reactions**.",  # noqa: E501
        )

        if self.concentrations is not None and self.concentrations:
            scheme, k, y0 = _prepare_simulation(
                self.model.scheme,
                k["M⁻ⁿ⁺¹·s⁻¹"],
                self.concentrations,
            )

            # TODO(schneiderfelipe): encapsulate everything in a function that depends
            # on the freeenergies as first parameter
            dydt = rx.get_dydt(scheme, k)

            y, r = rx.get_y(
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
                Column(
                    f"t = {y.t_max:.1g} s",
                    justify="right",
                    style="bright_green",
                ),
                title="initial and final concentrations\n〈M〉",
                box=self.box_style,
            )
            for i, name in enumerate(scheme.compounds):
                conc_table.add_row(
                    f"{i:d}",
                    name,
                    f"{y(y.t_min)[i]:.3f}",
                    f"{y(y.t_max)[i]:.3f}",
                )
            yield conc_table

            t_span = y.t_max - y.t_min
            active = ~np.isclose(
                y(y.t_min + 0.01 * t_span * np.random.rand()),  # noqa: NPY002
                y(y.t_max - 0.01 * t_span * np.random.rand()),  # noqa: NPY002
                rtol=0.01,
            )
            if self.plot == "all" or not np.any(active):
                active = np.array([True for _ in scheme.compounds])

            factor = y(y.t_max)[active].max()
            reference = y(y.t_max)[active] / factor

            step = 0.9  # multiply by this factor to decrease t_max
            n_max = np.log(1e-8) / np.log(step)

            # We plot until concentrations are within this value from the
            # final concentrations.
            alpha = 0.01

            t_max, i = y.t_max, 0
            while i < n_max and np.allclose(
                y(t_max)[active] / factor,
                reference,
                atol=alpha,
            ):
                t_max = step * t_max
                i += 1

            num = 100
            t = set(np.linspace(y.t_min, t_max, num=num))
            for i, name in enumerate(scheme.compounds):
                if not rx.is_transition_state(name):
                    res = minimize_scalar(
                        lambda t: -r(t)[i],  # noqa: B023
                        bounds=(y.t_min, (t_max + y.t_max) / 2),
                        method="bounded",
                    )
                    if y.t_min < res.x < t_max:
                        t.update(np.linspace(y.t_min, res.x, num=num // 2))
                        t.update(np.linspace(res.x, t_max, num=num // 2))
                        active[i] = True

            t.update(
                np.geomspace(
                    np.min([_t for _t in t if _t > 0.0]),  # noqa: PLR2004
                    t_max,
                    num=num,
                ),
            )
            t = np.array(sorted(t))
            if self.plot not in {"none", None}:
                if self.plot not in {"all", "active"}:
                    name = self.plot
                    plt.plot(t, y(t)[scheme.compounds.index(name)], label=name, lw=3)
                else:
                    for i, name in enumerate(scheme.compounds):
                        if active[i] and not rx.is_transition_state(name):
                            plt.plot(t, y(t)[i], label=name, lw=3)

                plt.legend()
                plt.xlabel("Time (s)")
                plt.ylabel("Concentration (M)")
                plt.show()

            if self.savepath is not None:
                np.savetxt(
                    self.savepath,
                    np.block([t[:, np.newaxis], y(t).T]),
                    header=f"t,{','.join(scheme.compounds)}",
                    delimiter=",",
                )
                yield Markdown(f"Simulation data was saved to **{self.savepath}**")


def _prepare_simulation(scheme, k, concentrations):
    """
    Help prepare some data before the simulation.

    Raises
    ------
    ValueError
        If concentrations are invalid or inconsistent with the scheme.
    """
    free_y0 = {}
    fixed_y0 = {}
    for spec in concentrations:
        fields = spec.split(":", 1)
        name, quantity = fields[0].strip(), fields[1].strip()

        if quantity.startswith("!"):
            d = fixed_y0
            quantity = quantity[1:]
        else:
            d = free_y0

        try:
            quantity = float(quantity)
        except (IndexError, ValueError):
            raise ValueError(  # noqa: B904, TRY003, TRY200
                "badly formatted concentrations: "  # noqa: EM102
                f"'{' '.join(concentrations)}'",  # noqa: RUF100
            )

        d[name] = quantity

    # TODO(schneiderfelipe): log stuff related to get_fixed_scheme
    scheme, k = rx.get_fixed_scheme(scheme, k, fixed_y0)

    y0 = np.zeros(len(scheme.compounds))
    for compound in free_y0:
        y0[scheme.compounds.index(compound)] = free_y0[compound]

    return scheme, k, y0


def main(arguments=None):
    """Command-line interface."""
    console = Console(width=max(105, shutil.get_terminal_size()[0]))
    levels = [logging.WARNING, logging.INFO, logging.DEBUG]

    # TODO(schneiderfelipe): some commands for concatenating/summing `.k`/`.jk`
    # files. This might be useful for some of the more complex operations I
    # want to be able to do in the future.
    parser = argparse.ArgumentParser(
        description=f"""
        {rx.__headline__}
        Read the user guide at {rx.__url_guide__} for more information and usage
        examples.

        Licensed under the terms of the {rx.__license__} License.
        If you publish work using this software, please cite
        https://doi.org/{rx.__doi__} and https://doi.org/{rx.__zenodo_doi__}.
        """,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "path",
        help="path to a source (`.k`) or compiled (`.jk`) model input file (if a source "  # noqa: E501
        "input file is given, but there is a compiled file available, the compiled "
        "file will be used; use --compile|-c to force recompilation of the "
        "source input file instead)",
    )
    parser.add_argument(
        "concentrations",
        help="(optional) initial compound concentrations (in moles per liter) "
        "in the form 'name:quantity' (if present, a microkinetic simulation "
        "will be performed; more than one entry can be given)",
        nargs="*",
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {rx.__version__}",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        help="increase output verbosity (can be given many times, each time "
        "the amount of logged data is increased)",
        action="count",
        default=0,
    )
    parser.add_argument(
        "-c",
        "--compile",
        # TODO(schneiderfelipe): should we consider --compile|-c always as a
        # do-nothing (no analysis)?
        help="force recompile a source (`.k`) into a compiled (`.jk`) model input file",
        action="store_true",
    )
    parser.add_argument(
        "--plot",
        help="plot the concentrations as a function of time from the "
        "performed microkinetics simulation: can be either 'none', 'all', "
        "'active' species only (i.e., the ones that actually change "
        "concentration during the simulation) or a single compound name (e.g. "
        "'NH3(w)')",
        # TODO(schneiderfelipe): validate inputs to avoid ValueError?
        default="none",
    )
    parser.add_argument(
        "-b",
        "--bias",
        help="an energy value (in kilocalories per mole) to be added to each "
        "individual compound in order to mitigate eventual systematic errors",
        default=0.0,
    )
    parser.add_argument(
        "--tunneling",
        help="specify the tunneling method employed (use --tunneling=none for "
        "no tunneling correction)",
        choices=["eckart", "wigner", "none"],
        default="eckart",
    )
    parser.add_argument(
        "--no-qrrho",
        help="disable the quasi-rigid rotor harmonic oscillator (QRRHO) "
        "approximations to both enthalpies and entropies (see "
        "doi:10.1021/jp509921r and doi:10.1002/chem.201200497)",
        choices=["both", "enthalpy", "entropy", "none"],
        default="both",
        dest="qrrho_descriptor",
    )
    parser.add_argument(
        "-T",
        "--temperature",
        help="set working temperature (in kelvins) to be used in "
        "thermochemistry and microkinetics",
        type=float,
        default=298.15,
    )
    parser.add_argument(
        "-p",
        "--pressure",
        help="set working pressure (in pascals) to be used in "
        "thermochemistry",  # noqa: RUF100
        type=float,
        default=constants.atm,
    )
    parser.add_argument(
        "--method",
        help="integrator used in solving the ODE system of the microkinetic "
        "simulation",
        choices=["RK23", "DOP853", "RK45", "LSODA", "BDF", "Radau"],
        default="RK23",
    )
    parser.add_argument(
        "--max-time",
        help="maximum microkinetic simulation time (in s) allowed",
        type=float,
        default=1 * 60 * 60,
    )
    parser.add_argument(
        "--rtol",
        help="relative local error of the ODE system integrator",
        type=float,
        default=1e-3,
    )
    parser.add_argument(
        "--atol",
        help="absolute local error of the ODE system integrator",
        type=float,
        default=1e-6,
    )
    args = parser.parse_args(arguments)

    try:
        args.bias = float(args.bias) * constants.kcal
        bias_message = f"{args.bias} kcal/mol"
    except ValueError:
        bias_message = f"fitting from {args.bias}"

    console.print(
        Markdown(
            f"""
# overreact {rx.__version__}

{rx.__headline__}

Licensed under the terms of the
[{rx.__license__} License]({rx.__url_repo__}/blob/main/LICENSE).
If you publish work using this software, **please cite
[doi:{rx.__doi__}](https://doi.org/{rx.__doi__})
*and*
[doi:{rx.__zenodo_doi__}](https://doi.org/{rx.__zenodo_doi__})**:

```
{rx.__citations__}
```

Read the user guide at [{rx.__url_guide__}]({rx.__url_guide__}) for more information
and usage examples. Other useful resources:

- [Questions and Discussions]({rx.__url_repo__}/discussions)
- [Bug Tracker]({rx.__url_repo__}/issues)
- [GitHub Repository]({rx.__url_repo__})
- [Python Package Index]({rx.__url_pypi__})

---

Inputs:
- Path           = {args.path}
- Concentrations = {args.concentrations}
- Verbose level  = {args.verbose}
- Compile?       = {args.compile}
- Plot?          = {args.plot}
- QRRHO?         = {args.qrrho_descriptor}
- Temperature    = {args.temperature} K
- Pressure       = {args.pressure} Pa
- Integrator     = {args.method}
- Max. Time      = {args.max_time}
- Rel. Tol.      = {args.rtol}
- Abs. Tol.      = {args.atol}
- Bias           = {bias_message}
- Tunneling      = {args.tunneling}

Parsing and calculating (this may take a while)…
            """,
        ),
        justify="left",
    )

    logging.basicConfig(
        level=levels[min(len(levels) - 1, args.verbose)],
        stream=sys.stdout,
    )
    for handler in logging.root.handlers:
        handler.setFormatter(rx.io.InterfaceFormatter("%(message)s"))

    model = rx.io.parse_model(args.path, force_compile=args.compile)
    report = Report(
        model,
        concentrations=args.concentrations,
        savepath=f"{os.path.splitext(args.path)[0]}.csv",  # noqa: PTH122
        plot=args.plot,
        qrrho_descriptor=args.qrrho_descriptor,
        temperature=args.temperature,
        pressure=args.pressure,
        bias=args.bias,
        tunneling=args.tunneling,
        method=args.method,
        max_time=args.max_time,
        rtol=args.rtol,
        atol=args.atol,
    )
    console.print(report, justify="left")


if __name__ == "__main__":
    # TODO: catch exceptions here and print a nice error
    # (we can reraise them later if we want).
    main()
