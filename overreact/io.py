#!/usr/bin/env python3  # noqa: EXE001

"""Basic I/O operations (such as reading source **input files**)."""


__all__ = ["parse_model"]


import json
import logging
import os
import textwrap
import warnings
from collections import defaultdict
from collections.abc import MutableMapping

import numpy as np

import overreact as rx
from overreact import _constants as constants

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    from cclib import ccopen

logger = logging.getLogger(__name__)


def parse_model(path: str, force_compile: bool = False):  # noqa: FBT001, FBT002
    """Parse either a source or model input file, whichever is available.

    A **source input file** (also known as a `.k` file) contains all the information needed
    to *create a model input file*.
    A **model input file** (also known as a `.jk` file) is a JSON encoded file with all the
    information needed to study microkinetic simulations from first principles.

    You probably won't need to use model input files directly, they are
    automatically created based on source input files.
    [**Take a look at our guide on how to write an source input file**](https://geem-lab.github.io/overreact-guide/input.html).

    This function attempts to parse a model input file if available. If not, a source
    input file is parsed and a model input file is generated from it. Extensions are
    guessed if none given (i.e., if only the base name given).

    Parameters
    ----------
    path : str
        Path to the model or source input file.
        If the final extension is not `.jk` or `.k`, it is guessed.
    force_compile : bool
        If True, a `.k` file will take precedence over any `.jk` file for reading. A
        `.jk` file is thus either generated or overwritten. This is sometimes
        needed to force an update with new data.

    Returns
    -------
    model : immutable dict-like

    Raises
    ------
    FileNotFoundError
        If the model or source input file is not found.

    Examples
    --------
    Some examples of how overreact "sees" your data below 😄:

    >>> model = parse_model("data/ethane/B97-3c/model.jk")
    >>> model.scheme
    Scheme(compounds=('S', 'E‡'),
           reactions=('S -> S',),
           is_half_equilibrium=(False,),
           A=((0.0,), (0.0,)),
           B=((-1.0,), (1.0,)))
    >>> model.compounds["S"]
    {'logfile': 'data/ethane/B97-3c/staggered.out',
     'energy': -209483812.77142256,
     'mult': 1,
     'atomnos': (6, 6, 1, 1, 1, 1, 1, 1),
     'atommasses': (12.011, 12.011, 1.008, 1.008, 1.008, 1.008, 1.008, 1.008),
     'atomcoords': ((-7.633588, 2.520693, -4.8e-05),
                    ...,
                    (-5.832852, 3.674431, 0.363239)),
     'vibfreqs': (307.57, 825.42, ..., 3071.11, 3071.45),
     'vibdisps': (((-1.7e-05, 3.4e-05, 5.4e-05),
                   ...,
                   (-0.011061, -0.030431, -0.027036)))}
    >>> model_from_source = parse_model("data/ethane/B97-3c/model.k",
    ...                                 force_compile=True)
    >>> model_from_source == model
    True
    >>> model_from_source = parse_model("data/ethane/B97-3c/model")
    >>> model_from_source == model
    True
    """  # noqa: E501
    if not path.endswith((".k", ".jk")):
        path = f"{path}.jk"
        logger.warning(f"assuming `.jk` file in {path}")  # noqa: G004
    name, _ = os.path.splitext(path)  # noqa: PTH122

    path_jk = f"{name}.jk"
    if not force_compile and os.path.isfile(path_jk):  # noqa: PTH113
        logger.info(f"parsing `.jk` file in {path_jk}")  # noqa: G004
        return _parse_model(path_jk)

    path_k = f"{name}.k"
    logger.info(f"parsing `.k` file in {path_k}")  # noqa: G004
    if not os.path.isfile(path_k):  # noqa: PTH113
        # TODO: add a nice error message here and everywhere?
        raise FileNotFoundError(  # noqa: TRY003
            f"no `.k` file found in {path_k}",  # noqa: EM102
        )

    model = _parse_source(path_k)
    with open(path_jk, "w") as f:  # noqa: PTH123
        logger.info(f"writing `.jk` file in {path_jk}")  # noqa: G004
        f.write(_unparse_model(model))

    return model


def _parse_model(file_or_path):
    """Parse a model input file (also known as a `.jk` file).

    A model input file is a JSON encoded file with all the information needed to
    study microkinetic simulations from first principles.

    Parameters
    ----------
    file_or_path : file or str

    Returns
    -------
    model : immutable dict-like

    Examples
    --------
    >>> model = _parse_model("data/ethane/B97-3c/model.jk")
    >>> model.scheme
    Scheme(compounds=('S', 'E‡'),
           reactions=('S -> S',),
           is_half_equilibrium=(False,),
           A=((1.,), (0.,)),
           B=((-1.,), (1.,)))
    >>> model.compounds["S"]
    {'logfile': 'data/ethane/B97-3c/staggered.out',
     'energy': -209483812.77142256,
     'mult': 1,
     'atomnos': (6, 6, 1, 1, 1, 1, 1, 1),
     'atommasses': (12.011, 12.011, 1.008, 1.008, 1.008, 1.008, 1.008, 1.008),
     'atomcoords': ((-7.633588, 2.520693, -4.8e-05),
                    ...,
                    (-5.832852, 3.674431, 0.363239)),
     'vibfreqs': (307.57, 825.42, ..., 3071.11, 3071.45),
     'vibdisps': (((-1.7e-05, 3.4e-05, 5.4e-05),
                   ...,
                   (-0.011061, -0.030431, -0.027036)))}
    >>> model.compounds["E‡"]
    {'logfile': 'data/ethane/B97-3c/eclipsed.out',
     'energy': -209472585.3539883,
     'mult': 1,
     'atomnos': (6, 6, 1, 1, 1, 1, 1, 1),
     'atommasses': (12.011, 12.011, 1.008, 1.008, 1.008, 1.008, 1.008, 1.008),
     'atomcoords': ((-7.640622, 2.51993, -1.6e-05),
                    ...,
                    (-5.730333, 2.893778, 0.996894)),
     'vibfreqs': (-298.94, 902.19, ..., 3077.75, 3078.05),
     'vibdisps': (((-6.7e-05, 2.3e-05, 3.4e-05),
                   ...,
                   (0.133315, 0.086028, 0.35746)))}
    """
    try:
        model = json.load(file_or_path)
    except AttributeError:
        with open(file_or_path) as stream:  # noqa: PTH123
            model = json.load(stream)

    if "scheme" in model:
        model["scheme"] = rx.parse_reactions(model["scheme"])

    return dotdict(model)


def _parse_source(file_path_or_str):
    """Parse a source input file (also known as a `.k` file).

    A source input file contains all the information needed to create a model input file
    (also known as a `.jk` file).

    Parameters
    ----------
    file_path_or_str : file or str

    Returns
    -------
    model : immutable dict-like

    Notes
    -----
    Compounds not cited in the reaction scheme are always ignored.

    Examples
    --------
    >>> model = _parse_source("data/ethane/B97-3c/model.k")
    >>> model.scheme
    Scheme(compounds=('S', 'E‡'),
           reactions=('S -> S',),
           is_half_equilibrium=(False,),
           A=((0.0,),
              (0.0,)),
           B=((-1.0,),
              (1.0,)))
    >>> print(_unparse_model(model))
    {"scheme":["S -> E‡ -> S"],"compounds":{"S":{"logfile":"da...,0.35746]]]}}}

    It is guaranteed that both the list of compounds in the scheme and in the
    keys of compounds match. This implementation detail is crucial for the
    proper internal behavior of overreact:

    >>> model = parse_model("data/perez-soto2020/RI/B3LYP-D3BJ/cc-pVTZ/model.k",
    ...                     force_compile=True)
    >>> model.scheme.compounds
    ('Benzaldehyde(dcm)', 'NButylamine(dcm)', 'A_N(dcm)', 'A_N_N(dcm)',
     'Water(dcm)', 'A_N_W(dcm)', 'A_N_N_W(dcm)', 'A_N_W_W(dcm)', 'TS1_#(dcm)',
     'Hemiaminal(dcm)', 'TS2_#(dcm)', 'I_W(dcm)', 'TS1N_#(dcm)', 'Int_N(dcm)',
     'TS2N_#(dcm)', 'I_N_W(dcm)', 'TS1W_#(dcm)', 'Int_W(dcm)', 'TS2W_#(dcm)',
     'I_W_W(dcm)', 'TS1NW_#(dcm)', 'Int_N_W(dcm)', 'TS2NW_#(dcm)',
     'I_N_W_W(dcm)', 'TS1WW_#(dcm)', 'Int_W_W(dcm)', 'TS2WW_#(dcm)',
     'I_W_W_W(dcm)', 'Imine(dcm)')
    >>> tuple(model.compounds) == model.scheme.compounds
    True
    """
    name = None
    sections = defaultdict(list)

    path = ("",)
    try:
        with open(file_path_or_str) as stream:  # noqa: PTH123
            lines = stream.readlines()
        dirname = os.path.dirname(file_path_or_str)  # noqa: PTH120
        if dirname not in path:
            path = (*path, dirname)
    except OSError:
        lines = file_path_or_str.split("\n")
    except TypeError:
        lines = file_path_or_str

    for line in lines:
        line = line.split("//")[0].strip()  # noqa: PLW2901
        if not line:
            continue

        if line[0] == "$":
            name = None if line[1:] == "end" else line[1:]
        elif name is not None:
            sections[name].append(line)

    if "scheme" in sections:
        sections["scheme"] = rx.parse_reactions(sections["scheme"])
    if "compounds" in sections:
        sections["compounds"] = parse_compounds(
            sections["compounds"],
            path=path,
            select=sections["scheme"].compounds,
        )
    return dotdict(sections)


def _unparse_source(model):
    """Unparse a source input file (also known as a `.k` file).

    A source input file contains all the information needed to create a model input file
    (also known as a `.jk` file).

    Parameters
    ----------
    model : immutable dict-like

    Returns
    -------
    source : str

    Examples
    --------
    >>> model = _parse_model("data/ethane/B97-3c/model.jk")
    >>> source = _unparse_source(model)
    >>> print(source)
    // generated by overreact
    $scheme
     S -> E‡ -> S
    $end
    $compounds
     S:
      logfile="data/ethane/B97-3c/staggered.out"
      energy=-209483812.77142256
      mult=1
      atomnos=[6, 6, 1, 1, 1, 1, 1, 1]
      atommasses=[12.011, 12.011, 1.008, 1.008, 1.008, 1.008, 1.008, 1.008]
      atomcoords=[[-7.633588, 2.520693, -4.8e-05], ...]
      vibfreqs=[307.57, 825.42, ..., 3071.11, 3071.45]
      vibdisps=[[[-1.7e-05, 3.4e-05, 5.4e-05], ..., [..., -0.027036]]]
     E‡:
      logfile="data/ethane/B97-3c/eclipsed.out"
      energy=-209472585.3539883
      mult=1
      ...
    $end
    <BLANKLINE>

    This is useful for representing the same model as an alternative source:

    >>> model = _parse_source(source)
    >>> model.scheme
    Scheme(compounds=('S', 'E‡'),
           reactions=('S -> S',),
           is_half_equilibrium=(False,),
           A=((1.,), (0.,)),
           B=((-1.,), (1.,)))
    >>> model.compounds["S"]
    {'logfile': 'data/ethane/B97-3c/staggered.out',
     'energy': -209483812.77142256,
     'mult': 1,
     'atomnos': (6, 6, 1, 1, 1, 1, 1, 1),
     'atommasses': (12.011, 12.011, 1.008, 1.008, 1.008, 1.008, 1.008, 1.008),
     'atomcoords': ((-7.633588, 2.520693, -4.8e-05),
                    ...,
                    (-5.832852, 3.674431, 0.363239)),
     'vibfreqs': (307.57, 825.42, ..., 3071.11, 3071.45),
     'vibdisps': (((-1.7e-05, 3.4e-05, 5.4e-05),
                   ...,
                   (-0.011061, -0.030431, -0.027036)))}
    >>> model.compounds["E‡"]
    {'logfile': 'data/ethane/B97-3c/eclipsed.out',
     'energy': -209472585.3539883,
     'mult': 1,
     'atomnos': (6, 6, 1, 1, 1, 1, 1, 1),
     'atommasses': (12.011, 12.011, 1.008, 1.008, 1.008, 1.008, 1.008, 1.008),
     'atomcoords': ((-7.640622, 2.51993, -1.6e-05),
                    ...,
                    (-5.730333, 2.893778, 0.996894)),
     'vibfreqs': (-298.94, 902.19, ..., 3077.75, 3078.05),
     'vibdisps': (((-6.7e-05, 2.3e-05, 3.4e-05),
                   ...,
                   (0.133315, 0.086028, 0.35746)))}
    """
    source = "// generated by overreact\n"
    if "scheme" in model:
        source += "$scheme\n"
        for line in rx.unparse_reactions(model.scheme).split("\n"):
            source += f" {line}\n"
        source += "$end\n"
    if "compounds" in model:
        source += "$compounds\n"
        for compound in model.compounds:
            source += f" {compound}:\n"
            for key in model.compounds[compound]:
                inline_json = json.dumps(
                    model.compounds[compound][key],
                    ensure_ascii=False,
                )
                source += f"  {key}={inline_json}\n"
        source += "$end\n"
    return source


def _unparse_model(model):
    """Unparse a model input file (also known as a `.jk` file).

    A model input file is a JSON encoded file with all the information needed to
    study microkinetic simulations from first principles.

    Parameters
    ----------
    model : immutable dict-like

    Returns
    -------
    json : str

    Examples
    --------
    >>> model = _parse_model("data/hickel1992/UM06-2X/6-311++G(d,p)/model.jk")
    >>> print(_unparse_model(model))
    {"scheme":["NH3(w) + OH·(w) -> NH3·OH#(w...,"atomcoords":[[0.0,0.0,0.0]]}}}
    """
    # create a new mutable object to avoid side effects
    model = dict(model.copy())

    if "scheme" in model:
        model["scheme"] = rx.unparse_reactions(model["scheme"]).split("\n")
    return json.dumps(model, ensure_ascii=False, separators=(",", ":"))


def _check_compounds(compounds):
    """Complete data from logfiles if missing in compounds dict-like.

    Parameters
    ----------
    compounds : dict-like
        A descriptor of the compounds.
        Mostly likely, this comes from a parsed model input file.
        See `overreact.io.parse_model`.

    Returns
    -------
    dict-like

    Examples
    --------
    >>> _check_compounds({"S": "data/ethane/B97-3c/staggered.out"})
    {'S': {'logfile': 'data/ethane/B97-3c/staggered.out',
           'energy': -209483812.77142256,
           'mult': 1,
           'atomnos': (6, 6, 1, 1, 1, 1, 1, 1),
           'atommasses': (12.011, 12.011, 1.008, 1.008, 1.008, 1.008, 1.008, 1.008),
           'atomcoords': ((-7.633588, 2.520693, -4.8e-05),
                           ...,
                          (-5.832852, 3.674431, 0.363239)),
           'vibfreqs': (307.57, 825.42, ..., 3071.11, 3071.45),
           'vibdisps': (((-1.7e-05, 3.4e-05, 5.4e-05),
                          ...,
                         (-0.011061, -0.030431, -0.027036)))}}
    >>> _check_compounds(_check_compounds({"S": "data/ethane/B97-3c/staggered.out"}))
    {'S': {'logfile': 'data/ethane/B97-3c/staggered.out',
           'energy': -209483812.77142256,
           'mult': 1,
           'atomnos': (6, 6, 1, 1, 1, 1, 1, 1),
           'atommasses': (12.011, 12.011, 1.008, 1.008, 1.008, 1.008, 1.008, 1.008),
           'atomcoords': ((-7.633588, 2.520693, -4.8e-05),
                           ...,
                          (-5.832852, 3.674431, 0.363239)),
           'vibfreqs': (307.57, 825.42, ..., 3071.11, 3071.45),
           'vibdisps': (((-1.7e-05, 3.4e-05, 5.4e-05),
                          ...,
                         (-0.011061, -0.030431, -0.027036)))}}
    """
    for name in compounds:
        if isinstance(compounds[name], str):
            compounds[name] = read_logfile(compounds[name])
    return dict(compounds)


def parse_compounds(text, path=("",), select=None):  # noqa: C901, PLR0912
    """Parse a set of compounds.

    Parameters
    ----------
    text : str, sequence of str or dict-like
        Compound descriptions or sequence of lines of it.
    path : sequence of str, optional
        Paths for searching logfiles.
    select : sequence of str, optional
        If defined, only those compounds will be returned.

    Returns
    -------
    compounds : immutable dict-like

    Raises
    ------
    FileNotFoundError
        If a logfile is not found.

    Examples
    --------
    >>> import overreact as rx

    >>> compounds = rx.parse_compounds("S: data/ethane/B97-3c/staggered.out")
    >>> compounds
    {'S': {'logfile': 'data/ethane/B97-3c/staggered.out',
           'energy': -209483812.77142256,
           'mult': 1,
           'atomnos': (6, 6, 1, 1, 1, 1, 1, 1),
           ...}}

    >>> compounds = rx.parse_compounds({"S": "data/ethane/B97-3c/staggered.out"})
    >>> compounds
    {'S': {'logfile': 'data/ethane/B97-3c/staggered.out',
           'energy': -209483812.77142256,
           'mult': 1,
           'atomnos': (6, 6, 1, 1, 1, 1, 1, 1),
           ...}}

    >>> compounds = rx.parse_compounds(["S: data/ethane/B97-3c/staggered.out",
    ...                              "E‡: data/ethane/B97-3c/eclipsed.out"])
    >>> compounds
    {'S': {'logfile': 'data/ethane/B97-3c/staggered.out',
           'energy': -209483812.77142256,
           'mult': 1,
           'atomnos': (6, 6, 1, 1, 1, 1, 1, 1),
           ...},
    'E‡': {'logfile': 'data/ethane/B97-3c/eclipsed.out',
           'energy': -209472585.3539883,
           'mult': 1,
           'atomnos': (6, 6, 1, 1, 1, 1, 1, 1),
           ...}}

    >>> compounds = rx.parse_compounds('''S: data/ethane/B97-3c/staggered.out
    ...                                E‡: data/ethane/B97-3c/eclipsed.out''')
    >>> compounds
    {'S': {'logfile': 'data/ethane/B97-3c/staggered.out',
           'energy': -209483812.77142256,
           'mult': 1,
           'atomnos': (6, 6, 1, 1, 1, 1, 1, 1),
           ...},
    'E‡': {'logfile': 'data/ethane/B97-3c/eclipsed.out',
           'energy': -209472585.3539883,
           'mult': 1,
           'atomnos': (6, 6, 1, 1, 1, 1, 1, 1),
           ...}}

    >>> compounds = rx.parse_compounds({"S": "data/ethane/B97-3c/staggered.out",
    ...                              "E‡": "data/ethane/B97-3c/eclipsed.out"})
    >>> compounds
    {'S': {'logfile': 'data/ethane/B97-3c/staggered.out',
           'energy': -209483812.77142256,
           'mult': 1,
           'atomnos': (6, 6, 1, 1, 1, 1, 1, 1),
           ...},
    'E‡': {'logfile': 'data/ethane/B97-3c/eclipsed.out',
           'energy': -209472585.3539883,
           'mult': 1,
           'atomnos': (6, 6, 1, 1, 1, 1, 1, 1),
           ...}}
    """
    if isinstance(text, dict):
        return _check_compounds(text)
    try:
        lines = text.split("\n")
    except AttributeError:
        lines = text
    name = None
    compounds = defaultdict(dict)
    for line in lines:
        if ":" in line:
            name, line = (x.strip() for x in line.split(":", 1))  # noqa: PLW2901
        if not line:
            continue

        if name is not None:
            if "=" in line:
                key, value = (x.strip() for x in line.split("=", 1))
            else:
                key, value = "logfile", line

            if key == "logfile":
                success = False
                value = value.strip('"')
                for p in path:
                    try:
                        # TODO: move on to use pathlib.
                        logger.info(
                            f"trying to read {os.path.join(p, value)}",  # noqa: E501, G004, PTH118
                        )
                        compounds[name].update(
                            read_logfile(os.path.join(p, value)),  # noqa: PTH118
                        )
                    except FileNotFoundError:
                        continue
                    success = True
                    break
                if not success:
                    raise FileNotFoundError(  # noqa: TRY003
                        f"could not find logfile '{value}' in path: {path}",  # noqa: E501, EM102
                    )
            else:
                # one-line JSON-encoded object
                compounds[name][key] = json.loads(value)
    if select is not None:
        # TODO(schneiderfelipe): this workaround still allow unused compounds
        # to be parsed! This should change in the future.
        compounds = {name: compounds[name] for name in select}

    # Apply `extra_energy_term`s
    for name in compounds:
        if "extra_energy_term" in compounds[name]:
            # TODO: this assumes that 1. there's a single `extra_energy_term` and 2. `energy` is present  # noqa: E501
            compounds[name]["energy"] += compounds[name]["extra_energy_term"]

    return dotdict(compounds)


def read_logfile(path):
    """Read a computational chemistry logfile.

    Parameters
    ----------
    path : str
        Path to logfile.

    Returns
    -------
    dict-like

    Raises
    ------
    FileNotFoundError
        If the logfile is not found.

    Examples
    --------
    >>> import overreact as rx

    Some Orca logfiles:

    >>> rx.io.read_logfile("data/symmetries/benzene.out")
    {'logfile': 'data/symmetries/benzene.out',
     'energy': -609176691.0746485,
     'mult': 1,
     'atomnos': (6, 6, 6, 6, 6, 6, 1, 1, 1, 1, 1, 1),
     'atommasses': (12.011, ..., 1.008),
     'atomcoords': ((1.856873, 1.370646, -0.904202),
                    ...,
                    (0.794845, 1.265216, -1.190139)),
     'vibfreqs': (397.68, 397.72, ..., 3099.78, 3109.35),
     'vibdisps': (((-0.030923, -0.073676, 0.142038),
                   ...,
                   (0.400329, 0.039742, 0.107782)))}

    >>> rx.io.read_logfile("data/hickel1992/UM06-2X/6-311++G(d,p)/OH·.out")
    {'logfile': 'data/hickel1992/UM06-2X/6-311++G(d,p)/OH·.out',
     'energy': -198844853.7713648,
     'mult': 2,
     'atomnos': (8, 1),
     'atommasses': (15.999, 1.008),
     'atomcoords': ((-2.436122, 1.35689, 0.0),
                    (-1.463378, 1.35689, 0.0)),
     'vibfreqs': (3775.24,),
     'vibdisps': (((0.062879, 0.0, 0.0),
                  (-0.998021, -0.0, 0.0)),),
     'hessian': ((0.51457715535, ..., -0.00010536738319),
                  ...,
                 (-0.00010536738319, ..., -0.001114845004))}

    A Gaussian logfile as another example:

    >>> rx.io.read_logfile("data/acetate/Gaussian09/wB97XD/6-311++G**/AcO-_0.gau.out")
    {'logfile': 'data/acetate/Gaussian09/wB97XD/6-311++G**/AcO-_0.gau.out',
     'energy': -600283832.3323932,
     'mult': 1,
     'atomnos': (6, 6, 1, 1, 1, 8, 8),
     'atommasses': (12.0, 12.0, 1.007825, 1.007825, 1.007825, 15.9949146, 15.9949146),
     'atomcoords': ((-0.174905, -0.001738, 0.000115),
                    ...,
                    (-0.721316, 1.137017, -4.3e-05)),
     'vibfreqs': (80.2535, 471.106, ..., 3129.0397, 3152.3619),
     'vibdisps': (((0.0, 0.0, 0.02),
                   ...,
                   (0.0, 0.0, 0.0)))}
    """
    if not (parser := ccopen(path)):
        raise FileNotFoundError(  # noqa: TRY003
            f"could not find logfile '{path}'",  # noqa: EM102
        )
    origin = parser.__class__.__name__.lower()
    logger.info(f"reading a {origin} logfile: {path}")  # noqa: G004
    try:
        ccdata = parser.parse()
        data = {
            "logfile": path,
            # This energy may lack dispersion, solvation, correlation, etc.
            "energy": ccdata.scfenergies[-1] * constants.eV * constants.N_A,
            "mult": ccdata.mult,
            "atomnos": rx._misc.totuple(ccdata.atomnos),  # noqa: SLF001
            "atommasses": rx._misc.totuple(ccdata.atommasses),  # noqa: SLF001
            "atomcoords": rx._misc.totuple(ccdata.atomcoords[-1]),  # noqa: SLF001
            "vibfreqs": rx._misc.totuple(ccdata.vibfreqs),  # noqa: SLF001
            "vibdisps": rx._misc.totuple(ccdata.vibdisps),  # noqa: SLF001
        }

        # This solves a current bug in cclib (see
        # https://github.com/cclib/cclib/issues/1080)
        if origin == "gaussian":
            data["atommasses"] = data["atommasses"][: len(data["atomnos"])]

        assert len(data["atomnos"]) == len(data["atommasses"])
        assert len(data["atomnos"]) == len(data["atomcoords"])

        # This properly parses the final single point energy from ORCA files.
        if origin == "orca":
            data.update(_read_orca_logfile(path))
    except AttributeError:
        # Here is the code to parse things on our own. This means we need to
        # parse much more than small pieces of information, as at this point
        # cclib has failed.
        if origin == "orca":
            try:
                # TODO(schneiderfelipe): only run the code below if we know it is
                # an ORCA logfile. The code in this final section should be very
                # specific in supplying *only* things cclib is not (yet) able to
                # parse. Should we add a check for this as well?
                data = _read_orca_logfile(path, minimal=False)
            except FileNotFoundError:
                raise FileNotFoundError(  # noqa: B904, TRY003, TRY200
                    f"could not parse logfile: '{path}'",  # noqa: EM102
                )  # noqa: RUF100
        else:
            raise
    return dotdict(data)


def _read_orca_hess(path):
    """Read an ORCA Hessian file.

    This function is mainly for convenience and probably a temporary reader,
    similarly to `_read_orca_logfile`.

    Parameters
    ----------
    path : str

    Returns
    -------
    array-like

    Examples
    --------
    >>> _read_orca_hess("data/symmetries/water.hess")
    array([[ 0.22070726,  0.14857971, -0.20392672, -0.20214311, -0.13491539,
             0.18797073, -0.0185631 , -0.0136627 ,  0.01595628],
           [ 0.1486261 ,  0.33491105,  0.1281212 , -0.10841476, -0.09216812,
             0.07856856, -0.04021103, -0.24273937, -0.20669001],
           [-0.20390437,  0.12811409,  0.4884253 ,  0.21794551,  0.12323367,
            -0.2279413 , -0.01404124, -0.25135026, -0.26048584],
           [-0.20210785, -0.10844526,  0.21796985,  0.20438786,  0.11486905,
            -0.21455521, -0.00232124, -0.00644584, -0.00337625],
           [-0.13495343, -0.09210889,  0.12327298,  0.11486784,  0.08696293,
            -0.09524441,  0.02006411,  0.00512763, -0.02800616],
           [ 0.18801693,  0.07861423, -0.2279229 , -0.21455684, -0.09524625,
             0.25384054,  0.02657781,  0.01665527, -0.02596059],
           [-0.01865457, -0.04015917, -0.01394534, -0.00232141,  0.02006353,
             0.0265781 ,  0.02097118,  0.0200932 , -0.01263546],
           [-0.01364819, -0.24287842, -0.25144318, -0.00644635,  0.00512802,
             0.01665733,  0.02009284,  0.237703  ,  0.23474131],
           [ 0.0160095 , -0.20678153, -0.26060801, -0.00337537, -0.02800411,
            -0.02595917, -0.01263597,  0.23474251,  0.28651783]])
    """
    with open(path) as file:  # noqa: PTH123
        while file:
            try:
                line = next(file)
            except StopIteration:
                break

            if line[:8] == "$hessian":
                n = int(next(file))
                hessian = np.empty((n, n))

                line = next(file)
                while line:
                    columns = [int(j) for j in line.split()]
                    for i in range(n):
                        entries = next(file).split()[1:]  # first is same as i
                        for j, entry in zip(columns, entries):
                            hessian[i, j] = float(entry)
                    line = next(file).strip()
        return hessian


# heavily inspired by pieces of cclib
def _read_orca_logfile(path, minimal=True):  # noqa: C901, FBT002, PLR0915
    """Read an ORCA logfile.

    This function is a temporary reader, to be used until cclib supports all
    features we need. In particular, this function parses ORCA/xtb logfiles.

    Parameters
    ----------
    path : str
    minimal : bool
        If set, only parse what is not properly done by cclib.

    Returns
    -------
    dict-like

    Examples
    --------
    >>> _read_orca_logfile("data/symmetries/benzene.out")
    {'energy': -609176691.0746485}
    >>> _read_orca_logfile("data/tanaka1996/UMP2/cc-pVTZ/Cl·.out")
    {'energy': ...,
     'hessian': ((...))}
    >>> _read_orca_logfile("data/symmetries/benzene.out", minimal=False)
    {'energy': -609176691.0746485,
     'logfile': 'data/symmetries/benzene.out',
     'mult': 1,
     'atomnos': (6, 6, 6, 6, 6, 6, 1, 1, 1, 1, 1, 1),
     'atomcoords': ((1.856873, 1.370646, -0.904202),
                    ...,
                    (0.794845, 1.265216, -1.190139)),
     'atommasses': (12.011, ..., 1.008),
     'vibfreqs': (397.68, 397.72, ..., 3099.78, 3109.35)}
    >>> _read_orca_logfile("data/tanaka1996/UMP2/cc-pVTZ/Cl·.out",
    ...                    minimal=False)
    {'energy': -1206878421.4741397,
     'hessian': ((2.4367721762e-11, ..., 1.0639954754e-11)),
     'logfile': 'data/tanaka1996/UMP2/cc-pVTZ/Cl·.out',
     'mult': 2,
     'atomnos': (17,),
     'atomcoords': ((0.0, 0.0, 0.0),),
     'atommasses': (35.453,),
     'vibfreqs': ()}
    """
    atomcoords = None
    atommasses = None
    vibfreqs = None
    hessian = None
    with open(path) as file:  # noqa: PTH123
        while file:
            try:
                line = next(file)
            except StopIteration:
                break

            if line[:25] == "FINAL SINGLE POINT ENERGY":
                energy = line.split()[4]
            elif not minimal:
                if line[1:13] == "Multiplicity":
                    mult = line.split()[3]
                elif line[10:14] == "spin":
                    mult = 2 * float(line.split()[2]) + 1
                elif line[10:25] == "number of atoms":
                    natom = int(line.split()[4])
                elif line[0:33] == "CARTESIAN COORDINATES (ANGSTROEM)":
                    next(file)
                    line = next(file)

                    atomnos = []
                    atomcoords = []
                    while len(line) > 1:
                        atom, x, y, z = line.split()
                        if atom[-1] != ">":
                            atomnos.append(rx._misc.atomic_number[atom])  # noqa: SLF001
                            atomcoords.append([float(x), float(y), float(z)])
                        line = next(file)

                    natom = len(atomnos)
                elif line[0:28] == "CARTESIAN COORDINATES (A.U.)":
                    next(file)
                    next(file)
                    line = next(file)

                    atommasses = []
                    while len(line) > 1:
                        if line[:32] == "* core charge reduced due to ECP":
                            break
                        if (
                            line.strip()
                            == "> coreless ECP center with (optional) point charge"
                        ):
                            break
                        _, lb, _, _, mass, x, y, z = line.split()
                        if lb[-1] != ">":
                            atommasses.append(float(mass))
                        line = next(file)
                elif line[:23] == "VIBRATIONAL FREQUENCIES":
                    line = next(file).strip()
                    while (
                        not line
                        or "-----------------------" in line
                        or "Scaling factor for frequencies" in line
                    ):
                        line = next(file).strip()

                    if natom > 1:
                        vibfreqs = []
                        while line:
                            vibfreqs.append(float(line.split()[1]))
                            line = next(file).strip()

                        nonzero = np.nonzero(vibfreqs)[0]
                        first_mode = nonzero[0]
                        vibfreqs = vibfreqs[first_mode:]
                    else:
                        # we have a single atom
                        vibfreqs = np.array([])
    data = {"energy": float(energy) * constants.hartree * constants.N_A}

    if hessian is None:
        try:
            hessian = _read_orca_hess(path.replace(".out", ".hess"))
            data.update({"hessian": rx._misc.totuple(hessian)})  # noqa: SLF001
        except FileNotFoundError:
            pass

    if minimal:
        return data

    # all non-minimal data is given below
    data.update({"logfile": path, "mult": int(mult)})

    if atomcoords is None:
        with open(path.replace(".out", ".xyz")) as file:  # noqa: PTH123
            n = int(next(file))
            next(file)

            atomnos = []
            atomcoords = []
            for _ in range(n):
                line = next(file)
                atom, x, y, z = line.split()
                atomnos.append(rx._misc.atomic_number[atom])  # noqa: SLF001
                atomcoords.append([float(x), float(y), float(z)])
    data.update(
        {
            "atomnos": rx._misc.totuple(atomnos),  # noqa: SLF001
            "atomcoords": rx._misc.totuple(atomcoords),  # noqa: SLF001
        },
    )

    if atommasses is None:
        logger.warning("using atomic masses from periodic table")
        atommasses = []
        for n in atomnos:
            atommasses.append(rx._misc.atomic_mass[n])  # noqa: SLF001
    data.update({"atommasses": rx._misc.totuple(atommasses)})  # noqa: SLF001

    if vibfreqs is not None:
        data.update({"vibfreqs": rx._misc.totuple(vibfreqs)})  # noqa: SLF001

    return data


# https://stackoverflow.com/a/23689767/4039050
class dotdict(dict):  # noqa: N801
    """Access dictionary attributes through dot.notation.

    This object is meant to be immutable, so that it can be hashed.

    Raises
    ------
    NotImplementedError
        If one attempts to change a value.

    Examples
    --------
    >>> mydict = dotdict({
    ...     "val": "it works like a dict",
    ...     "nested": {
    ...         "val": "nested works too"
    ...     },
    ... })
    >>> mydict.val
    'it works like a dict'
    >>> mydict.nested.val
    'nested works too'

    The constructor actually works recursively:

    >>> type(mydict.nested)
    <class 'overreact.io.dotdict'>
    """

    def __init__(self, *args, **kwargs) -> None:  # noqa: ANN101, ANN002, ANN003
        super().__init__(*args, **kwargs)

        for key, val in self.items():
            if isinstance(val, (list, np.ndarray)):
                super().__setitem__(key, rx._misc.totuple(val))  # noqa: SLF001
            elif isinstance(val, dict):
                super().__setitem__(key, dotdict(val))

    __getattr__ = dict.get

    def __setitem__(self, key, value):  # noqa: ANN101, ANN204
        """
        Set an item.

        This is not allowed and will raise an exception.

        Raises
        ------
        NotImplementedError
            If one attempts to change a value.
        """
        raise NotImplementedError(  # noqa: TRY003
            "dotdict objects are immutable",  # noqa: EM101
        )

    # https://stackoverflow.com/a/1151686/4039050
    # https://stackoverflow.com/a/1151705/4039050
    def __hash__(self):  # noqa: ANN101, ANN204, D105
        return hash(self._key())

    # https://stackoverflow.com/a/16162138/4039050
    def _key(self):  # noqa: ANN101
        return (frozenset(self), frozenset(self.items()))

    def __eq__(self, other):  # noqa: ANN101, ANN204, D105
        return self._key() == other._key()  # noqa: SLF001


# https://stackoverflow.com/a/61144084/4039050
class _LazyDict(MutableMapping):
    """Lazily evaluated dictionary."""

    _function = None

    def __init__(self, *args, **kwargs) -> None:  # noqa: ANN101, ANN002, ANN003
        self._dict = dict(*args, **kwargs)

    def __getitem__(self, key):  # noqa: ANN101, ANN204
        """Evaluate value."""
        value = self._dict[key]
        if not isinstance(value, dict):
            data = self._function(value)

            value = data
            self._dict[key] = data
        return value

    def __setitem__(self, key, value):  # noqa: ANN101, ANN204
        """Store value lazily."""
        self._dict[key] = value

    def __delitem__(self, key):  # noqa: ANN101, ANN204
        """Delete value."""
        return self._dict[key]

    def __iter__(self):  # noqa: ANN101, ANN204
        """Iterate over dictionary."""
        return iter(self._dict)

    def __len__(self):  # noqa: ANN101, ANN204
        """Evaluate size of dictionary."""
        return len(self._dict)


class InterfaceFormatter(logging.Formatter):
    """Simple logging interface."""

    def __init__(
        self,  # noqa: ANN101
        fmt=None,
        datefmt=None,
        style="%",
        *args,  # noqa: ANN002
        **kwargs,  # noqa: ANN003
    ) -> None:  # noqa: RUF100
        super().__init__(fmt=fmt, datefmt=datefmt, style=style)
        self.wrapper = textwrap.TextWrapper(*args, **kwargs)
        self.tab = 4 * " "

    def format(self, record):  # noqa: ANN101, A003
        """Format log message."""
        self.wrapper.initial_indent = self.tab
        self.wrapper.subsequent_indent = 2 * self.tab
        if record.module in {"api"}:
            self.wrapper.initial_indent = "\n@ "
            self.wrapper.subsequent_indent = "@   "
        return self.wrapper.fill(super().format(record))
