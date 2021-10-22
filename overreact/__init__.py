#!/usr/bin/env python3

"""
Main module of overreact.

Except when otherwise indicated, all units are in the international system.
Particularly, energy is in joules per mole.
"""

import pkg_resources as _pkg_resources

from overreact.api import get_enthalpies  # noqa: F401
from overreact.api import get_entropies  # noqa: F401
from overreact.api import get_freeenergies  # noqa: F401
from overreact.api import get_internal_energies  # noqa: F401
from overreact.api import get_k  # noqa: F401
from overreact.api import get_kappa  # noqa: F401
from overreact.core import get_transition_states  # noqa: F401
from overreact.core import is_transition_state  # noqa: F401
from overreact.core import parse_reactions  # noqa: F401
from overreact.core import Scheme  # noqa: F401
from overreact.core import unparse_reactions  # noqa: F401
from overreact.io import parse_compounds  # noqa: F401
from overreact.io import parse_model  # noqa: F401
from overreact.simulate import get_bias  # noqa: F401
from overreact.simulate import get_dydt  # noqa: F401
from overreact.simulate import get_fixed_scheme  # noqa: F401
from overreact.simulate import get_y  # noqa: F401
from overreact.thermo import change_reference_state  # noqa: F401
from overreact.thermo import get_delta  # noqa: F401
from overreact.thermo import get_reaction_entropies  # noqa: F401

__version__ = _pkg_resources.get_distribution(__name__).version
__license__ = "MIT"  # too hard to get from setup.py
__doi__ = "10.5281/zenodo.5154060"
__citation__ = r"""
@misc{overreact2021,
  title        = {
    \textbf{overreact}: a tool for creating and analyzing microkinetic models
    built from computational chemistry data, v1.0alpha
  },
  author       = {Schneider, Felipe S. S. and Caramori, Giovanni F.},
  year         = 2021,
  publisher    = {Zenodo},
  doi          = {10.5281/zenodo.5154060},
  url          = {https://zenodo.org/record/5154060},
  copyright    = {Open Access},
  howpublished = {\url{https://github.com/geem-lab/overreact}}
}
"""
__url__ = "https://geem-lab.github.io/overreact-guide"
__repo__ = "https://github.com/geem-lab/overreact"
__headline__ = "Construct precise chemical microkinetic models from first principles."
