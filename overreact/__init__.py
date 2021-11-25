#!/usr/bin/env python3

"""
.. include:: ../README.md
"""
__docformat__ = "restructuredtext"

import pkg_resources as _pkg_resources

from overreact.api import get_enthalpies  # noqa: F401
from overreact.api import get_entropies  # noqa: F401
from overreact.api import get_freeenergies  # noqa: F401
from overreact.api import get_internal_energies  # noqa: F401
from overreact.api import get_k  # noqa: F401
from overreact.api import get_kappa  # noqa: F401
from overreact.core import Scheme  # noqa: F401
from overreact.core import get_transition_states  # noqa: F401
from overreact.core import is_transition_state  # noqa: F401
from overreact.core import parse_reactions  # noqa: F401
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
__license__ = "MIT"  # I'm too lazy to get it from setup.py...
__doi__ = "10.5281/ZENODO.5643960"
__citation__ = r"""
@misc{overreact2021,
  howpublished = {\url{https://github.com/geem-lab/overreact}}
  year = {2021},
  author = {Schneider, F. S. S. and Caramori, G. F.},
  title = {
    \textbf{geem-lab/overreact}: a tool for creating and analyzing
    microkinetic models built from computational chemistry data, v1.0.1
  },
  doi = {10.5281/ZENODO.5643960},
  url = {https://zenodo.org/record/5643960},
  publisher = {Zenodo},
  copyright = {Open Access}
}
"""
__url_pypi__ = "https://pypi.org/project/overreact/"
__url_guide__ = "https://geem-lab.github.io/overreact-guide/"
__url_discussions__ = "https://github.com/geem-lab/overreact/discussions"
__url_issues__ = "https://github.com/geem-lab/overreact/issues"
__repo__ = "https://github.com/geem-lab/overreact"
__headline__ = "📈 Create and analyze chemical microkinetic models built from computational chemistry data."
