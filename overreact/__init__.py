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

__headline__ = "📈 Create and analyze chemical microkinetic models built from computational chemistry data."

__url_repo__ = "https://github.com/geem-lab/overreact"
__url_issues__ = f"{__url_repo__}/issues"
__url_discussions__ = f"{__url_repo__}/discussions"
__url_pypi__ = "https://pypi.org/project/overreact/"
__url_guide__ = "https://geem-lab.github.io/overreact-guide/"

__doi__ = "10.1002/jcc.26861"
__zenodo_doi__ = "10.5281/zenodo.5730603"
__citations__ = (
    r"""
@article{overreact_paper2022,
  title        = {Overreact, an in silico lab: Automative quantum chemical microkinetic simulations for complex chemical reactions},
  url          = {http://dx.doi.org/DOI_PLACEHOLDER},
  author       = {Schneider, Felipe S. S. and Caramori, Giovanni F.},
  DOI          = {DOI_PLACEHOLDER},
  ISSN         = {1096-987X},
  journal      = {Journal of Computational Chemistry},
  publisher    = {Wiley},
  year         = {2022},
  month        = {Apr}
}
@software{overreact_software2021,
  title        = {geem-lab/overreact: vVERSION_PLACEHOLDER | Zenodo},
  version      = {vVERSION_PLACEHOLDER},
  howpublished = {\url{URL_REPO_PLACEHOLDER}},
  url          = {https://doi.org/ZENODO_DOI_PLACEHOLDER},
  author       = {Schneider, Felipe S. S. and Caramori, Giovanni F.},
  DOI          = {ZENODO_DOI_PLACEHOLDER},
  publisher    = {Zenodo},
  year         = {2021},
  month        = {Nov}
}
""".replace(
        "DOI_PLACEHOLDER", __doi__
    )
    .replace("ZENODO_DOI_PLACEHOLDER", __zenodo_doi__)
    .replace("URL_REPO_PLACEHOLDER", __url_repo__)
    .replace("VERSION_PLACEHOLDER", __version__)
)
