[![DOI](https://zenodo.org/badge/214332027.svg)](https://zenodo.org/badge/latestdoi/214332027)
[![PyPI](https://img.shields.io/pypi/v/overreact)](https://pypi.org/project/overreact/)
[![build](https://github.com/geem-lab/overreact/actions/workflows/python-package.yml/badge.svg)](https://github.com/geem-lab/overreact/actions/workflows/python-package.yml)
[![codecov](https://codecov.io/gh/geem-lab/overreact/branch/main/graph/badge.svg?token=4WAVXCRXY8)](https://codecov.io/gh/geem-lab/overreact)
[![GitHub license](https://img.shields.io/github/license/geem-lab/overreact)](https://github.com/geem-lab/overreact/blob/main/LICENSE)
![Made in Brazil 🇧🇷](https://img.shields.io/badge/made%20in-Brazil-009c3b)

<div align="center">
    <img alt="overreact" src="https://raw.githubusercontent.com/geem-lab/overreact-guide/master/logo.png" />
</div>

**overreact** is a **library** and a **command-line tool** for building and
analyzing
[microkinetic models](https://geem-lab.github.io/overreact-guide/#microkinetic).
Data is parsed directly from computational chemistry output files thanks to
[`cclib`](https://cclib.github.io/) (see the
[list of supported programs](https://cclib.github.io/#summary)).

## Installation

**overreact** is a Python package, so you can easily install it with
[`pip`](https://pypi.org/project/pip/):

```bash
$ pip install "overreact[cli,fast]"
```

See the
[installation instructions](https://geem-lab.github.io/overreact-guide/install.html)
in our guide for more details.

## Citing **overreact**

If you use **overreact** in your research, please cite:

> F. S. S. Schneider and G. F. Caramori. **overreact**: a tool for creating and
> analyzing microkinetic models built from computational chemistry data.
> **2021**. Available at: <https://github.com/geem-lab/overreact>.

Here's the reference in [BibTeX](http://www.bibtex.org/) format:

<!-- @article{overreact,
  title = \textbf{overreact}: a tool for creating and analyzing microkinetic models built from computational chemistry data},
  author = {Schneider, F. S. S. and Caramori, G. F.},
  journal={J. Chem. Phys.},
  volume={155},
  number={1},
  pages={0},
  year = {2021},
  publisher={American Chemical Society (ACS)},
  doi={10.1063/1.5058983},
  url={https://doi.org/10.1063/1.5058983}
} -->

```bibtex
@misc{overreact2021,
  title        = {
    \textbf{overreact}: a tool for creating and analyzing microkinetic models
    built from computational chemistry data, ver. 1.0
  },
  author       = {Schneider, F. S. S. and Caramori, G. F.},
  year         = 2021,
  howpublished = {\url{https://github.com/geem-lab/overreact}}
}
```

A paper describing **overreact** is currently being prepared. When it is
published, the above BibTeX entry will be updated.

## License

**overreact** is open-source, released under the permissive **MIT license**. See
[our LICENSE file](https://github.com/geem-lab/overreact/blob/main/LICENSE).

## Funding

This project was developed at the [GEEM lab](https://geem-ufsc.org/)
([Federal University of Santa Catarina](https://en.ufsc.br/), Brazil), and was
partially funded by the
[Brazilian National Council for Scientific and Technological Development (CNPq)](https://cnpq.br/),
grant number 140485/2017-1.
