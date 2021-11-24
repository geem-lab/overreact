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
[installation guide](https://geem-lab.github.io/overreact-guide/install.html)
for more details.

## Citing **overreact**

If you use **overreact** in your research, please cite:

> F. S. S. Schneider and G. F. Caramori. _**geem-lab/overreact**: a tool for
> creating and analyzing microkinetic models built from computational chemistry
> data, v1.0.1_. **2021**.
> [DOI:10.5281/ZENODO.5643960](https://doi.org/10.5281/ZENODO.5643960). Freely
> available at: <<https://github.com/geem-lab/overreact>>.

Here's the reference in [BibTeX](http://www.bibtex.org/) format:

```bibtex
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
```

A paper describing **overreact** is currently being prepared. When it is
published, the above BibTeX entry will be updated.

## License

**overreact** is open-source, released under the permissive **MIT license**. See
[the LICENSE agreement](https://github.com/geem-lab/overreact/blob/main/LICENSE).

## Funding

This project was developed at the [GEEM lab](https://geem-ufsc.org/)
([Federal University of Santa Catarina](https://en.ufsc.br/), Brazil), and was
partially funded by the
[Brazilian National Council for Scientific and Technological Development (CNPq)](https://cnpq.br/),
grant number 140485/2017-1.
