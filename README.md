##

<div align="center">

[![PyPI](https://img.shields.io/pypi/v/overreact)](https://pypi.org/project/overreact/)
[![Python Versions](https://img.shields.io/pypi/pyversions/overreact)](https://pypi.org/project/overreact/)
[![CI](https://github.com/geem-lab/overreact/actions/workflows/python-package.yml/badge.svg)](https://github.com/geem-lab/overreact/actions/workflows/python-package.yml)
[![Coverage](https://codecov.io/gh/geem-lab/overreact/branch/main/graph/badge.svg?token=4WAVXCRXY8)](https://codecov.io/gh/geem-lab/overreact)
[![License](https://img.shields.io/github/license/geem-lab/overreact)](https://github.com/geem-lab/overreact/blob/main/LICENSE)

[![User guide](https://img.shields.io/badge/user%20guide-available-blue)](https://geem-lab.github.io/overreact-guide/)
[![GitHub Discussions](https://img.shields.io/github/discussions/geem-lab/overreact)](https://github.com/geem-lab/overreact/discussions)
[![GitHub issues](https://img.shields.io/github/issues-raw/geem-lab/overreact)](https://github.com/geem-lab/overreact/issues)

[![downloads/month](https://pepy.tech/badge/overreact/month)](https://pepy.tech/project/overreact)
[![total downloads](https://pepy.tech/badge/overreact)](https://pepy.tech/project/overreact)

[![DOI](https://zenodo.org/badge/214332027.svg)](https://zenodo.org/badge/latestdoi/214332027)

[![Made in Brazil 🇧🇷](https://img.shields.io/badge/made%20in-Brazil-009c3b)](https://github.com/geem-lab/overreact#funding)

</div>

<div align="center">
    <img alt="overreact" src="https://raw.githubusercontent.com/geem-lab/overreact-guide/master/logo.png" />
</div>

##

**overreact** is a **library** and a **command-line tool** for building and
analyzing homogeneous **microkinetic models** from **first-principles
calculations**:

```python
In [1]: from overreact import api

In [2]: api.get_k("S -> E‡ -> S",
   ...:           {"S": "data/ethane/B97-3c/staggered.out",
   ...:            "E‡": "data/ethane/B97-3c/eclipsed.out"})
Out[2]: array([8.16880917e+10])
```

<details>
    <summary style="cursor: pointer;">
        🤔 What is <strong>microkinetic modeling</strong>?
    </summary>
    <p>
        <strong>Microkinetic modeling</strong> is a technique used to predict the outcome
        of complex chemical reactions.
        It can be used
        to investigate the catalytic transformations
        of molecules.
        <strong>overreact</strong> makes it easy to create
        and analyze microkinetic models built
        from computational chemistry data.
    </p>
</details>

<br/>

<details>
    <summary style="cursor: pointer;">
        🧐 What do you mean by <strong>first-principles calculations</strong>?
    </summary>
    <p>
        We use the term <strong>first-principles calculations</strong> to refer to
        calculations performed using quantum chemical modern methods such as
        <a href="https://en.wikipedia.org/wiki/Hartree%E2%80%93Fock_method">Wavefunction</a>
        and
        <a href="https://en.wikipedia.org/wiki/Density_functional_theory">Density Functional</a>
        theories.
        For instance, the three-line example code above calculates the rate of methyl rotation in ethane (at
        <a href="https://doi.org/10.1063/1.5012601">B97-3c</a>).
        (Rather surprisingly, the error found is less than 2%
        <a href="http://dx.doi.org/10.1126/science.1132178">when compared to available experimental results</a>.)
    </p>
</details>

<br/>

**overreact** uses **precise thermochemical partition funtions**, **tunneling
corrections** and data is **parsed directly** from computational chemistry
output files thanks to [`cclib`](https://cclib.github.io/) (see the
[list of its supported programs](https://cclib.github.io/#summary)).

## Installation

**overreact** is a Python package, so you can easily install it with
[`pip`](https://pypi.org/project/pip/):

```console
$ pip install "overreact[cli,fast]"
```

See the
[installation guide](https://geem-lab.github.io/overreact-guide/install.html)
for more details.

> **🚀** **Where to go from here?** Take a look at the
> [short introduction](https://geem-lab.github.io/overreact-guide/tutorial.html).
> Or see
> [below](https://geem-lab.github.io/overreact-guide/intro.html#where-to-go-next)
> for more guidance.

## Citing **overreact**

If you use **overreact** in your research, please cite:

> F. S. S. Schneider and G. F. Caramori. _**geem-lab/overreact**: a tool for
> creating and analyzing microkinetic models built from computational chemistry
> data, v1.0.2_. **2021**.
> [DOI:10.5281/zenodo.5730603](https://zenodo.org/badge/latestdoi/214332027).
> Freely available at: <<https://github.com/geem-lab/overreact>>.

Here's the reference in [BibTeX](http://www.bibtex.org/) format:

```bibtex
@misc{overreact2021,
  howpublished = {\url{https://github.com/geem-lab/overreact}}
  year = {2021},
  author = {Schneider, F. S. S. and Caramori, G. F.},
  title = {
    \textbf{geem-lab/overreact}: a tool for creating and analyzing
    microkinetic models built from computational chemistry data, v1.0.2
  },
  doi = {10.5281/zenodo.5730603},
  url = {https://zenodo.org/record/5730603},
  publisher = {Zenodo},
  copyright = {Open Access}
}
```

> **✏️** A paper describing **overreact** is currently being prepared. When it
> is published, the above BibTeX entry will be updated.

## License

**overreact** is open-source, released under the permissive **MIT license**. See
[the LICENSE agreement](https://github.com/geem-lab/overreact/blob/main/LICENSE).

## Funding

This project was developed at the [GEEM lab](https://geem-ufsc.org/)
([Federal University of Santa Catarina](https://en.ufsc.br/), Brazil), and was
partially funded by the
[Brazilian National Council for Scientific and Technological Development (CNPq)](https://cnpq.br/),
grant number 140485/2017-1.
