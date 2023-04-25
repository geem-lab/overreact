--- <!-- prettier-ignore -->

<div align="center">
<p>
<a href="https://pypi.org/project/overreact/" >
<img src="https://img.shields.io/pypi/v/overreact" alt="PyPI" />
</a>
<a href="https://pypi.org/project/overreact/" >
<img src="https://img.shields.io/pypi/pyversions/overreact" alt="Python Versions" />
</a>
<a href="https://github.com/geem-lab/overreact/actions/workflows/python-package.yml" />
<img src="https://github.com/geem-lab/overreact/actions/workflows/python-package.yml/badge.svg" alt="CI" />
</a>
<a href="https://codecov.io/gh/geem-lab/overreact" >
<img src="https://codecov.io/gh/geem-lab/overreact/branch/main/graph/badge.svg?token=4WAVXCRXY8" alt="Coverage" />
</a>
<a href="https://github.com/geem-lab/overreact/blob/main/LICENSE">
<img src="https://img.shields.io/github/license/geem-lab/overreact" alt="License" />
</a>
</p>
<p>
<a href="https://geem-lab.github.io/overreact-guide/">
<img src="https://img.shields.io/badge/user%20guide-available-blue" alt="User guide" />
</a>
<a href="https://github.com/geem-lab/overreact/discussions">
<img src="https://img.shields.io/github/discussions/geem-lab/overreact" alt="GitHub Discussions" />
</a>
<a href="https://github.com/geem-lab/overreact/issues">
<img src="https://img.shields.io/github/issues-raw/geem-lab/overreact" alt="GitHub issues" />
</a>
</p>
<p>
<a href="https://pepy.tech/project/overreact" >
<img src="https://pepy.tech/badge/overreact/month" alt="downloads/month" />
</a>
<a href="https://pepy.tech/project/overreact" >
<img src="https://pepy.tech/badge/overreact" alt="total downloads" />
</a>
</p>
<p>
<a href="https://doi.org/10.1002/jcc.26861" >
<img src="https://img.shields.io/badge/DOI-10.1002%2Fjcc.26861-blue" alt="DOI" />
</a>
<a href="https://doi.org/10.5281/zenodo.7504800">
<img src="https://zenodo.org/badge/DOI/10.5281/zenodo.7504800.svg" alt="DOI" />
</a>
</p>
<p>
<a href="https://github.com/geem-lab/overreact#funding" >
<img src="https://img.shields.io/badge/made%20in-Brazil-009c3b" alt="Made in Brazil 🇧🇷" />
</a>
</p>
</div>

<div align="center">
<img alt="overreact" src="https://raw.githubusercontent.com/geem-lab/overreact-guide/master/logo.png" />
</div>

--- <!-- prettier-ignore -->

**overreact** is a **library** and a **command-line tool** for building and
analyzing homogeneous **microkinetic models** from **first-principles
calculations**:

```python
In [1]: from overreact import api  # the api

In [2]: api.get_k("S -> E‡ -> S",  # your model
   ...:           {"S": "data/ethane/B97-3c/staggered.out",  # your data
   ...:            "E‡": "data/ethane/B97-3c/eclipsed.out"})
Out[2]: array([8.16880917e+10])  # your results
```

The user specifies a set of
elementary reactions that are believed to be relevant for the overall chemical
phenomena. **overreact** offers a hopefully complete but simple environment for
hypothesis testing in first-principles chemical kinetics.

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

> Schneider, F. S. S.; Caramori, G. F.
> _**Overreact**, an in Silico Lab: Automative Quantum Chemical Microkinetic Simulations for Complex Chemical Reactions_.
> Journal of Computational Chemistry **2022**, 44 (3), 209–217.
> [doi:10.1002/jcc.26861](https://doi.org/10.1002/jcc.26861).

Here's the reference in [BibTeX](http://www.bibtex.org/) format:

```bibtex
@article{overreact_paper2022,
  title         = {Overreact, an in silico lab: Automative quantum chemical microkinetic simulations for complex chemical reactions},
  author        = {Schneider, Felipe S. S. and Caramori, Giovanni F.},
  year          = {2022},
  month         = {Apr},
  journal       = {Journal of Computational Chemistry},
  publisher     = {Wiley},
  volume        = {44},
  number        = {3},
  pages         = {209–217},
  doi           = {10.1002/jcc.26861},
  issn          = {1096-987x},
  url           = {http://dx.doi.org/10.1002/jcc.26861},
}
@software{overreact_software2021,
  title         = {geem-lab/overreact: v1.2.0 \vert{} Zenodo},
  author        = {Felipe S. S. Schneider and Let\'{\i}cia M. P. Madureira and  Giovanni F. Caramori},
  year          = {2023},
  month         = {Jan},
  publisher     = {Zenodo},
  doi           = {10.5281/zenodo.7504800},
  url           = {https://doi.org/10.5281/zenodo.7504800},
  version       = {v1.2.0},
  howpublished  = {\url{https://github.com/geem-lab/overreact}},
}
```

## License

**overreact** is open-source, released under the permissive **MIT license**. See
[the LICENSE agreement](https://github.com/geem-lab/overreact/blob/main/LICENSE).

## Funding

This project was developed at the [GEEM lab](https://geem-ufsc.org/)
([Federal University of Santa Catarina](https://en.ufsc.br/), Brazil), and was
partially funded by the
[Brazilian National Council for Scientific and Technological Development (CNPq)](https://cnpq.br/),
grant number 140485/2017-1.
