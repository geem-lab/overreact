# Contributing

If you want to contribute to the development of **overreact**, you can find the
source code on [GitHub](https://github.com/geem-lab/overreact). The recommended
way of contributing is by forking the repository and pushing your changes to the
forked repository.

> **💡** If you're interested in contributing to open-source projects, make sure
> to read
> [“How to Contribute to Open Source”](https://opensource.guide/how-to-contribute/)
> from [Open Source Guides](https://opensource.guide/) They may even have a
> translation for your native language!

We use [`uv`](https://docs.astral.sh/uv/) to develop the project, so first make sure it's installed.

After cloning your fork, run:

```console
$ git clone --recurse-submodules git@github.com:your-username/overreact.git # your-username is your GitHub username
$ cd overreact
$ uv sync --all-extras
```

> **⚠️** The `--recurse-submodules` flag is also required, not optional: the test
> suite reads sample data (logfiles, models) from the
> [`overreact-data`](https://github.com/geem-lab/overreact-data) submodule
> checked out at `data/`. If you already cloned without it (or the `data/`
> directory is empty), fetch it after the fact with:
>
> ```console
> $ git submodule update --init --recursive
> ```

> **⚠️** The `--all-extras` flag is required, not optional. Without it, `jax`/`jaxlib`
> (the `fast` extra) won't be installed, `overreact` silently falls back to NumPy,
> and several doctests in `overreact/simulate.py` will fail with mismatched output
> (`Array([...], ...)` from JAX vs. plain `array([...])` from NumPy). A plain
> `uv sync` is *not* enough to run the test suite cleanly.

Before submitting any pull requests, make sure all tests pass with:

```console
$ uv run pytest
```

### Git hooks

We use [`pre-commit`](https://pre-commit.com/) to run Ruff (lint and format)
and mypy locally before each commit, so most CI failures are caught before you
even push. It's a dev dependency, so it's already installed after `uv sync
--all-extras` above; you still need to install the hooks themselves once per
clone:

```console
$ uv run pre-commit install
```

From then on, `git commit` runs the checks automatically. To run them on
demand against the whole repository (e.g. before opening a PR):

```console
$ uv run pre-commit run --all-files
```

## Recommended practices

### Reporting issues

The easiest way to report a bug or request a feature is to
[create an issue on GitHub](http://github.com/geem-lab/overreact/issues). The
following greatly enhances our ability to solve the issue you are experiencing:

- Before anything, check if we haven't fixed your issue already in the repository
  by searching for similar issues in the
  [issue tracker](http://github.com/geem-lab/overreact/issues).
- Describe what you were doing when the error occurred, what
  happened, and what you expected to see.
  Also, include the full [traceback](https://realpython.com/python-traceback/) if
  there was an exception.
- Tell us the Python version you're using, as well as the versions of
  overreact (and other packages you might be using with it).
- **Please consider including a
  [minimal reproducible example](https://stackoverflow.com/help/minimal-reproducible-example)
  to help us identify the issue.** This also helps check that the issue is not
  with your own code.

### Asking questions

Use the [discussions](https://github.com/geem-lab/overreact/discussions) for
questions about your own code or on the use of overreact. (Please don't use the
[issue tracker](https://github.com/geem-lab/overreact/issues) for asking
questions, the discussions are a better place to ask questions 😄.)

### Submitting patches

- Include tests if your patch solves a bug, and explain clearly
  under which circumstances the bug happens. Make sure the test fails without
  your patch.
- Use [ruff format](https://docs.astral.sh/ruff/formatter/) to auto-format your code.
- Use [ruff check](https://docs.astral.sh/ruff/linter/) to check for code quality issues.
- Use
  [Numpydoc documentation strings](https://numpydoc.readthedocs.io/en/latest/format.html)
  to document your code.
- Include a string like “fixes #123” in your commit message (where 123 is the
  issue you fixed). See
  [Closing issues using keywords](https://help.github.com/articles/creating-a-pull-request/).
- Bump version according to [semantic versioning](https://semver.org/).

### Releasing

Publishing to PyPI is automatic: `.github/workflows/publish.yml` builds,
tests, and publishes a release the moment a `vX.Y.Z` tag lands on the
repository (using PyPI's
[Trusted Publishing](https://docs.pypi.org/trusted-publishers/), so no PyPI
token is stored as a secret). It also creates/updates the matching GitHub
release with the built sdist and wheel attached. So, to cut a release:

1. Make sure `version` in `pyproject.toml` is bumped to the version you're
   releasing (usually already done per-patch, per the semver bullet above;
   double-check it here). This must match the tag below exactly, or the
   workflow will refuse to publish.
2. On `main`, tag that commit and push the tag: `git tag vX.Y.Z && git push
   origin vX.Y.Z`.
3. Watch the "Publish" workflow run in
   [Actions](https://github.com/geem-lab/overreact/actions/workflows/publish.yml).
   It re-runs the full lint/type/test suite against the tagged commit before
   building and publishing, so a red run means nothing was published.

**One-time setup, before the first release under this workflow** (a PyPI
project owner needs to do this once, on
[pypi.org](https://pypi.org/manage/project/overreact/settings/publishing/)):
add a Trusted Publisher for GitHub with owner `geem-lab`, repository
`overreact`, workflow `publish.yml`, and environment `pypi`. The `pypi`
[environment](https://github.com/geem-lab/overreact/settings/environments)
that step creates can also be configured with required reviewers, if you
ever want a manual approval gate before a publish goes out.
