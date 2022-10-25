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

After cloning your fork, we recommend using [Poetry](https://python-poetry.org/)
for managing your contributions:

```console
$ git clone git@github.com:your-username/overreact.git  # your-username is your GitHub username
$ cd overreact
$ poetry install -E cli -E fast -E solvents  # all optional features
```

## Recommended practices

### Reporting issues

The easiest way to report a bug or request a feature is to
[create an issue on GitHub](http://github.com/geem-lab/overreact/issues). The
following greatly enhances our ability to solve the issue you are experiencing:

-   Before anything, check if we haven't fixed your issue already in the repository
    by searching for similar issues in the
    [issue tracker](http://github.com/geem-lab/overreact/issues).
-   Describe what you were doing when the error occurred, what
    happened, and what you expected to see.
    Also, include the full [traceback](https://realpython.com/python-traceback/) if
    there was an exception.
-   Tell us the Python version you're using, as well as the versions of
    overreact (and other packages you might be using with it).
-   **Please consider including a
    [minimal reproducible example](https://stackoverflow.com/help/minimal-reproducible-example)
    to help us identify the issue.** This also helps check that the issue is not
    with your own code.

### Asking questions

Use the [discussions](https://github.com/geem-lab/overreact/discussions) for
questions about your own code or on the use of overreact. (Please don't use the
[issue tracker](https://github.com/geem-lab/overreact/issues) for asking
questions, the discussions are a better place to ask questions 😄.)

### Submitting patches

-   Include tests if your patch solves a bug, and explain clearly
    under which circumstances the bug happens. Make sure the test fails without
    your patch.
-   Use [Black](https://black.readthedocs.io/) to auto-format your code.
-   Use
    [Numpydoc documentation strings](https://numpydoc.readthedocs.io/en/latest/format.html)
    to document your code.
-   Include a string like “fixes #123” in your commit message (where 123 is the
    issue you fixed). See
    [Closing issues using keywords](https://help.github.com/articles/creating-a-pull-request/).
-   Bump version according to [semantic versioning](https://semver.org/).
