#!/usr/bin/env python3

"""Small toy datasets for tests and benchmark."""

from collections.abc import MutableMapping
import os
import warnings

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    from cclib import ccopen
    from cclib.parser import ccData

_datapath = os.path.normpath(os.path.join(os.path.dirname(__file__), "../data/"))


# https://stackoverflow.com/a/61144084/4039050
class LazyDict(MutableMapping):
    """Lazily evaluated dictionary."""

    function = None

    def __init__(self, *args, **kargs):
        self._dict = dict(*args, **kargs)

    def __getitem__(self, key):
        """Evaluate value."""
        value = self._dict[key]
        if not isinstance(value, ccData):
            ccdata = self.function(value)
            ccdata.jobfilename = value

            value = ccdata
            self._dict[key] = ccdata
        return value

    def __setitem__(self, key, value):
        """Store value lazily."""
        self._dict[key] = value

    def __delitem__(self, key):
        """Delete value."""
        return self._dict[key]

    def __iter__(self):
        """Iterate over dictionary."""
        return iter(self._dict)

    def __len__(self):
        """Evaluate size of dictionary."""
        return len(self._dict)


def _load_logfile(path):
    try:
        return ccopen(path).parse()
    except AttributeError:
        raise ValueError(f"could not parse logfile: '{path}'")


logfiles = LazyDict()
logfiles.function = _load_logfile
for filename in os.listdir(_datapath):
    if filename.endswith(".out"):
        logfiles[filename[:-4]] = os.path.join(_datapath, filename)
