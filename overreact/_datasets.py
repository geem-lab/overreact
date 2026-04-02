"""Small toy datasets for tests and benchmark."""

from __future__ import annotations

from pathlib import Path

import overreact as rx

data_path = Path(__file__).parent.parent / "data"

logfiles = {}
for walk_dir in data_path.iterdir():
    if walk_dir.is_dir():
        name = walk_dir.name
        logfiles[name] = rx.io._LazyDict()
        logfiles[name]._function = rx.io.read_logfile
        for filepath in walk_dir.rglob("*.out"):
            rel = filepath.parent.relative_to(walk_dir)
            key = f"{filepath.stem}@{rel}".replace("@.", "")
            logfiles[name][key] = str(filepath)


if __name__ == "__main__":
    for name in logfiles:
        for compound in logfiles[name]:
            print(name, compound, logfiles[name][compound].logfile)
