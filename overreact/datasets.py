#!/usr/bin/env python3

"""Small toy datasets for tests and benchmark."""

import os

import overreact as rx

data_path = os.path.normpath(os.path.join(os.path.dirname(__file__), "../data/"))


logfiles = {}
for name in os.listdir(data_path):
    walk_dir = os.path.join(data_path, name)
    if os.path.isdir(walk_dir):
        logfiles[name] = rx.io._LazyDict()
        logfiles[name]._function = rx.io.read_logfile
        for root, _, files in os.walk(walk_dir):
            for filename in files:
                if filename.endswith(".out"):
                    logfiles[name][
                        f"{filename[:-4]}@{os.path.relpath(root, walk_dir)}".replace(
                            "@.", ""
                        )
                    ] = os.path.join(root, filename)


if __name__ == "__main__":
    for name in logfiles:
        for compound in logfiles[name]:
            print(name, compound, logfiles[name][compound].logfile)
