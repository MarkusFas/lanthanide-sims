#!/usr/bin/env python3
"""
Minimal reader for a generic LAMMPS log file: pulls Temp and Volume out of
every thermo table it finds (any block whose header line starts with
"Step" and includes "Temp" and "Volume" columns), prints their mean and
variance, and plots both series.

Multiple thermo blocks (e.g. separate equilibration/production runs in one
log) are concatenated automatically, as long as they share the same column
layout as the first block found; blocks with a different layout are
skipped rather than mixed in.

Usage
-----
    python plot_log_fluctuations.py [log.lammps]

(defaults to "log.lammps" in the current directory if no path is given)
"""
import sys

import numpy as np
import matplotlib.pyplot as plt


def read_thermo(path):
    """Parse every Step/.../Temp/.../Volume/... thermo table in a LAMMPS log."""
    columns = None
    active = False
    rows = []
    with open(path) as fh:
        for line in fh:
            tokens = line.split()
            if not tokens:
                active = False
                continue
            if tokens[0] == "Step" and "Temp" in tokens and "Volume" in tokens:
                if columns is None:
                    columns = tokens          # lock in the layout from the first matching block
                active = (tokens == columns)  # only read blocks with that exact layout
                continue
            if not active:
                continue
            if len(tokens) != len(columns):
                active = False
                continue
            try:
                rows.append([float(t) for t in tokens])
            except ValueError:
                active = False  # non-numeric line ("Loop time of ...", etc.) -> end of block

    if not rows:
        raise RuntimeError(f"No Step/Temp/Volume thermo table found in {path}")
    return columns, np.array(rows)


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "log.lammps"
    columns, data = read_thermo(path)
    temp = data[:, columns.index("Temp")]
    vol = data[:, columns.index("Volume")]

    print(f"Read {len(temp)} thermo rows from {path}")
    print(f"Temp:   mean = {temp.mean():.6g}   variance = {temp.var():.6g}   std = {temp.std():.6g}")
    print(f"Volume: mean = {vol.mean():.6g}   variance = {vol.var():.6g}   std = {vol.std():.6g}")

    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(7, 6))

    ax1.plot(temp, lw=0.8)
    ax1.axhline(temp.mean(), color="k", ls="--", lw=1)
    ax1.set_ylabel("Temp")

    ax2.plot(vol, lw=0.8, color="tab:orange")
    ax2.axhline(vol.mean(), color="k", ls="--", lw=1)
    ax2.set_ylabel("Volume")
    ax2.set_xlabel("thermo row")

    fig.tight_layout()
    fig.savefig("log_fluctuations.png", dpi=150)
    print("Wrote log_fluctuations.png")
    plt.show()


if __name__ == "__main__":
    main()
