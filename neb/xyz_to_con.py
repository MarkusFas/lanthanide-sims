#!/usr/bin/env python3
"""Convert one XYZ structure into an EON .con file via readcon."""
import sys

from ase.io import read
import readcon


def main(xyz_path, con_path):
    atoms = read(xyz_path)
    atoms.calc = None  # geometry only, no embedded forces
    readcon.ConFrame.from_ase(atoms).write_con(con_path)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
