uuport ase 
from ase.io import read, write
import numpy as np
import sys
import subprocess
from ase.data import atomic_masses, chemical_symbols
from argparse import ArgumentParser

from stoichiometry import solvate_molecule

PACKMOL_HEADER = """\
tolerance {tolerance}
filetype xyz
output {output_file}
seed {seed}
pbc {Lx} {Ly} {Lz}
"""

def write_packmol_input(
    components: dict,
    box_lengths,
    work_dir: Path,
    output_file: str = "solvated.xyz",
    tolerance: float = 2.0,
    seed: int = 12345,
) -> Path:
    """
    components: {name: {"filepath": ..., "count": ...}, ...} -- any number
    of entries, any mix of solvents/ions. Writes one structure block per
    entry, all packed into the same box.
    """
    Lx, Ly, Lz = box_lengths

    lines = [PACKMOL_HEADER.format(
        tolerance=tolerance, output_file=output_file, seed=seed, Lx=Lx, Ly=Ly, Lz=Lz
    )]
    for name, spec in components.items():
        count = int(spec["count"])
        if count <= 0:
            continue  # skip zero-count components rather than writing an empty block
        lines.append(STRUCTURE_BLOCK.format(
            filepath=spec["filepath"], count=count, Lx=Lx, Ly=Ly, Lz=Lz
        ))

    inp_path = work_dir / "packmol.inp"
    inp_path.write_text("".join(lines))
    return inp_path


def run_packmol(components: dict, box_lengths, work_dir: str = ".") -> dict:
    """Job: write packmol input from `components`, run packmol, return the packed structure."""
    # resolve to an absolute path: jobflow may run each job in its own
    # auto-created folder, so a relative work_dir here would not point to
    # the same place once a downstream job (e.g. run_lammps) reads it back
    work_dir = Path(work_dir).resolve()
    work_dir.mkdir(parents=True, exist_ok=True)
    output_file = "solvated.xyz"

    inp_path = write_packmol_input(
        components, box_lengths, work_dir, output_file=output_file
    )

    result = subprocess.run(
        ["packmol"], stdin=inp_path.open(), cwd=work_dir,
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"packmol failed:\n{result.stdout}\n{result.stderr}")

    return {
        "packed_xyz": str(work_dir / output_file),
        "box_lengths": tuple(box_lengths),
        "components": components,
    }



def generate_packmol_input(box_lengths, num_water, mol_file="molecule.xyz", water_file="water.xyz", output_file="output.xyz", packmol_file="packmol.inp"):
    """
    Generates a Packmol input file to solvate a molecule with water.

    Parameters:
    box_size (float): Length of the cubic box in Angstroms.
    num_water (int): Number of water molecules to insert.
    mol_file (str): XYZ file of the small molecule.
    water_file (str): XYZ file of a single water molecule.
    output_file (str): Name of the Packmol input file.
    """

    packmol_template = f"""tolerance 2.0
filetype xyz
output {output_file}
pbc {box_lengths[0]} {box_lengths[1]} {box_lengths[2]}
structure {mol_file}
  number 1
  fixed {box_lengths[0]/2} {box_lengths[1]/2} {box_lengths[2]/2} 0. 0. 0. 
  center
end structure

structure {water_file}
  number {num_water}
  inside box 0. 0. 0. {box_lengths[0]} {box_lengths[1]} {box_lengths[2]}
end structure
"""

    with open(packmol_file, "w") as f:
        f.write(packmol_template)

    print(f"Packmol input file '{packmol_file}' generated successfully.")



def center_molecule(mol_file, box_lengths):
    mol = read(mol_file)
    # Center the molecule in the middle of the box
    mol.center(vacuum=0.0)
    # Shift molecule to the exact center of the box
    print(box_lengths / 2)
    print(mol.get_center_of_mass())
    print(box_lengths / 2 - mol.get_center_of_mass())
    mol.translate(box_lengths / 2 - mol.get_center_of_mass())
    write(mol_file, mol, format='xyz')

def get_num(vol):
    mass = rho*vol
    num = (mass/molmass) * 6.022E23
    num -= 1 # correcting for the volume taken up by the dota molecules
    return round(num)

if __name__ == "__main__":

    parser = ArgumentParser(description="Create a water box around a molecule using Packmol")
    parser.add_argument("solute_file", help="XYZ file of the small molecule or atom")
    parser.add_argument("water_file", help="XYZ file of a single water molecule")
    parser.add_argument("output_file", help="Output file for the solvated system")
    parser.add_argument("--box_length", type=float, default=20.0, help="Length of the cubic box in Angstroms (default: 30.0)")
    parser.add_argument("--cation", default="Na", help="type of cation to dissolve")
    parser.add_argument("--anion", default="Cl", help="type of anion to dissolve")
    parser.add_argument("--cation-strength", type=float, default=0.0, help="concentration of cation to dissolve")
    parser.add_argument("--anion-strength", type=float, default=0.0,
                        help="concentration of anion to dissolve")
    args = parser.parse_args()

    components = solvate_molecule(
            )

    run_packmol()

