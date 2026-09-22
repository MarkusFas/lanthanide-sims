from ase.io import read, write
import numpy as np
import shutil
import subprocess
from argparse import ArgumentParser
from pathlib import Path
from stoichiometry import ion_counts_from_molarity, ion_xyz_file, N_A
from ase.data import chemical_symbols
PACKMOL_HEADER = """\
tolerance {tolerance}
filetype xyz
output {output_file}
seed {seed}
pbc {Lx} {Ly} {Lz}
"""

# Was referenced in write_packmol_input but never defined -- this was the
# main reason the script couldn't run at all.
STRUCTURE_BLOCK = """\
structure {filepath}
  number {count}
  inside box 0. 0. 0. {Lx} {Ly} {Lz}
end structure
"""

# Defaults for get_num(); previously these were bare globals that were
# never assigned anywhere (NameError as soon as get_num() was called).
WATER_DENSITY_G_PER_CM3 = 0.997
WATER_MOLAR_MASS_G_PER_MOL = 18.015

# Ion structure files (Na.xyz, Cl.xyz, ...) live here by convention -- a
# sibling of this script's own directory, per stoichiometry.py's layout.
DEFAULT_STRUCTURES_DIR = Path(__file__).resolve().parent.parent / "i-solvate" / "initial_structures"


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


def run_packmol(components: dict, box_lengths, work_dir: str = ".", output_file: str = "solvated.xyz") -> dict:
    """Job: write packmol input from `components`, run packmol, return the packed structure."""
    # resolve to an absolute path: jobflow may run each job in its own
    # auto-created folder, so a relative work_dir here would not point to
    # the same place once a downstream job (e.g. run_lammps) reads it back
    work_dir = Path(work_dir).resolve()
    work_dir.mkdir(parents=True, exist_ok=True)

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


def get_num(vol_A3, rho=WATER_DENSITY_G_PER_CM3, molmass=WATER_MOLAR_MASS_G_PER_MOL):
    """
    Number of solvent molecules to fill a box of volume `vol_A3` (in A^3)
    at bulk density `rho` (g/cm^3). 1 A^3 = 1e-24 cm^3, so we need that
    conversion here -- the original version multiplied rho*vol directly,
    which silently assumed vol was already in cm^3.
    """
    vol_cm3 = vol_A3 * 1e-24
    mass = rho * vol_cm3
    num = (mass / molmass) * N_A
    num -= 1  # correcting for the volume taken up by the dota molecule
    return round(num)


def _ion_counts(cation_strength, anion_strength, volume_A3):
    """Independent molar concentrations for cation/anion (each 1:1)."""
    n_cation, _ = ion_counts_from_molarity(cation_strength, volume_A3, 1, 0)
    _, n_anion = ion_counts_from_molarity(anion_strength, volume_A3, 0, 1)
    return n_cation, n_anion


def center_molecule(mol_file, box_lengths):
    mol = read(mol_file)
    # Center the molecule in the middle of the box
    mol.center(vacuum=0.0)
    # Shift molecule to the exact center of the box
    mol.translate(np.array(box_lengths) / 2 - mol.get_center_of_mass())
    write(mol_file, mol, format='xyz')


if __name__ == "__main__":

    parser = ArgumentParser(description="Create a water box around a molecule using Packmol")
    parser.add_argument("solute_file", help="XYZ file of the small molecule or atom")
    parser.add_argument("water_file", help="XYZ file of a single water molecule")
    parser.add_argument("output_file", help="Output file for the solvated system")
    parser.add_argument("--box_length", type=float, default=20.0, help="Length of the cubic box in Angstroms (default: 20.0)")
    parser.add_argument("--cation", default="Na", help="type of cation to dissolve")
    parser.add_argument("--anion", default="Cl", help="type of anion to dissolve")
    parser.add_argument("--cation-strength", type=float, default=0.0, help="concentration of cation to dissolve (mol/L)")
    parser.add_argument("--anion-strength", type=float, default=0.0, help="concentration of anion to dissolve (mol/L)")
    parser.add_argument("--work-dir", default=".", help="scratch directory for packmol.inp / intermediate output")
    parser.add_argument("--structures-dir", default=str(DEFAULT_STRUCTURES_DIR),
                        help=f"folder holding <cation>.xyz/<anion>.xyz, used only if ion "
                             f"strengths are nonzero (default: {DEFAULT_STRUCTURES_DIR})")
    args = parser.parse_args()

    box_lengths = np.array([args.box_length, args.box_length, args.box_length])
    vol_A3 = float(np.prod(box_lengths))
    num_water = get_num(vol_A3)
    n_cation, n_anion = _ion_counts(args.cation_strength, args.anion_strength, vol_A3)

    # packmol runs with cwd=work_dir (may differ from the invoking cwd), so
    # relative paths as typed on the CLI must be resolved to absolute first.
    solute_path = Path(args.solute_file).resolve()
    water_path = Path(args.water_file).resolve()
    components = {
        solute_path.stem: {"filepath": str(solute_path), "count": 1},
        water_path.stem: {"filepath": str(water_path), "count": num_water},
    }
    if n_cation:
        components[args.cation] = {
            "filepath": str(Path(args.structures_dir) / ion_xyz_file(args.cation)),
            "count": n_cation,
        }
    if n_anion:
        components[args.anion] = {
            "filepath": str(Path(args.structures_dir) / ion_xyz_file(args.anion)),
            "count": n_anion,
        }

    result = run_packmol(
        components,
        box_lengths,
        work_dir=args.work_dir,
        output_file="solvated.xyz",
    )

    shutil.move(result["packed_xyz"], args.output_file)
    summary = f"Wrote solvated system with {num_water} waters"
    if n_cation:
        summary += f", {n_cation} {args.cation}"
    if n_anion:
        summary += f", {n_anion} {args.anion}"
    print(f"{summary} to {args.output_file}")

    atoms = read(args.output_file)
    atoms.set_cell([args.box_length, args.box_length, args.box_length])
    atoms.set_pbc(True)
    specorder = chemical_symbols[1:103]  
    write("lammps.data", atoms, specorder=specorder, masses=True, format="lammps-data")
