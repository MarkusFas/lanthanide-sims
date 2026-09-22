"""
packmol.py

Build and run a packmol job that solvates a solute molecule in a solvent box,
optionally with a target ionic concentration, using the counts worked out by
stoichiometry.py's solvate_molecule().

Typical use:
    python packmol.py molecule.xyz solvated.xyz \\
        --solvent oxidane \\
        --box_length 40 \\
        --cation Na --cation-strength 0.15 \\
        --anion Cl --anion-strength 0.15

--solvents-csv defaults to i-solvate/initial_structures/solvents.csv (i.e.
initial_structures/solvents.csv next to this file) -- pass --solvents-csv to
override. Ion structure files (Na.xyz, Cl.xyz, ...) and the solvent's
<iupac_name>.xyz are assumed to live in --structures-dir (defaults to the
solvents.csv folder), per stoichiometry.py's convention.

Notes on what changed vs. the draft this was assembled from:
  - STRUCTURE_BLOCK was referenced but never defined -- added below.
  - `from pathlib import Path` was missing (Path was already used).
  - The old `generate_packmol_input()` (fixed/centered solute + water-only)
    is superseded by write_packmol_input()/run_packmol(), which handle any
    mix of components generically -- removed to avoid two competing paths.
  - `get_num()` relied on undefined globals (`rho`, `molmass`) and is fully
    superseded by stoichiometry.py's n_solvent_from_volume/ion_counts_from_molarity
    -- removed.
  - `center_molecule()` is kept below as an optional standalone helper (e.g.
    for pre-centering a solute .xyz for some other purpose) but is not
    called by default: every component, solute included, is packed with an
    "inside box" placement, so packmol itself handles positioning.
  - --cation-strength / --anion-strength are treated as independent molar
    concentrations (mol/L) of each ion, not a single salt molarity: each is
    fed through stoichiometry.ion_counts_from_molarity() on its own (with
    the other species' stoichiometry set to 0), so e.g. asymmetric or
    single-ion setups both work. For a proper 1 salt with stoichiometry
    (e.g. CaCl2), set --cation-strength/--anion-strength to the same salt
    molarity and use --cation-stoich/--anion-stoich to get the 1:2 ratio.
"""
from pathlib import Path
import subprocess
from argparse import ArgumentParser

from ase.io import read, write

from stoichiometry import solvate_molecule, box_volume_A3, ion_counts_from_molarity

# Hardcoded default location for solvents.csv, per stoichiometry.py's own
# convention (its docstring assumes i-solvate/initial_structures/solvents.csv).
# Resolved relative to this file so it works regardless of cwd -- this file
# lives in i-solvate/, so the default is i-solvate/initial_structures/solvents.csv.
# Override with --solvents-csv if yours lives somewhere else.
DEFAULT_SOLVENTS_CSV = Path(__file__).resolve().parent / "initial_structures" / "solvents.csv"

PACKMOL_HEADER = """\
tolerance {tolerance}
filetype xyz
output {output_file}
seed {seed}
pbc {Lx} {Ly} {Lz}
"""

STRUCTURE_BLOCK = """
structure {filepath}
  number {count}
  inside box 0. 0. 0. {Lx} {Ly} {Lz}
end structure
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
    of entries, any mix of solute/solvents/ions. Writes one structure block
    per entry, all packed into the same box.
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


def run_packmol(
    components: dict,
    box_lengths,
    work_dir: str = ".",
    output_file: str = "solvated.xyz",
    tolerance: float = 2.0,
    seed: int = 12345,
) -> dict:
    """Job: write packmol input from `components`, run packmol, return the packed structure."""
    # resolve to an absolute path: jobflow may run each job in its own
    # auto-created folder, so a relative work_dir here would not point to
    # the same place once a downstream job (e.g. run_lammps) reads it back
    work_dir = Path(work_dir).resolve()
    work_dir.mkdir(parents=True, exist_ok=True)
    inp_path = write_packmol_input(
        components, box_lengths, work_dir,
        output_file=output_file, tolerance=tolerance, seed=seed,
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


def center_molecule(mol_file, box_lengths):
    """Optional helper: recenter a structure file's center of mass on a box's
    center. Not called by default -- packmol places every component itself
    via "inside box"."""
    mol = read(mol_file)
    mol.center(vacuum=0.0)
    mol.translate(box_lengths / 2 - mol.get_center_of_mass())
    write(mol_file, mol, format="xyz")


def _ion_counts(cation_strength, anion_strength, volume_A3, cation_stoich, anion_stoich):
    """Independent molar concentrations for cation/anion (see module docstring)."""
    n_cation, _ = ion_counts_from_molarity(cation_strength, volume_A3, cation_stoich, 0)
    _, n_anion = ion_counts_from_molarity(anion_strength, volume_A3, 0, anion_stoich)
    return n_cation, n_anion


if __name__ == "__main__":
    parser = ArgumentParser(description="Solvate a molecule in a solvent box (+ optional ionic strength) with Packmol")
    parser.add_argument("solute_file", help="XYZ file of the small molecule/solute")
    parser.add_argument("output_file", help="Output file for the solvated system (written into --work-dir)")
    parser.add_argument("--solvent", required=True, help="Solvent name, matched against solvents.csv's Identifier or IUPAC_Name (e.g. 'Water' or 'oxidane')")
    parser.add_argument("--solvents-csv", default=str(DEFAULT_STRUCTURES
                                                      / solvents.csv), help=f"Path to solvents.csv (see stoichiometry.py for its expected columns; default: {DEFAULT_SOLVENTS_CSV})")
    parser.add_argument("--structures-dir", default=DEFAULT_STRUCTURES, help="Folder holding <solvent>.xyz / <ion>.xyz files (default: solvents-csv's parent folder)")
    parser.add_argument("--box_length", type=float, default=20.0, help="Length of the cubic box in Angstroms (default: 20.0)")
    parser.add_argument("--cation", default="Na", help="Cation name; its structure file is '<cation>.xyz' in --structures-dir")
    parser.add_argument("--anion", default="Cl", help="Anion name; its structure file is '<anion>.xyz' in --structures-dir")
    parser.add_argument("--cation-strength", type=float, default=0.0, help="Cation concentration to dissolve, mol/L (default: 0.0)")
    parser.add_argument("--anion-strength", type=float, default=0.0, help="Anion concentration to dissolve, mol/L (default: 0.0)")
    parser.add_argument("--cation-stoich", type=int, default=1, help="Cation stoichiometric coefficient of the salt (default: 1)")
    parser.add_argument("--anion-stoich", type=int, default=1, help="Anion stoichiometric coefficient of the salt (default: 1)")
    parser.add_argument("--tolerance", type=float, default=2.0, help="Packmol tolerance in Angstroms (default: 2.0)")
    parser.add_argument("--seed", type=int, default=12345, help="Packmol random seed (default: 12345)")
    parser.add_argument("--work-dir", default=".", help="Directory to write packmol.inp and the output structure into (default: current directory)")
    args = parser.parse_args()

    box_lengths = (args.box_length, args.box_length, args.box_length)
    volume_A3 = box_volume_A3(box_lengths)

    n_cation, n_anion = _ion_counts(
        args.cation_strength, args.anion_strength, volume_A3,
        args.cation_stoich, args.anion_stoich,
    )

    solvated = solvate_molecule(
        solvent_name=args.solvent,
        solute_xyz=args.solute_file,
        solvents_csv=args.solvents_csv,
        box_lengths=box_lengths,
        n_cation=n_cation,
        n_anion=n_anion,
        cation_name=args.cation if n_cation else None,
        anion_name=args.anion if n_anion else None,
        structures_dir=args.structures_dir,
    )

    print(f"Box: {solvated['box_lengths']} A, volume {solvated['volume_A3']:.1f} A^3")
    for name, spec in solvated["components"].items():
        print(f"  {name}: {spec['count']}")

    result = run_packmol(
        solvated["components"],
        solvated["box_lengths"],
        work_dir=args.work_dir,
        output_file=args.output_file,
        tolerance=args.tolerance,
        seed=args.seed,
    )
    print(f"Wrote packed structure to {result['packed_xyz']}")
