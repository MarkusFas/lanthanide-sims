"""
ion_md/lammps_setup.py

Writes a LAMMPS data file (atom_style atomic) from a packmol-packed
structure, writes the in.lammps run script (PET-MAD / metatomic MLIP
protocol: CG minimization -> NVT equilibration -> NPT production), and
runs LAMMPS.

Atom-type convention: LAMMPS atom type == atomic number Z (1..102). This
matches the fixed `pair_coeff * * 1 2 3 ... 102` and
`dump_modify ... element H He Li ... No` lines in the run script -- because
the mapping is the identity (type N -> element with atomic number N) for
the whole periodic table up to Z=102, those two lines never need to change
between runs, regardless of which elements are actually present in a given
system. The data file always declares all 102 types (with correct masses),
even though only a handful have atoms in them.
"""

import subprocess
from pathlib import Path

from ase.io import read
from ase.data import atomic_masses, atomic_numbers, chemical_symbols
from jobflow import job

N_ELEMENT_TYPES = 102  # matches the fixed pair_coeff / dump_modify element list

# LAMMPS input template. Placeholders use <<NAME>> (not {name}) because the
# script itself uses ${...} (LAMMPS variable syntax) and str.format() would
# collide with that.
LAMMPS_INPUT_TEMPLATE = """\
units metal  # Angstroms, eV, picoseconds
boundary p p p
atom_style atomic
read_data <<DATA_FILE>>
run_style verlet
# loads pet-mad model
pair_style metatomic &
    <<MODEL_PATH>> &
    device <<DEVICE>> &
#Lammps
pair_coeff * * 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 48 49 50 51 52 53 54 55 56 57 58 59 60 61 62 63 64 65 66 67 68 69 70 71 72 73 74 75 76 77 78 79 80 81 82 83 84 85 86 87 88 89 90 91 92 93 94 95 96 97 98 99 100 101 102
neigh_modify one 50000 page 500000 binsize 5.5
neighbor 2.0 bin
variable dt equal <<DT>>
timestep ${dt}
thermo <<THERMO_FREQ>>
thermo_style custom step temp pe etotal press vol
#MINIMIZATION
min_style cg
minimize 1.0e-6 1.0e-8 5000 10000
write_data lammps_min.data
velocity all create <<TEMP>> <<VEL_SEED>> mom yes rot yes
# NVT EQUILIBRATION
fix 1 all temp/csvr <<TEMP>> <<TEMP>> <<TDAMP>> <<TCSVR_SEED>>
fix 2 all nve
restart <<RESTART_FREQ>> restart.equil.*
run <<NVT_STEPS>>
unfix 1
unfix 2
write_data lammps_nvteq.data
# NpT EQUILIBRATION
dump dump_positions all custom <<DUMP_FREQ>> positions.lammpstrj id mass type element xu yu zu vx vy vz
dump_modify dump_positions element H He Li Be B C N O F Ne Na Mg Al Si P S Cl Ar K Ca Sc Ti V Cr Mn Fe Co Ni Cu Zn Ga Ge As Se Br Kr Rb Sr Y Zr Nb Mo Tc Ru Rh Pd Ag Cd In Sn Sb Te I Xe Cs Ba La Ce Pr Nd Pm Sm Eu Gd Tb Dy Ho Er Tm Yb Lu Hf Ta W Re Os Ir Pt Au Hg Tl Pb Bi Po At Rn Fr Ra Ac Th Pa U Np Pu Am Cm Bk Cf Es Fm Md No sort id
dump_modify dump_positions append yes
fix 4 all npt temp <<TEMP>> <<TEMP>> <<TDAMP>> iso <<PRESSURE>> <<PRESSURE>> <<PDAMP>>
restart <<RESTART_FREQ>> restart.prod.*
run <<NPT_STEPS>>
write_data lammps_npteq.data
write_restart restart.npt.final
"""

# Defaults taken straight from the reference in.lammps you sent -- override
# any of these as kwargs to run_lammps(), e.g. run_lammps(..., temp=298.0).
DEFAULT_PARAMS = {
    "MODEL_PATH": "/work/cosmo/fasching/mad-sol/models/pet_sol-s-best_nostress_D3_r20.pt",
    "DEVICE": "cuda",
    "DT": "0.0005",
    "TEMP": "330.0",
    "VEL_SEED": "87287",
    "TCSVR_SEED": "6547",
    "TDAMP": "0.05",
    "PDAMP": "0.5",
    "PRESSURE": "1.0",
    "NVT_STEPS": "20000",
    "NPT_STEPS": "300000",
    "RESTART_FREQ": "5000",
    "DUMP_FREQ": "100",
    "THERMO_FREQ": "100",
}


def write_lammps_data(atoms, box_lengths, data_path: Path) -> Path:
    """atom_style atomic data file; atom type == atomic number (see module docstring)."""
    Lx, Ly, Lz = box_lengths

    lines = [
        "LAMMPS data file (ion_md, atom_style atomic, type == atomic number)\n",
        f"{len(atoms)} atoms",
        f"{N_ELEMENT_TYPES} atom types",
        f"0.0 {Lx:.6f} xlo xhi",
        f"0.0 {Ly:.6f} ylo yhi",
        f"0.0 {Lz:.6f} zlo zhi",
        "",
        "Masses",
        "",
    ]
    for z in range(1, N_ELEMENT_TYPES + 1):
        lines.append(f"{z} {atomic_masses[z]:.4f}  # {chemical_symbols[z]}")

    lines += ["", "Atoms  # atomic", ""]
    for i, atom in enumerate(atoms, start=1):
        z = atomic_numbers[atom.symbol]
        x, y, zc = atom.position
        lines.append(f"{i} {z} {x:.6f} {y:.6f} {zc:.6f}")

    data_path.write_text("\n".join(lines) + "\n")
    return data_path


def write_lammps_input(data_file: str, in_path: Path, **params) -> Path:
    """params override DEFAULT_PARAMS, e.g. write_lammps_input(..., temp=298.0, device='cpu')."""
    merged = {**DEFAULT_PARAMS, **{k.upper(): str(v) for k, v in params.items()}}
    merged["DATA_FILE"] = data_file

    text = LAMMPS_INPUT_TEMPLATE
    for key, value in merged.items():
        text = text.replace(f"<<{key}>>", value)

    in_path.write_text(text)
    return in_path


@job
def run_lammps(packed_xyz: str, box_lengths, work_dir: str = ".", **params) -> dict:
    """
    Job: build lammps.data from the packmol-packed structure, build in.lammps
    from the template above, run lammps, return output paths.
    """
    # resolve to absolute -- jobflow may run this job in a different cwd
    # than the one run_packmol ran in, so relative paths would break
    work_dir = Path(work_dir).resolve()
    work_dir.mkdir(parents=True, exist_ok=True)
    packed_xyz = str(Path(packed_xyz).resolve())

    atoms = read(packed_xyz)
    data_path = write_lammps_data(atoms, box_lengths, work_dir / "lammps.data")
    in_path = write_lammps_input("lammps.data", work_dir / "in.lammps", **params)

    result = subprocess.run(
        ["lmp", "-in", in_path.name], cwd=work_dir,
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"lammps failed:\n{result.stdout}\n{result.stderr}")

    return {
        "log_file": str(work_dir / "log.lammps"),
        "data_file": str(data_path),
        "trajectory": str(work_dir / "positions.lammpstrj"),
        "final_restart": str(work_dir / "restart.npt.final"),
    }
