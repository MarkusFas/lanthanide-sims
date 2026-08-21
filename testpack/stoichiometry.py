"""
stoichiometry.py

Estimate ion (+ counterion) and solvent molecule counts for a packmol box,
given a target molarity and either:
  (a) a fixed box size            -> "3M NaCl in oxidane with this box size"
  (b) a fixed number of ion atoms -> "60 atoms of Na, 2M, in whatever box size"

Solvents are looked up from i-solvate/initial_structures/solvents.csv, e.g.:

    Identifier;PUBCHEM_CID;Charge;Density(g/cm3);IUPAC_Name
    Water;962;0;0.995;oxidane
    Ethanol;702;0;0.7893;ethanol
    ...

Lookup matches against either the "Identifier" column (e.g. "Water") or the
"IUPAC_Name" column (e.g. "oxidane"), case-insensitively.

solvents.csv is solvents only -- ions are NOT looked up from it. An ion's
.xyz filename is just "<cation_name>.xyz" / "<anion_name>.xyz" directly.
Both solvent and ion structure files are assumed to live in `structures_dir`
(defaults to solvents_csv's parent folder). Stoichiometry (cation:anion
ratio) defaults to 1:1 -- pass explicitly for non-1:1 salts (e.g. CaCl2 ->
cation_stoich=1, anion_stoich=2).

The CSV has no molar-mass column, so solvent molar masses are kept in
MOLAR_MASS_G_PER_MOL below, keyed by IUPAC_Name. Add an entry there for any
new solvent you add to the CSV.

Output shape: solvate_molecule() returns
    {
        "box_lengths": (Lx, Ly, Lz),
        "volume_A3": float,
        "components": {
            "<solvent iupac_name>": {"filepath": ..., "count": n_solvent},
            "<solute name>":        {"filepath": ..., "count": n_solute},
            "<cation_name>":        {"filepath": ..., "count": n_cation},
            "<anion_name>":         {"filepath": ..., "count": n_anion},
        },
    }
so downstream (packmol.py) can stay agnostic to what's actually in solution
-- it just loops over `components`.

APPROXIMATION: ion excluded volume is ignored when filling the box with
solvent -- solvent count is computed from the *full* box volume at the pure
solvent density. This slightly overestimates solvent count at high
concentration; it's meant to give packmol a reasonable starting box, refined
later by NPT equilibration in LAMMPS.
"""
import csv
from dataclasses import dataclass
from pathlib import Path

N_A = 6.02214076e23      # mol^-1
L_TO_A3 = 1e27            # 1 L  = 1e27 A^3
ML_TO_A3 = 1e24           # 1 mL = 1e24 A^3 (== 1 cm^3)

# Solvent molar masses (g/mol), keyed by lowercase IUPAC_Name from
# solvents.csv. Not in the CSV, so tracked here -- add new solvents here too.
MOLAR_MASS_G_PER_MOL = {
    "hexane": 86.178,
    "oxolane": 72.107,                      # THF
    "ethoxyethane": 74.123,                 # diethyl ether
    "methylsulfinylmethane": 78.129,        # DMSO
    "tetrachloromethane": 153.823,          # CCl4
    "methanol": 32.042,
    "benzene": 78.114,
    "oxidane": 18.015,                      # water
    "ethanol": 46.069,
    "1,3-dioxolan-2-one": 88.062,           # ethylene carbonate
    "ethyl-methyl-carbonate": 104.105,
    "acetic-acid": 60.052,
    "azane": 17.031,                        # ammonia
    "propan-2-ol": 60.096,                  # isopropanol
    "n,n-dimethylformamide": 73.095,        # DMF
    "toluene": 92.141,
    "chloroform": 119.378,
}


@dataclass
class SolventProperties:
    identifier: str        # human-friendly name, e.g. "Water"
    iupac_name: str        # e.g. "oxidane" -- also the .xyz filename stem
    density: float          # g/mL (== g/cm3)
    molar_mass: float       # g/mol
    xyz_file: str


def load_solvent_properties(solvents_csv, name: str) -> SolventProperties:
    """Look up a solvent's density (from CSV) + molar mass (from the table above)."""
    solvents_csv = Path(solvents_csv)
    with open(solvents_csv, newline="") as f:
        reader = csv.DictReader(f, delimiter=";")
        fieldmap = {k.strip().lower(): k for k in reader.fieldnames}
        required = ["identifier", "iupac_name", "density(g/cm3)"]
        missing = [col for col in required if col not in fieldmap]
        if missing:
            raise KeyError(
                f"{solvents_csv} is missing expected column(s) {missing}. "
                f"Found columns: {reader.fieldnames}."
            )
        for row in reader:
            identifier = row[fieldmap["identifier"]].strip()
            iupac_name = row[fieldmap["iupac_name"]].strip()
            if name.lower() in (identifier.lower(), iupac_name.lower()):
                density = float(row[fieldmap["density(g/cm3)"]])
                key = iupac_name.lower()
                if key not in MOLAR_MASS_G_PER_MOL:
                    raise KeyError(
                        f"No molar mass on file for '{iupac_name}'. "
                        "Add it to MOLAR_MASS_G_PER_MOL in stoichiometry.py."
                    )
                return SolventProperties(
                    identifier=identifier,
                    iupac_name=iupac_name,
                    density=density,
                    molar_mass=MOLAR_MASS_G_PER_MOL[key],
                    xyz_file=f"{iupac_name}.xyz",
                )
    raise ValueError(f"Solvent '{name}' not found in {solvents_csv}")


def ion_xyz_file(name: str) -> str:
    """Ion .xyz filename -- direct convention, not looked up from solvents.csv."""
    return f"{name}.xyz"


def box_volume_A3(box_lengths) -> float:
    """box_lengths: (Lx, Ly, Lz) in Angstrom -> volume in Angstrom^3."""
    Lx, Ly, Lz = box_lengths
    return Lx * Ly * Lz


def cubic_box_edge(volume_A3: float) -> float:
    """Edge length (Angstrom) of a cube with the given volume."""
    return volume_A3 ** (1 / 3)


def volume_from_n_solvent(n_solvent: int, solvent: SolventProperties) -> float:
    """Inverse of n_solvent_from_volume: box volume (A^3) holding n_solvent
    molecules of this solvent at its pure density."""
    mass_g = n_solvent * solvent.molar_mass / N_A
    volume_mL = mass_g / solvent.density
    return volume_mL * ML_TO_A3


def ion_counts_from_molarity(
    molarity: float,
    volume_A3: float,
    cation_stoich: int = 1,
    anion_stoich: int = 1,
):
    """
    molarity: mol salt formula units / L of solution (e.g. 3.0 for 3M NaCl)
    volume_A3: box volume in Angstrom^3
    cation_stoich / anion_stoich: stoichiometric coefficients of the salt,
        e.g. NaCl -> (1, 1), CaCl2 -> (1, 2), Na2SO4 -> (2, 1)

    Returns (n_cation, n_anion) as molecule counts.
    """
    volume_L = volume_A3 / L_TO_A3
    n_formula_units = molarity * volume_L * N_A
    n_cation = round(n_formula_units * cation_stoich)
    n_anion = round(n_formula_units * anion_stoich)
    return n_cation, n_anion


def volume_from_ion_count(n_cation: int, molarity: float, cation_stoich: int = 1) -> float:
    """Given a fixed cation count and target molarity, back out box volume (A^3)."""
    n_formula_units = n_cation / cation_stoich
    volume_L = n_formula_units / (molarity * N_A)
    return volume_L * L_TO_A3


def n_solvent_from_volume(volume_A3: float, solvent: SolventProperties) -> int:
    """Number of solvent molecules to fill volume_A3 at the solvent's pure density."""
    volume_mL = volume_A3 / ML_TO_A3
    mass_g = volume_mL * solvent.density
    return round(mass_g / solvent.molar_mass * N_A)


def _resolve_structures_dir(structures_dir, solvents_csv) -> Path:
    # resolve to absolute: this dict gets embedded into a job's arguments and
    # read back much later (possibly from a different working directory), so
    # a relative path here would silently point at the wrong place downstream
    path = Path(structures_dir) if structures_dir else Path(solvents_csv).parent
    return path.resolve()


def _build_components(
    structures_dir: Path,
    solvent: SolventProperties,
    n_solvent: int,
    cation_name: str,
    n_cation: int,
    anion_name: str,
    n_anion: int,
) -> dict:
    return {
        solvent.iupac_name: {
            "filepath": str(structures_dir / solvent.xyz_file),
            "count": n_solvent,
        },
        cation_name: {
            "filepath": str(structures_dir / ion_xyz_file(cation_name)),
            "count": n_cation,
        },
        anion_name: {
            "filepath": str(structures_dir / ion_xyz_file(anion_name)),
            "count": n_anion,
        },
    }


def solvate_molecule(
    solvent_name: str,
    solute_xyz: str,
    solvents_csv,
    solute_name: str | None = None,
    n_solute: int = 1,
    box_lengths=None,
    n_solvent: int | None = None,
    n_cation: int = 0,
    n_anion: int = 0,
    cation_name: str | None = None,
    anion_name: str | None = None,
    structures_dir=None,
) -> dict:
    """
    Pack n_solute copies of an arbitrary small molecule (structure file anywhere
    on disk -- independent of solvents_csv/structures_dir) into a box of solvent,
    optionally with some ions dissolved too.

    Give exactly one of `box_lengths` or `n_solvent`:
      - box_lengths: fixed box size -> solvent count is computed to fill it
      - n_solvent:   fixed solvent molecule count -> box size is computed to hold it

    Ions are given as raw counts (not molarity) -- pass n_cation/n_anion directly
    together with cation_name/anion_name. Leave both at 0 for no ions.

    e.g. fixed box, no ions:
        solvate_molecule("oxidane", "/home/fasching/molecules/caffeine.xyz",
                          solvents_csv, box_lengths=(40, 40, 40))

    e.g. fixed water count, with ions:
        solvate_molecule("oxidane", "/home/fasching/molecules/caffeine.xyz",
                          solvents_csv, n_solvent=2000,
                          n_cation=5, n_anion=5, cation_name="Na", anion_name="Cl")
    """
    if (box_lengths is None) == (n_solvent is None):
        raise ValueError("give exactly one of box_lengths or n_solvent")
    if (n_cation or n_anion) and not (cation_name and anion_name):
        raise ValueError("cation_name/anion_name required when n_cation/n_anion are given")

    solvent = load_solvent_properties(solvents_csv, solvent_name)
    structures_dir = _resolve_structures_dir(structures_dir, solvents_csv)

    solute_path = Path(solute_xyz).resolve()
    if not solute_path.is_file():
        raise FileNotFoundError(f"solute structure file not found: {solute_path}")
    if solute_name is None:
        solute_name = solute_path.stem

    if box_lengths is not None:
        volume = box_volume_A3(box_lengths)
        n_solvent = n_solvent_from_volume(volume, solvent)
    else:
        volume = volume_from_n_solvent(n_solvent, solvent)
        edge = cubic_box_edge(volume)
        box_lengths = (edge, edge, edge)

    components = {
        solvent.iupac_name: {"filepath": str(structures_dir / solvent.xyz_file), "count": n_solvent},
        solute_name: {"filepath": str(solute_path), "count": n_solute},
    }
    if n_cation or n_anion:
        components[cation_name] = {"filepath": str(structures_dir / ion_xyz_file(cation_name)), "count": n_cation}
        components[anion_name] = {"filepath": str(structures_dir / ion_xyz_file(anion_name)), "count": n_anion}

    return {
        "box_lengths": tuple(box_lengths),
        "volume_A3": volume,
        "components": components,
    }
