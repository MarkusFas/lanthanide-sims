"""
ion_md/composition.py

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

Output shape: solvate_from_box() / solvate_from_ion_count() both return

    {
        "box_lengths": (Lx, Ly, Lz),
        "volume_A3": float,
        "components": {
            "<solvent iupac_name>": {"filepath": ..., "count": n_solvent},
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
                        "Add it to MOLAR_MASS_G_PER_MOL in composition.py."
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


def solvate_from_box(
    molarity: float,
    box_lengths,
    solvent_name: str,
    cation_name: str,
    anion_name: str,
    solvents_csv,
    cation_stoich: int = 1,
    anion_stoich: int = 1,
    structures_dir=None,
) -> dict:
    """
    Case: fixed box size, target molarity.
    e.g. solvate_from_box(3.0, (40, 40, 40), "oxidane", "Na", "Cl", solvents_csv)
    """
    solvent = load_solvent_properties(solvents_csv, solvent_name)
    structures_dir = _resolve_structures_dir(structures_dir, solvents_csv)

    volume = box_volume_A3(box_lengths)
    n_cation, n_anion = ion_counts_from_molarity(molarity, volume, cation_stoich, anion_stoich)
    n_solvent = n_solvent_from_volume(volume, solvent)

    components = _build_components(
        structures_dir, solvent, n_solvent, cation_name, n_cation, anion_name, n_anion
    )

    return {
        "box_lengths": tuple(box_lengths),
        "volume_A3": volume,
        "components": components,
    }


def solvate_from_ion_count(
    molarity: float,
    n_cation: int,
    solvent_name: str,
    cation_name: str,
    anion_name: str,
    solvents_csv,
    cation_stoich: int = 1,
    anion_stoich: int = 1,
    structures_dir=None,
) -> dict:
    """
    Case: fixed number of cations, target molarity, box size computed
    (returned as a cubic edge length).
    e.g. solvate_from_ion_count(2.0, 60, "oxidane", "Na", "Cl", solvents_csv)
    """
    solvent = load_solvent_properties(solvents_csv, solvent_name)
    structures_dir = _resolve_structures_dir(structures_dir, solvents_csv)

    n_formula_units = n_cation / cation_stoich
    n_anion = round(n_formula_units * anion_stoich)

    volume = volume_from_ion_count(n_cation, molarity, cation_stoich)
    box_edge = cubic_box_edge(volume)
    n_solvent = n_solvent_from_volume(volume, solvent)

    components = _build_components(
        structures_dir, solvent, n_solvent, cation_name, n_cation, anion_name, n_anion
    )

    return {
        "box_lengths": (box_edge, box_edge, box_edge),
        "volume_A3": volume,
        "components": components,
    }


if __name__ == "__main__":
    solvents_csv = "../../initial_structures/solvents.csv"

    # "3M NaCl in oxidane with this box size"
    result_a = solvate_from_box(
        molarity=3.0,
        box_lengths=(40.0, 40.0, 40.0),
        solvent_name="oxidane",
        cation_name="Na",
        anion_name="Cl",
        solvents_csv=solvents_csv,
    )
    print("fixed box:", result_a)

    # "60 atoms of Na, 2M, in whatever box size"
    result_b = solvate_from_ion_count(
        molarity=2.0,
        n_cation=60,
        solvent_name="oxidane",
        cation_name="Na",
        anion_name="Cl",
        solvents_csv=solvents_csv,
    )
    print("fixed ion count:", result_b)
