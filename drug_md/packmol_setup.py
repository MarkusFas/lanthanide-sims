"""
ion_md/packmol.py

Agnostic packmol input writer + job. Doesn't know or care whether a
component is a solvent, cation, or anion -- it just takes a `components`
dict of {name: {"filepath": ..., "count": ...}} (the shape produced by
ion_md.composition.solvate_from_box / solvate_from_ion_count) and writes
one packmol `structure` block per entry.
"""

import subprocess
from pathlib import Path
from jobflow import job

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


@job
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
