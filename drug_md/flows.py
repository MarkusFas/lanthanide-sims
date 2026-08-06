"""
drug_md/flows.py

Wires the composition -> packmol -> lammps -> postprocess jobs into a Flow.
"""

from jobflow import Flow
from drug_md import run_packmol
from drug_md import run_lammps
from drug_md.postprocess import run_postprocessing


def build_solvation_flow(
    composition: dict,
    model_path: str,
    work_dir: str = ".",
    lammps_params: dict | None = None,
    extra_metadata: dict | None = None,
) -> Flow:
    """
    composition: output of ion_md.composition.solvate_from_box /
        solvate_from_ion_count, i.e.
        {"molarity": ..., "box_lengths": ..., "components": {...}}
    model_path: path to the MLIP model run_lammps should use. Lives here
        (not in `composition`) because which MLIP to use for the MD run is
        unrelated to how the box gets packed.
    lammps_params: any other run_lammps overrides (temp, device, nvt_steps,
        ...), see lammps_setup.DEFAULT_PARAMS for the full list.
    extra_metadata: anything to tag onto the Flow *in addition* to what's
        auto-derived below (molarity, box, components, model_path). Don't
        re-pass molarity/model_path/etc. here -- they're already pulled
        from `composition` and `model_path` so there's exactly one place
        each value is set, instead of risking the two drifting apart.
    """
    lammps_params = dict(lammps_params or {})
    lammps_params.setdefault("model_path", model_path)

    pack_job = run_packmol(
        components=composition["components"],
        box_lengths=composition["box_lengths"],
        work_dir=work_dir,
    )
    lammps_job = run_lammps(
        packed_xyz=pack_job.output["packed_xyz"],
        box_lengths=pack_job.output["box_lengths"],
        work_dir=work_dir,
        **lammps_params,
    )
    postprocess_job = run_postprocessing(
        log_file=lammps_job.output["log_file"],
        trajectory=lammps_job.output["trajectory"],
        work_dir=work_dir,
    )

    flow = Flow(
        [pack_job, lammps_job, postprocess_job],
        output=postprocess_job.output,
    )

    metadata = {
        "model_path": model_path,
        "molarity": composition.get("molarity"),
        "box_lengths": composition["box_lengths"],
        "component_counts": {
            name: spec["count"] for name, spec in composition["components"].items()
        },
    }
    if extra_metadata:
        metadata.update(extra_metadata)
    flow.update_metadata(metadata)

    return flow
