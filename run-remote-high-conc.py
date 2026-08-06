import subprocess
from pathlib import Path
from jobflow_remote import submit_flow, set_run_config
from ion_md.composition import solvate_from_ion_count
from ion_md.flows import build_solvation_flow

WORKER_HOSTS = {
    "cpu_worker": ("jed.hpc.epfl.ch", "/scratch/fasching/petsol_highconc/jfremote_run"),
    "gpu_worker": ("kuma.hpc.epfl.ch", "/scratch/fasching/petsol_highconc/jfremote_run"),
}

def ensure_worker_dirs():
    for worker, (host, work_dir) in WORKER_HOSTS.items():
        subprocess.run(["ssh", host, f"mkdir -p {work_dir}"], check=True)

ensure_worker_dirs()

solvents_csv = "/work/cosmo/fasching/PET-SOL/i-solvate/initial_structures/solvents.csv"
models_dir = Path("/work/cosmo/malosso/petsol/models")

models = ["pet_sol-s-best_nostress_D3_r20.pt"]#, "pet_sol-other_variant.pt"]
ions = [("Na", "Cl")]
molarities = [1.0]

for model_file in models:
    for cation, anion in ions:
        for molarity in molarities:
            anion_stoich = 2 if cation == "Ca" else 1
            composition = solvate_from_ion_count(
                molarity=molarity,
                n_cation=60,
                solvent_name="oxidane",
                cation_name=cation,
                anion_name=anion,
                anion_stoich=anion_stoich,
                solvents_csv=solvents_csv,
            )

            model_name = Path(model_file).stem
            run_name = f"{cation}{anion}_{molarity}M_{model_name}"
            flow = build_solvation_flow(
                composition,
                model_path=str(models_dir / model_file),
                work_dir=f"/scratch/fasching/pet-sol/high-conc/runs/{run_name}",
            )

            flow = set_run_config(
                flow, name_filter="run_packmol", worker="cpu_worker",
                resources={"partition": "standard", "ntasks": 1, "time": "00:15:00"},
                exec_config="packmol_env",
            )
            flow = set_run_config(
                flow, name_filter="run_lammps", worker="gpu_worker",
                resources={"nodes": 1, "ntasks": 1, "partition": "gpu", "qverbatim": "#SBATCH --gres=gpu:1"},
                exec_config="lammps_gpu_env",
            )
            flow = set_run_config(
                flow, name_filter="run_postprocessing", worker="cpu_worker",
                resources={"time": "00:15:00"},
                exec_config="postprocess_env",
            )

            submit_flow(flow, worker="cpu_worker")
