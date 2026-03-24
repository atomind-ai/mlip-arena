from pathlib import Path

from ase.calculators.calculator import BaseCalculator
from dask.distributed import Client
from dask_jobqueue import SLURMCluster
from prefect import flow
from prefect_dask import DaskTaskRunner

from mlip_arena.flows.diatomics import homonuclear_diatomics
from mlip_arena.flows.eos_bulk import wbm_eos_bulk
from mlip_arena.flows.ev import run as wbm_ev_scan
from mlip_arena.models import REGISTRY, MLIPEnum


@flow
def benchmark_model(model: str | BaseCalculator):
    """Run all core benchmarks sequentially for a given model."""
    if isinstance(model, BaseCalculator):
        model_name = model.__class__.__name__
        family = "custom"
    else:
        model_name = model
        if model_name not in MLIPEnum.__members__:
            print(f"Model {model_name} is not in MLIPEnum natively. Using directly.")
        family = REGISTRY[model_name]["family"] if model_name in REGISTRY else "custom"

    # 1. Diatomics
    print(f"Starting homonuclear diatomics benchmark for {model_name}...")
    # NOTE: diatomics flow does not iterate over db_path, it generates molecules
    # and internally writes {model_name}.json to the parents of the output directory
    # to maintain compatibility with `serve/rank/homonuclear-diatomics.py`
    out_dir_diatomics = Path.cwd() / "diatomics" / family
    homonuclear_diatomics.with_options(
        name=f"diatomics-{model_name}", refresh_cache=True
    )(model=model_name, run_dir=out_dir_diatomics)

    # 2. EOS Bulk
    print(f"Starting EOS bulk benchmark for {model_name}...")
    out_dir_eos = Path.cwd() / "eos_bulk"
    wbm_eos_bulk.with_options(name=f"eos_bulk-{model_name}", refresh_cache=True)(
        model=model_name, run_dir=out_dir_eos
    )

    # 3. E-V
    print(f"Starting E-V scan benchmark for {model_name}...")
    out_dir_ev = Path.cwd() / "ev"
    wbm_ev_scan.with_options(name=f"ev-{model_name}", refresh_cache=True)(
        model=model_name, run_dir=out_dir_ev
    )


if __name__ == "__main__":
    # ------------------------------------------------------------------------
    # USER TEMPLATE
    # Modify the parameters below to run the benchmark for your model
    # ------------------------------------------------------------------------

    # 1. Model Configuration
    # You can provide a registered string name OR an ASE BaseCalculator.
    # String example:
    # model = "MACE-MP(M)"
    #
    # ASE Calculator example:
    # from mace.calculators import mace_mp
    # model = mace_mp(model="medium", dispersion=False, default_dtype="float64", device="cuda")

    model = "MACE-MP(M)"

    # 2. SLURM Allocation Configuration
    nodes = 1
    gpus_per_node = 1
    account = "m3828"
    qos = "regular"
    walltime = "04:00:00"
    # ------------------------------------------------------------------------

    # Infer job name safely based on model type
    job_model_name = model if isinstance(model, str) else model.__class__.__name__

    cluster_kwargs = dict(
        cores=1,
        memory="64 GB",
        processes=1,
        shebang="#!/bin/bash",
        account=account,
        walltime=walltime,
        job_script_prologue=[
            "source ~/.bashrc",
            "module load python",
            "module load cudatoolkit/12.4",
            "source activate /pscratch/sd/c/cyrusyc/.conda/mlip-arena",
        ],
        job_directives_skip=["-n", "--cpus-per-task", "-J"],
        job_extra_directives=[
            f"-J benchmark_{job_model_name}",
            f"-q {qos}",
            f"-N {nodes}",
            "-C gpu",
            f"-G {gpus_per_node}",
            # "--exclusive",
        ],
    )

    cluster = SLURMCluster(**cluster_kwargs)
    print("--------------------------------------------------")
    print(f"Generating SLURM cluster jobs with script:\n{cluster.job_script()}")
    print("--------------------------------------------------")

    # Scale appropriately to minimum jobs or let it adapt
    cluster.adapt(minimum_jobs=1, maximum_jobs=50)
    client = Client(cluster)

    print(f"Dask dashboard available at: {client.dashboard_link}")

    # Launch Prefect flow using Dask task runner
    # The run_dir logic expects this script to be run from `mlip-arena/benchmarks/` root
    benchmark_model.with_options(
        task_runner=DaskTaskRunner(address=client.scheduler.address),
        log_prints=True,
    )(model=model)
