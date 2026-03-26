"""
================================================================================
MLIP Arena - Benchmark Job Submission Template
================================================================================
This script configures and submits the MLIP Arena core benchmarks
via a Dask-Jobqueue on a SLURM cluster.

INSTRUCTIONS:
1. Provide your model as a registered string OR an ASE calculator instance.
2. Adjust the cluster allocation parameters below.
3. Run directly: `python asymptotes.py`
================================================================================
"""

from pathlib import Path

from ase.calculators.calculator import BaseCalculator
from dask.distributed import Client
from dask_jobqueue import SLURMCluster
from prefect import flow
from prefect.context import FlowRunContext
from prefect_dask import DaskTaskRunner

from mlip_arena.flows.diatomics import homonuclear_diatomics
from mlip_arena.flows.eos_bulk import run_db as EOSFlow
from mlip_arena.flows.ev import run_db as EVFlow
from mlip_arena.models import REGISTRY, MLIPEnum


@flow
def asymptotic_behaviors(model: str | BaseCalculator):
    ctx = FlowRunContext.get()
    parent_task_runner = ctx.task_runner

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
    out_dir_diatomics = Path.cwd() / "diatomics" / family / model_name
    homonuclear_diatomics.with_options(name=f"diatomics-{model_name}", task_runner=parent_task_runner)(
        model=model, run_dir=out_dir_diatomics
    )

    # 2. EOS Bulk
    print(f"Starting EOS bulk benchmark for {model_name}...")
    out_dir_eos = Path.cwd() / "eos_bulk"
    EOSFlow.with_options(name=f"eos_bulk-{model_name}", task_runner=parent_task_runner)(
        model=model, run_dir=out_dir_eos, dataset_file="wbm_subset.db"
    )

    # 3. E-V
    print(f"Starting E-V scan benchmark for {model_name}...")
    out_dir_ev = Path.cwd() / "ev"
    EVFlow.with_options(name=f"ev-{model_name}", task_runner=parent_task_runner)(
        model=model, run_dir=out_dir_ev, dataset_file="wbm_subset.db"
    )


if __name__ == "__main__":
    # ==============================================================================
    # 1. JOB CONFIGURATION
    # ==============================================================================

    # Example A: Registered string model (e.g., "MACE-MP(M)", "CHGNet")
    MODEL = "NequIP-OAM-L"

    # Example B: Custom ASE Calculator
    # from mace.calculators import mace_mp
    # MODEL = mace_mp(model="medium", dispersion=False, default_dtype="float64", device="cuda")

    # SLURM environment configuration
    SLURM_CONFIG = {
        "nodes": 1,
        "gpus_per_node": 1,
        "account": "matgen",
        "qos": "preempt",
        "walltime": "02:00:00",
        # Example virutal environment (must modify)
        "job_script_prologue": [
            "source ~/.bashrc",
            "module load python",
            "module load cudatoolkit/12.4",
            "source activate /pscratch/sd/c/cyrusyc/.conda/mlip-arena",
        ],
    }

    job_model_name = MODEL if isinstance(MODEL, str) else MODEL.__class__.__name__

    cluster_kwargs = dict(
        cores=1,
        memory="64 GB",
        processes=1,
        shebang="#!/bin/bash",
        account=SLURM_CONFIG["account"],
        walltime=SLURM_CONFIG["walltime"],
        job_script_prologue=SLURM_CONFIG["job_script_prologue"],
        job_directives_skip=["-n", "--cpus-per-task", "-J"],
        job_extra_directives=[
            f"-J benchmark_{job_model_name}",
            f"-q {SLURM_CONFIG['qos']}",
            f"-N {SLURM_CONFIG['nodes']}",
            "-C gpu",
            f"-G {SLURM_CONFIG['gpus_per_node']}",
            # "-o /dev/null",  # Redirect stdout to null
            # "-e /dev/null",  # Redirect stderr to null
        ],
    )

    cluster = SLURMCluster(**cluster_kwargs)
    print("--------------------------------------------------------------------------------")
    print(f"Generating SLURM cluster jobs with script:\n{cluster.job_script()}")
    print("--------------------------------------------------------------------------------")

    cluster.adapt(minimum_jobs=1, maximum_jobs=50)
    client = Client(cluster)

    print(f"Dask dashboard available at: {client.dashboard_link}")

    # ==============================================================================
    # 2. JOB EXECUTION
    # ==============================================================================

    asymptotic_behaviors.with_options(
        task_runner=DaskTaskRunner(address=client.scheduler.address),
        log_prints=True,
    )(model=MODEL)
