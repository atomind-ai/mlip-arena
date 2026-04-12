from pathlib import Path

from ase.calculators.calculator import BaseCalculator
from dask.distributed import Client
from dask_jobqueue import SLURMCluster
from huggingface_hub import hf_hub_download
from prefect import flow
from prefect.context import FlowRunContext
from prefect_dask import DaskTaskRunner

from mlip_arena.flows.conservation import differential_entropy_along_nve_trajectory


@flow
def distribution_shifts(calculator: str | BaseCalculator, calculator_kwargs: dict | None = None):
    ctx = FlowRunContext.get()
    parent_task_runner = ctx.task_runner

    work_dir = Path(__file__).parent / "energy_conservation"

    dH, sampled_structures = differential_entropy_along_nve_trajectory.with_options(
        task_runner=parent_task_runner,
    )(
        calculator=calculator,
        calculator_kwargs=calculator_kwargs,
        input_path=Path(__file__).parent / "mptrj_eq_test.extxyz",
        reference_path=hf_hub_download(
            repo_id="atomind/mlip-arena", filename="mptrj_subset.extxyz", repo_type="dataset"
        ),
        start_idx=0,
        end_idx=-1,
        step=100,
        work_dir=work_dir,
    )


if __name__ == "__main__":
    # ==============================================================================
    # 1. JOB CONFIGURATION
    # ==============================================================================

    # Example A: Registered string model available in mlip_arena.models.MLIPENum (e.g., "MACE-MP(M)", "CHGNet")
    calculator = "ORBv2"

    # Example B: Custom ASE Calculator class and arguments (e.g., custom MLIP)
    # from ase.calculators.lj import LennardJones
    # calculator = LennardJones
    # calculator_kwargs = dict(rc=10.0, sigma=1.0, epsilon=0.01)

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

    job_model_name = calculator if isinstance(calculator, str) else calculator.__class__.__name__

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
            f"-J arena:shifts:{job_model_name}",
            f"-q {SLURM_CONFIG['qos']}",
            f"-N {SLURM_CONFIG['nodes']}",
            "-C gpu",
            f"-G {SLURM_CONFIG['gpus_per_node']}",
            "-o /dev/null",  # Redirect stdout to null
            "-e /dev/null",  # Redirect stderr to null
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

    distribution_shifts.with_options(
        task_runner=DaskTaskRunner(address=client.scheduler.address), log_prints=True, persist_result=False
    )(
        calculator=calculator,
        # calculator_kwargs=calculator_kwargs # Uncomment for custom ASE Calculator class
    )
