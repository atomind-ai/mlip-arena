from asymptotes import asymptotic_behaviors
from dask.distributed import Client
from dask_jobqueue import SLURMCluster
from prefect_dask import DaskTaskRunner

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

asymptotic_behaviors.with_options(
    task_runner=DaskTaskRunner(address=client.scheduler.address),
    log_prints=True,
)(model=MODEL)

# TODO: Add Distribution Shift, Stability and Reactivity
