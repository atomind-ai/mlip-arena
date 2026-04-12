from asymptotes import asymptotic_behaviors
from dask.distributed import Client
from dask_jobqueue import SLURMCluster
from prefect_dask import DaskTaskRunner

# ==============================================================================
# 1. JOB CONFIGURATION
# ==============================================================================

# Example A: Registered string model (e.g., "MACE-MP(M)", "CHGNet")
calculator = "NequIP-OAM-L"

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
    persist_result=False,
)(
    calculator=calculator,
    # calculator_kwargs=calculator_kwargs # Uncomment for custom ASE Calculator class
)

# TODO: Add Distribution Shift, Stability and Reactivity
