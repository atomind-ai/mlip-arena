from __future__ import annotations

from pathlib import Path

from mlip_arena.models import REGISTRY, MLIPEnum
from mlip_arena.tasks.stability.flow import compression, heating

if __name__ == "__main__":
    from dask.distributed import Client
    from dask_jobqueue import SLURMCluster
    from prefect_dask import DaskTaskRunner

    nodes_per_alloc = 1
    gpus_per_alloc = 4

    cluster_kwargs = dict(
        cores=1,
        memory="64 GB",
        processes=1,
        shebang="#!/bin/bash",
        account="matgen",
        walltime="04:00:00",
        job_mem="0",
        job_script_prologue=[
            "source ~/.bashrc",
            "module load python",
            "source activate /pscratch/sd/c/cyrusyc/.conda/mlip-arena",
        ],
        job_directives_skip=["-n", "--cpus-per-task", "-J"],
        job_extra_directives=[
            "-J arena-stability",
            "-q preempt",
            "--time-min=00:30:00",
            "--comment=12:00:00",
            f"-N {nodes_per_alloc}",
            "-C gpu",
            f"-G {gpus_per_alloc}",
        ],
    )

    cluster = SLURMCluster(**cluster_kwargs)
    print(cluster.job_script())
    cluster.adapt(minimum_jobs=10, maximum_jobs=50)
    client = Client(cluster)

    for model in MLIPEnum:
        run_dir = Path(__file__).parent / f"{REGISTRY[model.name]['family']}"

        heating.with_options(
            task_runner=DaskTaskRunner(address=client.scheduler.address),
            log_prints=True,
        )(model, run_dir)

    for model in MLIPEnum:
        run_dir = Path(__file__).parent / f"{REGISTRY[model.name]['family']}"

        compression.with_options(
            task_runner=DaskTaskRunner(address=client.scheduler.address),
            log_prints=True,
        )(model, run_dir)
