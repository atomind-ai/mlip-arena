from pathlib import Path

from dask.distributed import Client
from dask_jobqueue import SLURMCluster
from huggingface_hub import hf_hub_download
from prefect import flow, task
from prefect_dask import DaskTaskRunner

from ase.db import connect
from mlip_arena.models import MLIPEnum
from mlip_arena.tasks.eos import run as EOS


@task
def get_atoms_from_db(db_path: Path | str):
    db_path = Path(db_path)
    if not db_path.exists():
        db_path = hf_hub_download(
            repo_id="atomind/mlip-arena",
            repo_type="dataset",
            subfolder=f"{Path().resolve().name}",
            filename=str(db_path),
        )
    with connect(db_path) as db:
        for row in db.select():
            yield row.toatoms()


@flow
def run_from_db(db_path: Path | str):
    futures = []
    for atoms in get_atoms_from_db(db_path):
        for mlip in MLIPEnum:
            future = EOS.submit(
                atoms=atoms,
                calculator_name=mlip.name,
                calculator_kwargs={},
                optimizer="FIRE",
                criterion=dict(fmax=0.1, steps=1000),
                concurrent=False,
            )
            futures.append(future)
    return [f.result() for f in futures]


if __name__ == "__main__":
    nodes_per_alloc = 1
    gpus_per_alloc = 4
    ntasks = 1

    cluster_kwargs = dict(
        cores=1,
        memory="64 GB",
        shebang="#!/bin/bash",
        account="m3828",
        walltime="00:30:00",
        job_mem="0",
        job_script_prologue=[
            "source ~/.bashrc",
            "module load python",
            "source activate /pscratch/sd/c/cyrusyc/.conda/mlip-arena",
        ],
        job_directives_skip=["-n", "--cpus-per-task", "-J"],
        job_extra_directives=[
            "-J eos",
            "-q debug",
            f"-N {nodes_per_alloc}",
            "-C cpu",
            # f"-G {gpus_per_alloc}",
        ],
    )

    cluster = SLURMCluster(**cluster_kwargs)
    print(cluster.job_script())
    cluster.adapt(minimum_jobs=2, maximum_jobs=2)
    client = Client(cluster)

    run_from_db_ = run_from_db.with_options(
        task_runner=DaskTaskRunner(address=client.scheduler.address), log_prints=True
    )

    results = run_from_db_(db_path="sqs_Fe-Ni-Cr.db")
