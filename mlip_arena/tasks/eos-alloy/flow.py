from functools import partial
from pathlib import Path

import pandas as pd
from dask.distributed import Client
from dask_jobqueue import SLURMCluster
from huggingface_hub import hf_hub_download
from prefect import Task, flow, task
from prefect.client.schemas.objects import TaskRun
from prefect.futures import wait
from prefect.states import State
from prefect_dask import DaskTaskRunner

from ase import Atoms
from ase.db import connect
from ase.formula import Formula
from mlip_arena.models import REGISTRY, MLIPEnum
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


def save_to_hdf(
    tsk: Task, run: TaskRun, state: State, fpath: Path | str, table_name: str
):
    """
    Define a hook on completion of EOS task to save results to HDF5 file.
    """
    if state.is_completed():
        result = state.result()

        atoms = run.parameters["atoms"]
        calculator_name = run.parameters["calculator_name"]

        formula = Formula(atoms.get_chemical_symbols())

        compositions = {e: c / len(atoms) for e, c in formula.count()}

        df = pd.DataFrame(
            {
                "method": calculator_name,
                "formula": str(formula),
                "total_run_time": run.total_run_time,
                "v0": result["v0"],
                "e0": result["e0"],
                "b0": result["b0"],
                "b1": result["b1"],
                "volume": result["eos"]["volumes"],
                "energy": result["eos"]["energies"],
                **compositions,
            }
        )

        with pd.HDFStore(fpath, mode="a") as store:
            store.append(table_name, df, format="table", data_columns=True)


@flow
def run_from_db(db_path: Path | str, out_path: Path | str, table_name: str):
    EOS_ = EOS.with_options(
        on_completion=[partial(save_to_hdf, fpath=out_path, table_name=table_name)]
    )

    futures = []
    for atoms in get_atoms_from_db(db_path):
        for mlip in MLIPEnum:
            if not REGISTRY[mlip.name]["npt"]:
                continue
            future = EOS_.submit(
                atoms=atoms,
                calculator_name=mlip.name,
                calculator_kwargs={},
                optimizer="FIRE",
                criterion=dict(fmax=0.1, steps=1000),
                concurrent=False,
            )
            futures.append(future)

    wait(futures)

    return [
        f.result(timeout=None, raise_on_failure=False)
        for f in futures
        if f.state.is_completed()
    ]


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

    results = run_from_db_(
        db_path="sqs_Fe-Ni-Cr.db", out_path="eos.h5", table_name="Fe-Ni-Cr"
    )
