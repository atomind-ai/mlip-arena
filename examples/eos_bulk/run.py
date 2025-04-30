from pathlib import Path

import numpy as np
import pandas as pd
from ase.db import connect
from dask.distributed import Client
from dask_jobqueue import SLURMCluster
from prefect import flow, task
from prefect.runtime import task_run
from prefect_dask import DaskTaskRunner
from prefect.cache_policies import INPUTS, TASK_SOURCE

from mlip_arena.models import REGISTRY, MLIPEnum
from mlip_arena.tasks.utils import get_calculator


@task
def load_wbm_structures():
    """
    Load the WBM structures from a ASE DB file.
    """
    with connect("../wbm_structures.db") as db:
        for row in db.select():
            yield row.toatoms(add_additional_information=True)


# def save_result(
#     tsk: Task,
#     run: TaskRun,
#     state: State,
#     model_name: str,
#     id: str,
# ):
#     result = run.state.result()

#     assert isinstance(result, dict)

#     result["method"] = model_name
#     result["id"] = id
#     result.pop("atoms", None)

#     fpath = Path(f"{model_name}")
#     fpath.mkdir(exist_ok=True)

#     fpath = fpath / f"{result['id']}.pkl"

#     df = pd.DataFrame([result])
#     df.to_pickle(fpath)


@task(
    name="EOS bulk - WBM",
    task_run_name=lambda: f"{task_run.task_name}: {task_run.parameters['atoms'].get_chemical_formula()} - {task_run.parameters['model'].name}",
    cache_policy=TASK_SOURCE + INPUTS,
)
def eos_bulk(atoms, model):

    calculator = get_calculator(
        model
    )  # avoid sending entire model over prefect and select freer GPU

    result = OPT.with_options(
        refresh_cache=True,
    )(
        atoms,
        calculator,
        optimizer="FIRE",
        criterion=dict(
            fmax=0.1,
        ),
    )

    result = EOS.with_options(
        refresh_cache=True,
        # on_completion=[functools.partial(
        #     save_result,
        #     model_name=model.name,
        #     id=atoms.info["key_value_pairs"]["wbm_id"],
        # )],
    )(
        atoms=result["atoms"],
        calculator=calculator,
        optimizer="FIRE",
        npoints=21,
        max_abs_strain=0.2,
        concurrent=False
    )

    result["method"] = model.name
    result["id"] = atoms.info["key_value_pairs"]["wbm_id"]
    result.pop("atoms", None)

    fpath = Path(f"{model.name}")
    fpath.mkdir(exist_ok=True)

    fpath = fpath / f"{result['id']}.pkl"

    df = pd.DataFrame([result])
    df.to_pickle(fpath)


@flow
def run_all():
    futures = []
    for atoms in load_wbm_structures():
        for model in MLIPEnum:
            if "eos_bulk" not in REGISTRY[model.name].get("gpu-tasks", []):
                continue
            result = eos_bulk.submit(atoms, model)
            futures.append(result)
    return [f.result(raise_on_failure=False) for f in futures]


if __name__ == "__main__":
    nodes_per_alloc = 1
    gpus_per_alloc = 1
    ntasks = 1
    
    cluster_kwargs = dict(
        cores=4,
        memory="64 GB",
        shebang="#!/bin/bash",
        account="m3828",
        walltime="00:50:00",
        job_mem="0",
        job_script_prologue=[
            "source ~/.bashrc",
            "module load python",
            "source activate /pscratch/sd/c/cyrusyc/.conda/mlip-arena",
        ],
        job_directives_skip=["-n", "--cpus-per-task", "-J"],
        job_extra_directives=[
            "-J eos_bulk",
            "-q regular",
            f"-N {nodes_per_alloc}",
            "-C gpu",
            f"-G {gpus_per_alloc}",
            "--exclusive",
        ],
    )
    
    cluster = SLURMCluster(**cluster_kwargs)
    print(cluster.job_script())
    cluster.adapt(minimum_jobs=20, maximum_jobs=40)
    client = Client(cluster)
    
    run_all.with_options(
        task_runner=DaskTaskRunner(address=client.scheduler.address),
        log_prints=True,
    )()
