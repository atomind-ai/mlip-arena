import functools
from pathlib import Path

import pandas as pd
from ase.db import connect
from dask.distributed import Client
from dask_jobqueue import SLURMCluster
from prefect import Task, flow, task
from prefect.client.schemas.objects import TaskRun
from prefect.states import State
from prefect_dask import DaskTaskRunner

from mlip_arena.models import REGISTRY, MLIPEnum
from mlip_arena.tasks.eos import run as EOS
from mlip_arena.tasks.optimize import run as OPT
from mlip_arena.tasks.utils import get_calculator


@task
def load_wbm_structures():
    """
    Load the WBM structures from a JSON file.
    """
    with connect("wbm_structures.db") as db:
        for row in db.select():
            yield row.toatoms(add_additional_information=True)


def save_result(
    tsk: Task,
    run: TaskRun,
    state: State,
    model_name: str,
    id: str,
):
    result = run.state.result()

    assert isinstance(result, dict)

    result["method"] = model_name
    result["id"] = id
    result.pop("atoms", None)

    # fpath = Path(f"{model_name}.parquet")
    fpath = Path(f"{model_name}")
    fpath.mkdir(exist_ok=True)

    fpath = fpath / f"{result['id']}.pkl"

    df = pd.DataFrame([result])
    df.to_pickle(fpath)

    # if fpath.exists():
    #     df = pd.read_parquet(fpath)
    #     df = pd.concat([df, pd.DataFrame([result])], ignore_index=True)
    # else:
    #     df = pd.DataFrame([result])

    # df.drop_duplicates(subset=["id", "method"], keep="last", inplace=True)
    # df.to_parquet(fpath)


@task
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

    return EOS.with_options(
        refresh_cache=True,
        on_completion=[functools.partial(
            save_result,
            model_name=model.name,
            id=atoms.info["key_value_pairs"]["wbm_id"],
        )],
    )(
        atoms=result["atoms"],
        calculator=calculator,
        optimizer="FIRE",
        npoints=21,
        max_abs_strain=0.2,
        concurrent=False
    )


@flow
def run_all():
    futures = []
    for atoms in load_wbm_structures():
        for model in MLIPEnum:
            if "eos-bulk" not in REGISTRY[model.name].get("gpu-tasks", []):
                continue
            result = eos_bulk.submit(atoms, model)
            futures.append(result)
    return [f.result(raise_on_failure=False) for f in futures]


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
        "-J eos-bulk",
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
