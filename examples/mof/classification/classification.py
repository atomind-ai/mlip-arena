import functools
import itertools
from pathlib import Path

import pandas as pd
from ase import Atoms
from ase.build import molecule
from dask.distributed import Client
from dask_jobqueue import SLURMCluster
from prefect import Task, flow, task
from prefect.client.schemas.objects import TaskRun
from prefect.states import State
from prefect_dask import DaskTaskRunner
from tqdm.auto import tqdm

from mlip_arena.models import MLIPEnum
from mlip_arena.tasks.mof.flow import widom_insertion
from mlip_arena.tasks.utils import get_calculator


def load_row_from_df(fpath: str):
    df = pd.read_pickle(fpath)

    for _, row in df.iterrows():
        yield row


def save_result(
    tsk: Task,
    run: TaskRun,
    state: State,
    row: pd.DataFrame,
    model_name: str,
    gas: Atoms,
    fpath: str,
):
    result = run.state.result()

    assert isinstance(result, dict)

    copied = row.copy()
    copied["model"] = model_name
    copied["gas"] = gas

    for k, v in result.items():
        copied[k] = v

    fpath = Path(f"{model_name}.pkl")

    if fpath.exists():
        df = pd.read_pickle(fpath)
        df = pd.concat([df, pd.DataFrame([copied])], ignore_index=True)
    else:
        df = pd.DataFrame([copied])

    df.drop_duplicates(subset=["name", "model"], keep="last", inplace=True)
    df.to_pickle(fpath)


# Orchestrate your awesome dask workflow runner

nodes_per_alloc = 1
gpus_per_alloc = 1
ntasks = 1

cluster_kwargs = dict(
    cores=4,
    memory="64 GB",
    shebang="#!/bin/bash",
    account="m3828",
    walltime="04:00:00",
    job_mem="0",
    job_script_prologue=[
        "source ~/.bashrc",
        "module load python",
        "source activate /pscratch/sd/c/cyrusyc/.conda/mlip-arena",
    ],
    job_directives_skip=["-n", "--cpus-per-task", "-J"],
    job_extra_directives=[
        "-J mof",
        "-q regular",
        f"-N {nodes_per_alloc}",
        "-C gpu",
        f"-G {gpus_per_alloc}",
        # "--exclusive",
    ],
)

cluster = SLURMCluster(**cluster_kwargs)
print(cluster.job_script())
cluster.adapt(minimum_jobs=10, maximum_jobs=20)
client = Client(cluster)


@task
def run_one(model, row, gas):
    return widom_insertion.with_options(
        refresh_cache=False,
        on_completion=[functools.partial(
            save_result,
            row=row,
            model_name=model.name,
            gas=gas,
            fpath=f"{model.name}.pkl"
        )]
    )(
        structure=row["structure"],
        gas=gas,
        calculator=get_calculator(
            model,
            dispersion=True
        ),
        criterion=dict(fmax=0.05, steps=50),
        init_structure_optimize_loops = 10,
    )


@flow
def run_all():
    futures = []
    gas = molecule("CO2")

    for model, row in tqdm(itertools.product(MLIPEnum, load_row_from_df("input.pkl"))):

        if model.name not in ["MACE-MPA", "MatterSim", "SevenNet", "M3GNet", "ORBv2"]:
            continue

        fpath = Path(f"{model.name}.pkl")

        if fpath.exists():
            df = pd.read_pickle(fpath)
            if row['name'] in df['name'].values:
                continue
            
        try:
            print(model, row['name'])
            future = run_one.submit(
                model,
                row,
                gas,
            )
            futures.append(future)
        except Exception:
            continue

    return [f.result(raise_on_failure=False) for f in futures]

# run_all()
run_all.with_options(
    task_runner=DaskTaskRunner(address=client.scheduler.address),
    log_prints=True,
)()
