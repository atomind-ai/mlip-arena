from __future__ import annotations

import os
from pathlib import Path

from ase import Atoms, units
from ase.calculators.calculator import BaseCalculator
from dotenv import load_dotenv
from prefect import flow, task
from prefect.cache_policies import INPUTS, TASK_SOURCE
from prefect.futures import wait

from mlip_arena.models import REGISTRY, MLIPEnum
from mlip_arena.tasks.md import run as MD
from mlip_arena.tasks.stability.input import get_atoms_from_db
from mlip_arena.tasks.utils import get_calculator

load_dotenv()

HF_TOKEN = os.environ.get("HF_TOKEN", None)
MP_API_KEY = os.environ.get("MP_API_KEY", None)


@task(cache_policy=TASK_SOURCE + INPUTS)
def nvt_heat_one(
    atoms: Atoms,
    model: MLIPEnum | BaseCalculator,
):
    """Run a 10 ps NVT MD simulation with linear heating schedule.

    This task invokes the MD task (wrapped via Prefect) to perform an NVT
    simulation using the provided calculator or MLIP model. It is intended
    to probe whether the model remains stable when the system is heated
    from 300 K to 3000 K over a short timeframe.

    Parameters
    - atoms: ASE Atoms object representing the system to simulate. A copy
      is typically submitted by the caller.
    - model: either an MLIPEnum entry (selects a registered model) or an
      already-constructed ASE BaseCalculator.

    Returns
    - The result produced by the MD task. On exception, the exception object
      is returned (the calling flow records and filters results).
    """

    calculator = (
        get_calculator(
            model.name,
            calculator_kwargs=None,
        )
        if isinstance(model, MLIPEnum)
        else model
    )

    model_name = model.name if isinstance(model, MLIPEnum) else model.__class__.__name__
    traj_dir = (
        Path(__file__).parent / f"{REGISTRY[model_name]['family']}"
        if isinstance(model, MLIPEnum)
        else Path("custom")
    )

    return MD.with_options(
        # timeout_seconds=600,
        # retries=1,
        refresh_cache=True
    )(
        atoms=atoms,
        # wrap get_calculator in task to dynamically assign GPU device
        calculator=calculator,
        ensemble="nvt",
        dynamics="nose-hoover",
        time_step=None,
        dynamics_kwargs=dict(
            ttime=25 * units.fs,
            # pfactor=((75 * units.fs) ** 2) * 1e2 * units.GPa
        ),
        total_time=1e4,  # 10 ps
        temperature=[300, 3000],
        pressure=None,
        traj_file=traj_dir
        / f"{model_name}_{atoms.info.get('material_id', 'random')}_{atoms.get_chemical_formula()}_nvt.traj",
        traj_interval=10,
    )


@task(cache_policy=TASK_SOURCE + INPUTS)
def npt_compress_one(
    atoms: Atoms,
    model: MLIPEnum | BaseCalculator,
):
    """Run a 10 ps NPT MD simulation with linear pressure ramp.

    This task invokes the MD task (wrapped via Prefect) to perform an NPT
    simulation where the pressure ramps up to probe structural response and
    potential instabilities under compression.

    Parameters
    - atoms: ASE Atoms object representing the system to simulate.
    - model: either an MLIPEnum entry (selects a registered model) or an
      already-constructed ASE BaseCalculator.

    Returns
    - The result produced by the MD task.
    """

    calculator = (
        get_calculator(
            model.name,
            calculator_kwargs=None,
        )
        if isinstance(model, MLIPEnum)
        else model
    )

    model_name = model.name if isinstance(model, MLIPEnum) else model.__class__.__name__
    traj_dir = (
        Path(__file__).parent / f"{REGISTRY[model_name]['family']}"
        if isinstance(model, MLIPEnum)
        else Path("custom")
    )

    return MD.with_options(timeout_seconds=600, retries=2, refresh_cache=True)(
        atoms=atoms,
        calculator=calculator,
        ensemble="npt",
        dynamics="nose-hoover",
        time_step=None,
        dynamics_kwargs=dict(
            ttime=25 * units.fs, pfactor=((75 * units.fs) ** 2) * 1e2 * units.GPa
        ),
        total_time=1e4,  # 5e4, # fs
        temperature=[300, 3000],
        pressure=[0, 5e2 * units.GPa],  # 500 GPa / 10 ps = 50 GPa / 1 ps
        traj_file=traj_dir
        / f"{model_name}_{atoms.info.get('material_id', 'random')}_{atoms.get_chemical_formula()}_npt.traj",
        traj_interval=10,
    )


@flow
def heating(model: MLIPEnum | BaseCalculator):
    """Prefect flow to run NVT heating tasks for many database structures.

    This flow iterates over structures from the 'random-mixture.db' dataset
    and submits nvt_heat_one tasks for each structure. It waits for all
    submitted futures and returns the list of completed results.

    Parameters
    - model: MLIPEnum or BaseCalculator to use for the simulations.

    Returns
    - A list of results from completed tasks. Failed tasks are filtered out.
    """

    futures = []
    # To download the database automatically, `huggingface_hub login` or provide HF_TOKEN
    for atoms in get_atoms_from_db("random-mixture.db", force_download=False):
        future = nvt_heat_one.with_options(
            timeout_seconds=600, retries=2, refresh_cache=False
        ).submit(atoms.copy(), model)
        futures.append(future)

    wait(futures)

    return [
        f.result(timeout=None, raise_on_failure=False)
        for f in futures
        if f.state.is_completed()
    ]


@flow
def compression(model: MLIPEnum | BaseCalculator):
    """Prefect flow to run NPT compression tasks for many database structures.

    This flow iterates over structures from the 'random-mixture.db' dataset
    and submits npt_compress_one tasks for each structure. It waits for
    completion and returns the list of successful results.

    Parameters
    - model: MLIPEnum or BaseCalculator to use for the simulations.

    Returns
    - A list of results from completed tasks. Failed tasks are filtered out.
    """

    futures = []
    # To download the database automatically, `huggingface_hub login` or provide HF_TOKEN
    for atoms in get_atoms_from_db("random-mixture.db", force_download=False):
        future = npt_compress_one.with_options(
            timeout_seconds=600, retries=2, refresh_cache=False
        ).submit(atoms.copy(), model)
        futures.append(future)

    wait(futures)

    return [
        f.result(timeout=None, raise_on_failure=False)
        for f in futures
        if f.state.is_completed()
    ]


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

    heating.with_options(
        task_runner=DaskTaskRunner(address=client.scheduler.address), 
        log_prints=True
    )()

    compression.with_options(
        task_runner=DaskTaskRunner(address=client.scheduler.address), 
        log_prints=True
    )()
