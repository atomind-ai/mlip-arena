import json
from pathlib import Path

import numpy as np
import pandas as pd
from ase.db import connect
from dask.distributed import Client
from dask_jobqueue import SLURMCluster
from prefect import Task, flow, task
from prefect.client.schemas.objects import TaskRun
from prefect.states import State
from prefect_dask import DaskTaskRunner

from mlip_arena.models import REGISTRY, MLIPEnum
from mlip_arena.tasks.utils import get_calculator


@task
def load_wbm_structures():
    """
    Load the WBM structures from an ASE database file.
    
    Reads structures from 'wbm_structures.db' and yields them as ASE Atoms objects
    with additional metadata preserved from the database.
    
    Yields:
        ase.Atoms: Individual atomic structures from the WBM database with preserved
                  metadata in the .info dictionary.
    """
    with connect("../wbm_structures.db") as db:
        for row in db.select():
            yield row.toatoms(add_additional_information=True)


def save_result(
    tsk: Task,
    run: TaskRun,
    state: State,
    model_name: str,
    id: str,
):
    """
    Save calculation results to a pickle file in a model-specific directory.
    
    This function handles the persistence of calculation results from tasks,
    organizing outputs by model name and structure ID.
    
    Args:
        tsk: The Prefect Task object that produced the result.
        run: The TaskRun object containing execution information.
        state: The State object representing the task's execution state.
        model_name: Name of the model used for the calculation.
        id: Structure identifier for the calculation.
    
    Note:
        Creates a directory named after the model if it doesn't exist.
        Saves results as a pandas DataFrame in pickle format.
    """
    result = run.state.result()

    assert isinstance(result, dict)

    result["method"] = model_name
    result["id"] = id
    result.pop("atoms", None)

    fpath = Path(f"{model_name}")
    fpath.mkdir(exist_ok=True)

    fpath = fpath / f"{result['id']}.pkl"

    df = pd.DataFrame([result])
    df.to_pickle(fpath)


@task
def ev_scan(atoms, model):
    """
    Perform an energy-volume scan for a given model and atomic structure.
    
    This function applies uniaxial strain to the structure in all three dimensions,
    maintaining the fractional coordinates of atoms, and computes the energy at each
    deformation point using the specified model.
    
    Args:
        atoms: ASE atoms object containing the structure to analyze.
        model: MLIPEnum model to use for the energy calculations.
        
    Returns:
        dict: Results dictionary containing:
            - method (str): The name of the model used
            - id (str): The WBM ID of the structure
            - eos (dict): Energy of state data with:
                - volumes (list): Volume of the unit cell at each strain point
                - energies (list): Computed potential energy at each strain point
                
    Note:
        The strain range is fixed at ±20% with 21 evenly spaced points.
        Results are also saved as a JSON file in a directory named after the model.
    """
    calculator = get_calculator(
        model
    )  # avoid sending entire model over prefect and select freer GPU

    wbm_id = atoms.info["key_value_pairs"]["wbm_id"]

    c0 = atoms.get_cell()
    max_abs_strain = 0.2
    npoints = 21
    volumes = []
    energies = []
    for uniaxial_strain in np.linspace(-max_abs_strain, max_abs_strain, npoints):
        cloned = atoms.copy()
        scale_factor = uniaxial_strain + 1
        cloned.set_cell(c0 * scale_factor, scale_atoms=True)
        cloned.calc = calculator
        volumes.append(cloned.get_volume())
        energies.append(cloned.get_potential_energy())

    data = {
        "method": model.name,
        "id": wbm_id,
        "eos": {
            "volumes": volumes, "energies": energies
        }
    }

    with open(f"{model.name}/{wbm_id}.json", "w", encoding="utf-8") as f:
        json.dump(data, f)

    return data


@flow
def submit_tasks():
    """
    Create and submit energy-volume scan tasks for subsampled WBM structures and applicable models.
    
    This flow function:
    1. Loads all structures from the WBM database
    2. Iterates through available models in MLIPEnum
    3. Filters models based on their capability to handle the 'wbm_ev' GPU task
    4. Submits parallel ev_scan tasks for all valid (structure, model) combinations
    5. Collects and returns results from all tasks
    
    Returns:
        list: Results from all executed tasks (successful or failed)
    """
    futures = []
    for atoms in load_wbm_structures():
        for model in MLIPEnum:
            if "wbm_ev" not in REGISTRY[model.name].get("gpu-tasks", []):
                continue
            result = ev_scan.submit(atoms, model)
            futures.append(result)
    return [f.result(raise_on_failure=False) for f in futures]


nodes_per_alloc = 1
gpus_per_alloc = 1
ntasks = 1

cluster_kwargs = dict(
    cores=1,
    memory="64 GB",
    processes=1,
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
        "-J wbm_ev",
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

submit_tasks.with_options(
    task_runner=DaskTaskRunner(address=client.scheduler.address),
    log_prints=True,
)()
