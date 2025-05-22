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

@task(
    name="E-V Scan",
    task_run_name=lambda: f"{task_run.task_name}: {task_run.parameters['atoms'].get_chemical_formula()} - {task_run.parameters['model'].name}",
    cache_policy=TASK_SOURCE + INPUTS,
)
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

    fpath = Path(f"{model.name}") / f"{wbm_id}.json"
    fpath.parent.mkdir(exist_ok=True)

    df = pd.DataFrame([data])
    df.to_json(fpath)

    return df


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
            try:
                result = ev_scan.submit(atoms, model)
            except Exception as e:
                print(f"Failed to submit task for {model.name}: {e}")
                continue
            futures.append(result)
    return [f.result(raise_on_failure=False) for f in futures]

if __name__ == "__main__":
    nodes_per_alloc = 1
    gpus_per_alloc = 1
    ntasks = 1

    cluster_kwargs = dict(
        cores=1,
        memory="64 GB",
        processes=1,
        shebang="#!/bin/bash",
        account="m3828",
        walltime="00:30:00",
        # job_mem="0",
        job_script_prologue=[
            "source ~/.bashrc",
            "module load python",
            "source activate /pscratch/sd/c/cyrusyc/.conda/mlip-arena",
        ],
        job_directives_skip=["-n", "--cpus-per-task", "-J"],
        job_extra_directives=[
            "-J wbm_ev",
            "-q debug",
            f"-N {nodes_per_alloc}",
            "-C gpu",
            f"-G {gpus_per_alloc}",
            "--exclusive",
        ],
    )

    cluster = SLURMCluster(**cluster_kwargs)
    print(cluster.job_script())
    cluster.adapt(minimum_jobs=2, maximum_jobs=2)
    client = Client(cluster)

    submit_tasks.with_options(
        task_runner=DaskTaskRunner(address=client.scheduler.address),
        log_prints=True,
    )()
