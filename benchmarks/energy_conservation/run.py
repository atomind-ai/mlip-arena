"""
Task for running MD simulations and computing the differential entropy
of the simulated structures with respect to a reference dataset.

See https://github.com/dskoda/quests for differential entropy details.
"""

from __future__ import annotations

import os
from datetime import datetime

import numpy as np
from ase.io import read
from prefect import task
from prefect.cache_policies import INPUTS, TASK_SOURCE
from prefect.runtime import task_run

from mlip_arena.models import MLIPEnum
from mlip_arena.tasks.md import run as MD
from mlip_arena.tasks.utils import logger

try:
    from quests.descriptor import get_descriptors
    from quests.entropy import delta_entropy
except ImportError as e:
    logger.warning(e)
    logger.warning(
        "quests is not installed. Please install it using `pip install quests` or following the instructions at https://github.com/dskoda/quests to use this module."
    )


def get_entropy_from_path(
    subset_path, dataset_path, dataset_desc_out_path, k=32, cutoff=5.0, h=0.015
):
    """
    Computes the differential entropy of a subset of structures with respect
    to a reference dataset.

    Arguments:
        subset_path (str): Path to the file containing the subset of structures.
        dataset_path (str): Path to the file containing the full dataset of structures without the subset.
        dataset_desc_out_path (str): Path to save the descriptors of the full dataset.
        k (int, optional): Number of nearest neighbors used for descriptor calculation. Default is 32.
        cutoff (float, optional): Cutoff distance for descriptor calculation. Default is 5.0.
        h (float, optional): Bandwidth for the Gaussian kernel. Default is 0.015.

    Returns:
        np.ndarray: The differential entropy of the subset with respect to the dataset.
    """

    x_structures = read(dataset_path, index=":")
    x_desc = get_descriptors(x_structures, k=k, cutoff=cutoff)
    np.save(dataset_desc_out_path, x_desc)

    y_structures = read(subset_path, index=":")
    y_desc = get_descriptors(y_structures, k=k, cutoff=cutoff)

    dH = delta_entropy(y_desc, x_desc, h=h)
    return dH


def get_trajectory_entropy(
    trajectory_dir,
    start_idx,
    end_idx,
    step,
    dataset_desc_path,
    k=32,
    cutoff=5.0,
    h=0.015,
):
    """
    Computes the differential entropy of a subset of structures in a trajectory with respect
    to a reference dataset.

    Arguments:
        trajectory_dir (str): Path to the directory containing the trajectory files.
        start_idx (int): Starting index of the subset of structures to select from each trajectory.
        end_idx (int): Ending index of the subset of structures to select from each trajectory.
        step (int): Step size of the subset of structures to select from each trajectory.
        dataset_desc_path (str): Path to the file containing the descriptors of the full dataset of structures without the subset.
        k (int, optional): Number of nearest neighbors used for descriptor calculation. Default is 32.
        cutoff (float, optional): Cutoff distance for descriptor calculation. Default is 5.0.
        h (float, optional): Bandwidth for the Gaussian kernel. Default is 0.015.

    Choose start_idx, end_idx, step to select which structures to compute the differential entropy for, based on what sliding window is chosen.
    e.g. window of size 5 with stride 2 means we select every other structure starting at index 2 (middle of the first window) to the -2 index (middle of the last window)

    Returns:
        np.ndarray: The differential entropy of the subset of structures in the trajectory with respect to the dataset.
    """
    structures = []
    for traj_file in sorted(os.listdir(trajectory_dir)):
        traj = read(os.path.join(trajectory_dir, traj_file), index=":")
        every_other = traj[start_idx:end_idx:step]
        structures.extend(every_other)

    desc = get_descriptors(structures, k=k, cutoff=cutoff)
    x_desc = np.load(dataset_desc_path)
    dH = delta_entropy(desc, x_desc, h=h)
    return dH


def run_simulations(model_names, structures, out_dir):
    """
    Runs simulations on a list of structures.

    Parameters:
        model_names (list[str]): List of models to use.
        structures (list[ase.Atoms]): List of structures to simulate.
        out_dir (str): Directory to save the simulation trajectories to.

    Notes:
        Structures are replicated to have at least 100 atoms and at most 500 atoms.
        Structures are simulated with NVE MD at 1000 K for 5 ps.
        Simulation trajectories are saved to files in out_dir, with each file named according to the index of the structure in the list.
    """
    min_atoms = 100
    max_atoms = 500

    futures = []

    for model_name in model_names:
        os.makedirs(out_dir, exist_ok=True)
        model = MLIPEnum[model_name]
        calc = model.value()

        for i, atoms in enumerate(structures):
            logger.info(
                f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Running {model_name} on structure number {i}"
            )

            # Replicate the structure
            n_atoms = len(atoms)
            rep_factor = int(
                np.ceil((min_atoms / n_atoms) ** (1 / 3))
            )  # cube root since it's a 3D replication
            supercell_atoms = atoms.repeat((rep_factor, rep_factor, rep_factor))
            if len(supercell_atoms) > max_atoms:
                logger.info(
                    f"Skipping structure {i} because it has too many atoms ({len(supercell_atoms)} > {max_atoms})"
                )
                continue  # skip if it becomes too large

            # Run NVE MD @ 1000K for 5 ps
            future = MD.submit(
                supercell_atoms,
                calculator=calc,
                ensemble="nve",
                dynamics="velocityverlet",
                time_step=1.0,  # fs
                total_time=5000,  # 5 ps = 5000 fs
                temperature=1000.0,
                traj_file=f"{out_dir}/{i}.traj",
                traj_interval=100,
                zero_linear_momentum=True,
                zero_angular_momentum=True,
            )
            futures.append(future)

    return [f.result(raise_on_failure=False) for f in futures]


def _generate_task_run_name():
    task_name = task_run.task_name
    parameters = task_run.parameters

    trajectory_dir = parameters["trajectory_dir"]
    dataset_desc_path = parameters["dataset_desc_path"]

    return f"{task_name}: {trajectory_dir} - {dataset_desc_path}"


@task(
    name="Entropy along trajectory",
    task_run_name=_generate_task_run_name,
    cache_policy=TASK_SOURCE + INPUTS,
)
def run(
    dataset_path,
    model_names,
    structures,
    trajectory_dir,
    start_idx,
    end_idx,
    step,
    dataset_desc_path,
    dH_out_path,
    k=32,
    cutoff=5.0,
    h=0.015,
):
    # Get descriptors for the dataset. This should exclude the subset of structures used for simulations.
    # This may take a while if the dataset is large - in that case, would recommend splitting the structures into separate chunks.
    x_structures = read(dataset_path, index=":")
    x_desc = get_descriptors(x_structures, k=k, cutoff=cutoff)
    np.save(dataset_desc_path, x_desc)

    # Run simulations
    run_simulations(model_names, structures, trajectory_dir)

    # Get entropy for structures along trajectories
    dH = get_trajectory_entropy(
        trajectory_dir,
        start_idx,
        end_idx,
        step,
        dataset_desc_path,
        k=k,
        cutoff=cutoff,
        h=h,
    )
    np.save(dH_out_path, dH)
