"""
Task for running MD simulations and computing the differential entropy
of the simulated structures with respect to a reference dataset.

See https://github.com/dskoda/quests for differential entropy details.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
from ase import Atoms
from ase.io import read, write
from prefect import flow, task
from prefect.cache_policies import INPUTS, TASK_SOURCE
from prefect.runtime import flow_run

from mlip_arena.models import MLIPEnum
from mlip_arena.tasks.md import run as MD
from mlip_arena.tasks.utils import BaseCalculator, get_calculator, logger

try:
    from quests.descriptor import get_descriptors
    from quests.entropy import delta_entropy
except ImportError as e:
    logger.warning(e)
    logger.warning(
        "quests is not installed. Please install it using `pip install quests` or following the instructions at https://github.com/dskoda/quests to use this module."
    )


@task(
    cache_policy=TASK_SOURCE + INPUTS,
)
def get_descriptors_from_path(
    input_path: Path, k: int = 32, cutoff: float = 5.0, output_path: Path | None = None
) -> np.ndarray:
    output_path = (
        input_path.with_stem(f"{input_path.stem}_desc_{k}_{cutoff}").with_suffix(".npy")
        if output_path is None
        else output_path
    )

    if output_path.exists():
        ref_descriptors = np.load(output_path)
    else:
        ref_structures = read(input_path, index=":")
        ref_descriptors = get_descriptors(ref_structures, k=k, cutoff=cutoff)
        np.save(output_path, ref_descriptors)

    return ref_descriptors


@task(
    cache_policy=TASK_SOURCE + INPUTS,
)
def get_entropy_from_path(
    subset_path: Path,
    reference_path: Path,
    k: int = 32,
    cutoff: float = 5.0,
    h: float = 0.015,
):
    """
    Computes the differential entropy of a subset of structures with respect
    to a reference dataset.

    Arguments:
        subset_path (Path): Path to the file containing the subset of structures.
        reference_path (Path): Path to the file containing the full dataset of structures without the subset.
        k (int, optional): Number of nearest neighbors used for descriptor calculation. Default is 32.
        cutoff (float, optional): Cutoff distance for descriptor calculation. Default is 5.0.
        h (float, optional): Bandwidth for the Gaussian kernel. Default is 0.015.

    Returns:
        np.ndarray: The differential entropy of the subset with respect to the dataset.
    """

    ref_descriptors = get_descriptors_from_path(reference_path, k=k, cutoff=cutoff)
    y_desc = get_descriptors_from_path(subset_path, k=k, cutoff=cutoff)
    return delta_entropy(y_desc, ref_descriptors, h=h)


def get_trajectory_entropy(
    trajectory_dir,
    start_idx,
    end_idx,
    step,
    reference_path,
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
        reference_path (str): Path to the file containing the descriptors of the full dataset of structures without the subset.
        k (int, optional): Number of nearest neighbors used for descriptor calculation. Default is 32.
        cutoff (float, optional): Cutoff distance for descriptor calculation. Default is 5.0.
        h (float, optional): Bandwidth for the Gaussian kernel. Default is 0.015.

    Choose start_idx, end_idx, step to select which structures to compute the differential entropy for, based on what sliding window is chosen.
    e.g. window of size 5 with stride 2 means we select every other structure starting at index 2 (middle of the first window) to the -2 index (middle of the last window)

    Returns:
        dH (np.ndarray): an (M,) vector of differential entropy dH ( Y | X )
            or (np.ndarray): an (H,M) matrix if 'h' is a vector of length H
        structures (list[ase.Atoms]): List of structures selected from the trajectory.
    """
    structures = []
    for traj_file in sorted(os.listdir(trajectory_dir)):
        try:
            traj = read(os.path.join(trajectory_dir, traj_file), index=":")
        except Exception:
            continue

        every_other = traj[start_idx:end_idx:step]
        structures.extend(every_other)

    descriptors = get_descriptors(structures, k=k, cutoff=cutoff)

    ref_descriptors = get_descriptors_from_path(reference_path, k=k, cutoff=cutoff)
    return delta_entropy(descriptors, ref_descriptors, h=h), structures


def run_simulations(model: MLIPEnum | BaseCalculator | str, structures: list[Atoms], out_dir: Path):
    """
    Runs simulations on a list of structures.

    Parameters:
        model (MLIPEnum | BaseCalculator | str): Model to use.
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

    out_dir.mkdir(parents=True, exist_ok=True)

    for i, atoms in enumerate(structures):
        # Replicate the structure
        n_atoms = len(atoms)
        rep_factor = int(np.ceil((min_atoms / n_atoms) ** (1 / 3)))  # cube root since it's a 3D replication
        supercell_atoms = atoms.repeat((rep_factor, rep_factor, rep_factor))
        if len(supercell_atoms) > max_atoms:
            logger.info(f"Skipping structure {i} because it has too many atoms ({len(supercell_atoms)} > {max_atoms})")
            continue  # skip if it becomes too large

        # Run NVE MD @ 1000K for 5 ps
        future = MD.submit(
            supercell_atoms,
            calculator=get_calculator(model),
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


def _generate_flow_run_name():
    name = flow_run.flow_name
    parameters = flow_run.parameters

    model = parameters["model"]
    reference_path = parameters["reference_path"]

    return f"{name}: {model} - {reference_path}"


@flow(
    name="Entropy along trajectory",
    flow_run_name=_generate_flow_run_name,
)
def differential_entropy_along_nve_trajectory(
    model: MLIPEnum | str,
    structures: list[Atoms],
    reference_path: Path,
    start_idx: int,
    end_idx: int,
    step: int,
    k: int = 32,
    cutoff: float = 5.0,
    h: float = 0.015,
    work_dir: Path | None = None,
):
    """
    Computes the differential entropy of a subset of structures in a trajectory with respect
    to a reference dataset.

    Arguments:
        model (MLIPEnum | BaseCalculator | str): Model to use.
        structures (list[ase.Atoms]): List of structures to simulate.
        reference_path (Path): Path to the file containing the descriptors of the full dataset of structures without the subset.
        start_idx (int): Starting index of the subset of structures to select from each trajectory.
        end_idx (int): Ending index of the subset of structures to select from each trajectory.
        step (int): Step size of the subset of structures to select from each trajectory.
        k (int, optional): Number of nearest neighbors used for descriptor calculation. Default is 32.
        cutoff (float, optional): Cutoff distance for descriptor calculation. Default is 5.0.
        h (float, optional): Bandwidth for the Gaussian kernel. Default is 0.015.
        work_dir (Path | None, optional): Directory to save the simulation trajectories to. Default is None.

    Returns:
        dH (np.ndarray): an (M,) vector of differential entropy dH ( Y | X )
            or (np.ndarray): an (H,M) matrix if 'h' is a vector of length H
        sampled_structures (list[ase.Atoms]): List of structures selected from the trajectory.
    """

    if isinstance(model, MLIPEnum):
        model_name = model.name
    elif isinstance(model, BaseCalculator):
        model_name = model.__class__.__name__
    elif isinstance(model, str) and hasattr(MLIPEnum, model):
        model_name = model
    else:
        raise ValueError(f"Unsupported model: {model}")

    # Run simulations
    out_dir = work_dir / model_name if work_dir is not None else Path.cwd() / model_name
    run_simulations(model, structures, out_dir)

    # Get entropy for structures along trajectories
    dH, sampled_structures = get_trajectory_entropy(
        out_dir,
        start_idx,
        end_idx,
        step,
        reference_path,
        k=k,
        cutoff=cutoff,
        h=h,
    )
    np.save(out_dir / "dH.npy", dH)
    write(out_dir / "dH.traj", sampled_structures)

    return dH, sampled_structures
