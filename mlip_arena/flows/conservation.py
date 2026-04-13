"""Task for running MD simulations and computing the differential entropy of
the simulated structures with respect to a reference dataset.

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
from prefect.futures import wait
from prefect.runtime import flow_run, task_run

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
    """Get QUESTS descriptors from a file path. Caches results as .npy file.

    Args:
        input_path (Path): Path to the structure file (e.g. .traj, .extxyz).
        k (int, optional): Number of nearest neighbors. Defaults to 32.
        cutoff (float, optional): Cutoff distance in Å. Defaults to 5.0.
        output_path (Path, optional): Path to save descriptors. Defaults to None.

    Returns:
        np.ndarray: Calculated or loaded descriptors.
    """
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
    """Computes the differential entropy of a subset of structures with respect
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
    """Computes the differential entropy of a subset of structures in a
    trajectory with respect to a reference dataset.

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


@task(
    task_run_name=lambda: (
        f"{task_run.task_name}: {task_run.parameters['atoms'].get_chemical_formula()} - {task_run.parameters['calculator']}"
    ),
    cache_policy=TASK_SOURCE + INPUTS,
)
def run_nve_md(
    atoms: Atoms, calculator: MLIPEnum | BaseCalculator | str, calculator_kwargs: dict | None, traj_file: Path
):
    """Run NVE molecular dynamics simulation.

    Args:
        atoms (Atoms): ASE Atoms structure.
        calculator (MLIPEnum | BaseCalculator | str): Model or calculator to use.
        calculator_kwargs (dict, optional): Kwargs for calculator initialization.
        traj_file (Path): Path to save the trajectory file.

    Returns:
        dict: Results from the MD task.
    """
    return MD.with_options(
        refresh_cache=True,
    )(
        atoms,
        calculator=get_calculator(
            calculator, calculator_kwargs
        ),  # wrap calculater inside task for separate calculator instances
        ensemble="nve",
        dynamics="velocityverlet",
        time_step=1.0,  # fs
        total_time=5000,  # 5 ps = 5000 fs
        temperature=1000.0,
        traj_file=traj_file,
        traj_interval=1,
        zero_linear_momentum=True,
        zero_angular_momentum=True,
    )


def run_simulations(
    calculator: MLIPEnum | BaseCalculator | str, calculator_kwargs: dict | None, structures: list[Atoms], out_dir: Path
):
    """Runs simulations on a list of structures.

    Parameters:
        calculator (MLIPEnum | BaseCalculator | str): Model to use.
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
        rep_factor = int(np.floor((min_atoms / n_atoms) ** (1 / 3)))  # cube root since it's a 3D replication
        supercell_atoms = atoms.repeat((rep_factor, rep_factor, rep_factor))
        if len(supercell_atoms) > max_atoms:
            logger.info(f"Skipping structure {i} because it has too many atoms ({len(supercell_atoms)} > {max_atoms})")
            continue  # skip if it becomes too large

        # Run NVE MD @ 1000K for 5 ps
        future = run_nve_md.submit(
            supercell_atoms,
            calculator=calculator,
            calculator_kwargs=calculator_kwargs,
            traj_file=out_dir / f"{i}.traj",
        )
        futures.append(future)
    wait(futures)

    return [f.result(raise_on_failure=False) for f in futures]


def _generate_flow_run_name():
    name = flow_run.flow_name
    parameters = flow_run.parameters

    model = parameters["model"]
    reference_path = parameters["reference_path"].stem

    return f"{name}: {model} - {reference_path}"


@flow(
    name="Entropy along trajectory",
    flow_run_name=_generate_flow_run_name,
)
def differential_entropy_along_nve_trajectory(
    calculator: MLIPEnum | BaseCalculator | str,
    calculator_kwargs: dict | None,
    input_path: Path,
    reference_path: Path,
    start_idx: int,
    end_idx: int,
    step: int,
    k: int = 32,
    cutoff: float = 5.0,
    h: float = 0.015,
    work_dir: Path | None = None,
):
    """Computes the differential entropy of a subset of structures in a
    trajectory with respect to a reference dataset.

    Arguments:
        model (MLIPEnum | BaseCalculator | str): Model to use.
        input_path (Path): Path to the directory containing the trajectory files.
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
    if isinstance(calculator, MLIPEnum):
        model_name = calculator.name
    elif isinstance(calculator, BaseCalculator):
        model_name = calculator.__class__.__name__
    elif isinstance(calculator, str) and hasattr(MLIPEnum, calculator):
        model_name = calculator
    else:
        raise ValueError(f"Unsupported calculator: {calculator}")

    # Run simulations
    out_dir = work_dir / model_name if work_dir is not None else Path.cwd() / model_name

    structures = read(input_path, index=":")
    run_simulations(calculator, calculator_kwargs, structures, out_dir)

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
