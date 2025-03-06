"""
Widom insertion workflow to calculate Henry coefficient and heat of adsorption for a given MOF structure and gas molecule.


This script is heavily adapted from the `DAC-SIM <https://github.com/hspark1212/DAC-SIM>`_ package. Please cite the original work if you use this script.

References
~~~~~~~~~~~
- Lim, Y., Park, H., Walsh, A., & Kim, J. (2024). Accelerating CO₂ Direct Air Capture Screening for Metal-Organic Frameworks with a Transferable Machine Learning Force Field.
"""

from collections import defaultdict
from pathlib import Path
from typing import IO, Any

import numpy as np
from prefect import flow, task
from prefect.cache_policies import INPUTS, TASK_SOURCE
from prefect.futures import wait
from prefect.logging import get_run_logger
from prefect.runtime import task_run
from prefect.states import State
from tqdm.auto import tqdm

from ase import Atoms, units
from ase.atoms import Atoms
from ase.build import molecule
from ase.filters import Filter
from ase.io.trajectory import Trajectory, TrajectoryWriter
from ase.optimize.optimize import Optimizer
from ase.calculators.calculator import BaseCalculator
from mlip_arena.models import MLIPEnum
from mlip_arena.tasks.optimize import run as OPT
from mlip_arena.tasks.utils import get_calculator, logger

from .grid import get_accessible_positions
from .input import get_atoms_from_db


def add_molecule(gas: Atoms, rotate: bool = True, translate: tuple = None) -> Atoms:
    """
    Add a molecule to the simulation cell

    Parameters
    ----------
    gas : Atoms
        The gas molecule to add
    rotate : bool, optional
        If True, rotate the molecule randomly, by default True
    translate : tuple, optional
        The translation of the molecule, by default None

    Returns
    -------
    Atoms
        The gas molecule added to the simulation cell

    Raises
    ------
    ValueError
        If the translate is not a 3-tuple, raise an error

    Examples
    --------
    >>> from ml_mc.utils import molecule, add_gas
    >>> gas = molecule('H2O')
    >>> gas = add_gas(gas, rotate=True, translate=(0, 0, 0))
    """
    gas = gas.copy()
    if rotate:
        angle = np.random.rand() * 360
        axis = np.random.rand(3)
        gas.rotate(v=axis, a=angle)
    if translate is not None:
        if len(translate) != 3:
            raise ValueError("translate must be a 3-tuple")
        gas.translate(translate)
    return gas


def get_atomic_density(atoms: Atoms) -> float:
    """
    Calculate atomic density of the atoms.

    Parameters
    ----------
    atoms : Atoms
        The Atoms object to operate on.

    Returns
    -------
    float
        Atomic density of the atoms in kg/m³.
    """
    volume = atoms.get_volume() * 1e-30  # Convert Å³ to m³
    total_mass = sum(atoms.get_masses()) * units._amu  # Convert amu to kg
    return total_mass / volume


def _generate_task_run_name():
    task_name = task_run.task_name
    parameters = task_run.parameters

    structure = parameters["structure"]
    gas = parameters["gas"]
    calculator = parameters["calculator"]

    return f"{task_name}: {structure.get_chemical_formula()} + {gas.get_chemical_formula()} - {calculator}"


@task(
    name="Widom Insertion",
    task_run_name=_generate_task_run_name,
    cache_policy=TASK_SOURCE + INPUTS,
)
def widom_insertion(
    # init
    structure: Atoms,
    gas: Atoms,
    calculator: BaseCalculator,
    optimizer: Optimizer | str = "FIRE",
    optimizer_kwargs: dict | None = None,
    filter: Filter | str | None = "FrechetCell",
    filter_kwargs: dict | None = None,
    criterion: dict | None = dict(fmax=0.05, steps=50),
    temperature: float = 300,
    init_structure_optimize_loops: int = 10,
    init_gas_optimize: bool = True,
    traj_file: str | Path | None = None,
    # run
    num_insertions: int = 5000,
    grid_spacing: float = 0.15,
    cutoff_distance: float = 1.50,
    min_interplanar_distance: float = 6.0,
    fold: int = 3,
    random_seed: int | None = None,
) -> dict[str, Any] | State:
    """
    Run the Widom insertion algorithm to calculate the Henry coefficient and heat of adsorption.

    Parameters
    ----------
    num_insertions : int, default=5000
        Number of random insertions of the gas molecule during simulation.
    grid_spacing : float, default=0.15
        Spacing of the grid for possible gas insertion points, in angstroms.
    cutoff_distance : float, default=1.50
        When the distance between framework atoms and the gas molecule is less than this value, the insertion is rejected. In angstroms.
    min_interplanar_distance : float, default=6.0
        When the interplanar distance of the framework is less than this value, a supercell is constructed. In angstroms.
    fold : int, default=3
        Number of repetitions of Widom insertion to improve statistics.
    random_seed : int, optional
        Seed for the random number generator for reproducibility.

    Returns
    -------
    Dict[str, Any]
        Dictionary containing the calculated Henry coefficient (mol/kg Pa), averaged interaction energy (eV), and heat of adsorption (kJ/mol) over the number of folds.
    """

    structure = structure.copy()
    gas = gas.copy()

    # Optimize structure and gas molecule
    while init_structure_optimize_loops > 0:
        logger.info("Optimizing cell")
        state = OPT(
            atoms=structure,
            calculator=calculator,
            optimizer=optimizer,
            optimizer_kwargs=optimizer_kwargs,
            filter=filter,
            filter_kwargs=filter_kwargs,
            criterion=criterion,
            return_state=True,
        )

        if state.is_failed():
            return state

        result = state.result(raise_on_failure=False)
        structure = result["atoms"]
        if result["converged"]:
            break

        logger.info("Optimizing atoms with fixed cell")
        state = OPT(
            atoms=structure,
            calculator=calculator,
            optimizer=optimizer,
            optimizer_kwargs=optimizer_kwargs,
            filter=None,
            filter_kwargs=None,
            criterion=criterion,
            return_state=True,
        )

        if state.is_failed():
            return state
        
        result = state.result(raise_on_failure=False)
        structure = result["atoms"]
        if result["converged"]:
            break

        init_structure_optimize_loops -= 1

    if init_gas_optimize:
        logger.info("Optimizing gas molecule")
        state = OPT(
            atoms=gas,
            calculator=calculator,
            optimizer=optimizer,
            optimizer_kwargs=optimizer_kwargs,
            filter=None,
            criterion=criterion,
            return_state=True,
        )

        if state.is_failed():
            return state

        gas = state.result(raise_on_failure=False)["atoms"]

    # Calculate accessible positions
    ret = get_accessible_positions(
        structure=structure,
        grid_spacing=grid_spacing,
        cutoff_distance=cutoff_distance,
        min_interplanar_distance=min_interplanar_distance,
    )
    pos_grid = ret["pos_grid"]
    idx_accessible_pos = ret["idx_accessible_pos"]
    structure = ret["structure"]  # supercell structure if necessary

    logger.info(
        f"Number of accessible positions: {len(idx_accessible_pos)} out of total {len(pos_grid)}"
    )

    calc = calculator
    # Calculate energies for structure and gas
    energy_structure = calc.get_potential_energy(structure)
    energy_gas = calc.get_potential_energy(gas)

    # Set random seed if provided
    if random_seed is not None:
        np.random.seed(random_seed)
        logger.info(f"Setting random seed: {random_seed}")

    if traj_file is not None:
        traj_file = Path(traj_file)
        traj_file.parent.mkdir(parents=True, exist_ok=True)
        # TODO: checkpoint and restart
        traj = Trajectory(traj_file, "a")
    else:
        traj = None

    # Run Widom insertion algorithm
    
    results = defaultdict(list)
    for ifold in range(fold):

        nsteps = 0

        np.random.shuffle(idx_accessible_pos)
        interaction_energies = np.zeros(num_insertions)

        pbar = tqdm(total=num_insertions, desc=f"Fold {ifold + 1}/{fold}")
        for rand_idx in idx_accessible_pos:
            # assert rand_idx in idx_accessible_pos

            if nsteps >= num_insertions:
                break

            # Add gas molecule to the accessible position
            pos = pos_grid[rand_idx]
            added_gas = add_molecule(gas, rotate=True, translate=pos)
            structure_with_gas = structure + added_gas
            structure_with_gas.wrap()  # wrap atoms to unit cell

            # Calculate interaction energy
            structure_with_gas.calc = calc
            total_energy = structure_with_gas.get_potential_energy()  # [eV]
            interaction_energy = total_energy - energy_structure - energy_gas  # [eV]

            boltzmann_factor = np.exp(
                -interaction_energy / (temperature * units._k / units._e)
            )

            # Handle exponential overflow that can cause numerical instability

            max_exp_arg = 700  # np.exp(700) is close to the max float64
            if boltzmann_factor > np.exp(max_exp_arg):
                logger.warning(
                    f"Exponential overflow detected. Rejecting this step and retrying."
                )
                continue

            interaction_energies[nsteps] = interaction_energy
            nsteps += 1
            pbar.update(1)

            # Write trajectory
            if isinstance(traj, TrajectoryWriter):
                traj.write(structure_with_gas)

        pbar.close()

        assert nsteps == num_insertions, "Cannot reach the number of insertions due to too many invalid steps."

        # Calculate ensemble averages properties
        # units._e [J/eV], units._k [J/K], units._k / units._e # [eV/K]
        boltzmann_factors = np.exp(
            -interaction_energies / (temperature * units._k / units._e)
        )

        # KH = <exp(-E/RT)> / (R * T)
        atomic_density = get_atomic_density(structure)  # [kg / m^3]
        kh = (
            boltzmann_factors.sum()
            / num_insertions
            / (units._k * units._Nav)  # R = [J / mol K] = [Pa m^3 / mol K]
            / temperature  # T = [K] -> [mol/ m^3 Pa]
            / atomic_density  #  = [kg / m^3] -> [mol / kg Pa]
        )  # [mol/kg Pa]

        # U = < E * exp(-E/RT) > / <exp(-E/RT)> # [eV]
        u = (interaction_energies * boltzmann_factors).sum() / boltzmann_factors.sum()

        # Qst = U - RT # [kJ/mol]
        qst = (u * units._e - units._k * temperature) * units._Nav * 1e-3

        results["henry_coefficient"].append(kh)
        results["averaged_interaction_energy"].append(u)
        results["heat_of_adsorption"].append(qst)
        
    return results


@flow
def run(
    db_path: Path | str = "mofs.db",
):
    states = []
    for model in MLIPEnum:
        for atoms in tqdm(get_atoms_from_db(db_path)):
            state = widom_insertion.submit(
                atoms,
                molecule("CO2"),
                calculator=get_calculator(
                    model,
                    dispersion=True,
                ),
                return_state=True,
            )
            states.append(state)

    wait(states)
    return [s.result(raise_on_failture=False) for s in states if s.is_completed()]
