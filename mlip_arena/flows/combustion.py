from __future__ import annotations

from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
from ase import Atoms, io, units
from ase.calculators.calculator import BaseCalculator
from ase.neighborlist import NeighborList, natural_cutoffs
from prefect import flow, task
from tqdm.auto import tqdm
from huggingface_hub import hf_hub_download

from mlip_arena.models import MLIPEnum
from mlip_arena.tasks.md import run as MD
from mlip_arena.tasks.utils import resolve_calculator_name


def identify_water_molecules(atoms):
    """Identify and count water molecules (H2O) in a structure.

    Args:
        atoms (Atoms): ASE Atoms object.

    Returns:
        int: Number of water molecules found.
    """
    nl = NeighborList(natural_cutoffs(atoms), self_interaction=False, bothways=True)
    nl.update(atoms)

    water_molecule_count = 0
    visited_atoms = set()

    for atom in atoms:
        if atom.symbol == "O" and atom.index not in visited_atoms:
            indices, offsets = nl.get_neighbors(atom.index)
            hydrogen_indices = [idx for idx in indices if atoms[idx].symbol == "H"]
            if len(hydrogen_indices) == 2:
                water_molecule_count += 1
                visited_atoms.update([atom.index, *hydrogen_indices])

    return water_molecule_count


@task
def get_runtime_stats(traj: list[Atoms], atoms0: Atoms):
    """Analyze trajectory for runtime statistics, thermodynamics, and reaction
    progress.

    Args:
        traj (list[Atoms]): List of ASE Atoms from trajectory.
        atoms0 (Atoms): Initial Atoms structure for reference.

    Returns:
        dict: A dictionary of statistics including energies, temperatures, pressures, and water molecule count.
    """
    first_valid_info = None
    for atoms in traj:
        if "datetime" in atoms.info and "step" in atoms.info:
            first_valid_info = atoms.info
            break

    restarts = []
    steps, times = [], []
    PEs, KEs, Ts, Ps = [], [], [], []
    com_drifts = []
    nproducts = []

    info_ok = first_valid_info is not None

    for i, atoms in enumerate(tqdm(traj, desc="Analyzing trajectory")):
        try:
            energy = atoms.get_potential_energy()
            assert np.isfinite(energy), f"invalid energy: {energy}"
        except Exception:
            continue

        if info_ok:
            r = atoms.info.get("restart", 0)
            dt = atoms.info.get("datetime", first_valid_info["datetime"])
            st = atoms.info.get("step", 0)
        else:
            r = 0
            dt = datetime.now()
            st = i * 1000

        restarts.append(r)
        times.append(dt)
        steps.append(st)

        PEs.append(energy)
        KEs.append(atoms.get_kinetic_energy())
        Ts.append(atoms.get_temperature())

        try:
            Ps.append(atoms.get_stress()[:3].mean())
        except Exception:
            Ps.append(np.nan)

        com_drifts.append((atoms.get_center_of_mass() - atoms0.get_center_of_mass()).tolist())
        nproducts.append(identify_water_molecules(atoms))

    restarts = np.array(restarts)
    times = np.array(times)
    steps = np.array(steps)

    # Identify unique blocks
    unique_restarts = np.unique(restarts)

    total_time_seconds = 0.0
    total_steps = 0

    if info_ok:
        # Iterate over unique blocks to calculate averages
        for block in unique_restarts:
            # Get the indices corresponding to the current block
            indices = restarts == block
            if np.any(indices):
                # Extract the corresponding data values
                block_time = times[indices][-1] - times[indices][0]
                total_time_seconds += block_time.total_seconds()
                total_steps += steps[indices][-1] - steps[indices][0]
    else:
        total_time_seconds = 0.0
        total_steps = steps[-1] if len(steps) != 0 else 0

    if info_ok and len(traj) > 1:
        try:
            target_steps = traj[1].info.get("target_steps", 2000000)
        except Exception:
            target_steps = 2000000
    else:
        target_steps = 2000000

    natoms = len(atoms0)

    return {
        "natoms": natoms,
        "total_time_seconds": total_time_seconds,
        "total_steps": total_steps,
        "steps_per_second": total_steps / total_time_seconds if total_time_seconds != 0 else 0,
        "seconds_per_step": total_time_seconds / total_steps if total_steps != 0 else float("inf"),
        "seconds_per_step_per_atom": total_time_seconds / total_steps / natoms if total_steps != 0 else float("inf"),
        "energies": PEs,
        "kinetic_energies": KEs,
        "temperatures": Ts,
        "pressures": Ps,
        "target_steps": target_steps,
        "final_step": steps[-1] if len(steps) != 0 else 0,
        "timestep": steps.tolist() if hasattr(steps, "tolist") else steps,
        "com_drifts": com_drifts,
        "nproducts": nproducts,
    }


@flow
def hydrogen_combustion(
    run_dir: Path,
    calculator: str | MLIPEnum | BaseCalculator,
    calculator_kwargs: dict | None = None,
    dispersion: bool = False,
    dispersion_kwargs: dict | None = None,
):
    """Run hydrogen combustion simulation and analyze results.

    Parameters:
        run_dir (Path): Directory where output files (.traj and .json) will be saved.
        calculator (str | MLIPEnum | BaseCalculator): Model identifier, enum, or instantiated ASE Calculator.
        calculator_kwargs (dict, optional): Additional parameters passed to the model's calculator constructor. Defaults to None.
        dispersion (bool, optional): Whether to enable dispersion corrections (e.g. via TorchDFTD3Calculator). Defaults to False.
        dispersion_kwargs (dict, optional): Keyword arguments for the dispersion calculator. Defaults to None.

    Returns:
        dict: Dictionary containing final state atoms and execution stats from the molecular dynamics simulation.
    """
    atoms_path = hf_hub_download(
        repo_id="atomind/mlip-arena",
        filename="combustion/H256O128.extxyz",
        repo_type="dataset",
    )
    atoms = io.read(atoms_path)
    assert isinstance(atoms, Atoms)

    model_name = resolve_calculator_name(calculator)
    traj_file = run_dir / f"{model_name}_{atoms.get_chemical_formula()}.traj"
    json_fpath = run_dir / f"{model_name}_{atoms.get_chemical_formula()}.json"

    future = MD.submit(
        atoms=atoms,
        calculator=calculator,
        calculator_kwargs=calculator_kwargs,
        dispersion=dispersion,
        dispersion_kwargs=dispersion_kwargs,
        ensemble="nvt",
        dynamics="nose-hoover",
        time_step=None,
        dynamics_kwargs=dict(ttime=25 * units.fs, pfactor=None),
        total_time=1_000_000,
        temperature=[300, 3000, 3000, 300],
        pressure=None,
        velocity_seed=0,
        traj_file=traj_file,
        traj_interval=1000,
        restart=True,
    )
    result = future.result()

    traj = io.read(traj_file, index=":")

    assert len(traj) >= 2000, f"Trajectory has only {len(traj)} frames and is not complete."

    assert np.allclose(traj[0].positions, atoms.positions), "Initial positions do not match."

    stats = get_runtime_stats(traj, atoms0=traj[0])

    formula = traj_file.stem.split("_")[-1]
    reaction = "hydrogen"
    max_nproducts = 128  # Maximum possible number of water molecules (for H256O128)

    data = {
        "formula": formula,
        "method": model_name,
        "reaction": reaction,
        **stats,
        "yield": stats["nproducts"][-1] / max_nproducts,
    }

    df = pd.DataFrame([data])
    df.to_json(json_fpath, orient="records")

    return result
