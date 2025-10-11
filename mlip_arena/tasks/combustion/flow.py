from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from ase import Atoms, io, units
from ase.calculators.calculator import BaseCalculator
from ase.neighborlist import NeighborList, natural_cutoffs
from prefect import flow, task
from tqdm.auto import tqdm

from mlip_arena.models import MLIPEnum
from mlip_arena.tasks.md import run as MD
from mlip_arena.tasks.utils import get_calculator


def identify_water_molecules(atoms):
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
    restarts = []
    steps, times = [], []
    Ts, Ps, PEs, KEs = [], [], [], []
    com_drifts = []
    nproducts = []

    for atoms in tqdm(traj, desc="Analyzing trajectory"):
        try:
            energy = atoms.get_potential_energy()
            assert np.isfinite(energy), f"invalid energy: {energy}"

            restarts.append(atoms.info["restart"])
            times.append(atoms.info["datetime"])
            steps.append(atoms.info["step"])

            PEs.append(energy)
            KEs.append(atoms.get_kinetic_energy())
            Ts.append(atoms.get_temperature())

            try:
                Ps.append(atoms.get_stress()[:3].mean())
            except Exception:
                Ps.append(np.nan)

            com_drifts.append(
                (atoms.get_center_of_mass() - atoms0.get_center_of_mass()).tolist()
            )
            nproducts.append(identify_water_molecules(atoms))
        except Exception:
            continue

    restarts = np.array(restarts)
    times = np.array(times)
    steps = np.array(steps)

    # Identify unique blocks
    unique_restarts = np.unique(restarts)

    total_time_seconds = 0
    total_steps = 0

    # Iterate over unique blocks to calculate averages
    for block in unique_restarts:
        # Get the indices corresponding to the current block
        # indices = np.where(restarts == block)[0]
        indices = restarts == block
        # Extract the corresponding data values
        block_time = times[indices][-1] - times[indices][0]
        total_time_seconds += block_time.total_seconds()
        total_steps += steps[indices][-1] - steps[indices][0]

    target_steps = traj[1].info["target_steps"]
    natoms = len(atoms0)

    return {
        "natoms": natoms,
        "total_time_seconds": total_time_seconds,
        "total_steps": total_steps,
        "steps_per_second": total_steps / total_time_seconds
        if total_time_seconds != 0
        else 0,
        "seconds_per_step": total_time_seconds / total_steps
        if total_steps != 0
        else float("inf"),
        "seconds_per_step_per_atom": total_time_seconds / total_steps / natoms
        if total_steps != 0
        else float("inf"),
        "energies": PEs,
        "kinetic_energies": KEs,
        "temperatures": Ts,
        "pressures": Ps,
        "target_steps": target_steps,
        "final_step": steps[-1] if len(steps) != 0 else 0,
        "timestep": steps,
        "com_drifts": com_drifts,
        "nproducts": nproducts,
    }


@flow
def hydrogen_combustion(model: str | BaseCalculator, run_dir: Path):
    """Run hydrogen combustion simulation and analyze results."""
    atoms = io.read(Path(__file__).parent / "H256O128.extxyz")
    assert isinstance(atoms, Atoms)

    model = MLIPEnum[model] if isinstance(model, str) else model
    calculator = get_calculator(model) if isinstance(model, MLIPEnum) else model

    model_name = model.name if isinstance(model, MLIPEnum) else calculator.__class__.__name__
    traj_file = run_dir / f"{model_name}_{atoms.get_chemical_formula()}.traj"
    json_fpath = run_dir / f"{model_name}_{atoms.get_chemical_formula()}.json"

    result = MD(
        atoms=atoms,
        calculator=calculator,
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

    traj = io.read(traj_file, index=":")

    assert len(traj) >= 2000, (
        f"Trajectory has only {len(traj)} frames and is not complete."
    )

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
