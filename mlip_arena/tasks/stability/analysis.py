from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
from ase import Atoms, units
from ase.io import read
from loguru import logger
from tqdm.auto import tqdm


def get_runtime_stats(traj: list[Atoms], atoms0: Atoms):
    """
    Compute runtime statistics for an ASE trajectory.
    
    Parameters
    ----------
    traj : list[ase.Atoms]
        Sequence of ASE Atoms frames. Each frame is expected to have an `info`
        dict containing at least the keys 'restart', 'datetime', and 'step'. The
        function reads `target_steps` from traj[1].info.
    atoms0 : ase.Atoms
        Reference Atoms object used to compute center-of-mass drift and to
        determine the number of atoms.
    
    Returns
    -------
    dict
        Dictionary with runtime and per-frame statistics:
        - natoms: int, number of atoms from atoms0.
        - total_time_seconds: float, sum of durations of unique restart blocks (seconds).
        - total_steps: int, sum of step differences across unique restart blocks.
        - steps_per_second: float, total_steps / total_time_seconds (0 if total_time_seconds == 0).
        - seconds_per_step: float, total_time_seconds / total_steps (inf if total_steps == 0).
        - seconds_per_step_per_atom: float, seconds per step normalized by atom count (inf if total_steps == 0).
        - energies: list of potential energies for frames successfully parsed.
        - kinetic_energies: list of per-frame kinetic energies.
        - temperatures: list of frame temperatures.
        - pressures: list of mean pressures per frame (NaN if unavailable).
        - target_steps: int, target number of steps read from traj[1].info.
        - final_step: int, last recorded step number (0 if no valid frames).
        - timestep: np.ndarray of step numbers for valid frames.
        - com_drifts: list of center-of-mass drift vectors (relative to atoms0).
    """
    restarts = []
    steps, times = [], []
    Ts, Ps, Es, KEs = [], [], [], []
    com_drifts = []

    for atoms in traj:
        try:
            energy = atoms.get_potential_energy()
            assert np.isfinite(energy), f"invalid energy: {energy}"
        except Exception:
            continue

        restarts.append(atoms.info["restart"])
        times.append(atoms.info["datetime"])
        steps.append(atoms.info["step"])
        Es.append(energy)
        KEs.append(atoms.get_kinetic_energy())
        Ts.append(atoms.get_temperature())
        try:
            Ps.append(atoms.get_stress()[:3].mean())
        except Exception:
            Ps.append(np.nan)
        com_drifts.append(
            (atoms.get_center_of_mass() - atoms0.get_center_of_mass()).tolist()
        )

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
        "energies": Es,
        "kinetic_energies": KEs,
        "temperatures": Ts,
        "pressures": Ps,
        "target_steps": target_steps,
        "final_step": steps[-1] if len(steps) != 0 else 0,
        "timestep": steps,
        "com_drifts": com_drifts,
    }


def gather_results(
    run_dir: Path, prefix: str, run_type: Literal["nvt", "npt"]
) -> pd.DataFrame:
    df = pd.DataFrame()

    run_dir = Path(run_dir)
    files = list(run_dir.glob(f"{prefix}_*{run_type}.traj"))

    for fpath in tqdm(files, desc=prefix):
        try:
            traj = read(fpath, index=":")
        except Exception as e:
            logger.warning(f"Error reading {fpath}: {e}")
            continue

        try:
            stats = get_runtime_stats(traj, atoms0=traj[0])
            df = pd.concat(
                [
                    df,
                    pd.DataFrame(
                        {
                            # "model": model_name,
                            "formula": traj[0].get_chemical_formula(),
                            "normalized_timestep": stats["timestep"]
                            / stats["target_steps"],
                            "normalized_final_step": stats["final_step"]
                            / stats["target_steps"],
                            "pressure": np.array(stats["pressures"]) / units.GPa,
                        }
                        | stats
                    ),
                ],
                ignore_index=True,
            )
        except Exception as e:
            logger.warning(f"Error processing {fpath}: {e}")
            continue
    return df