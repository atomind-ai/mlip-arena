"""
Define molecular dynamics task.

This script has been adapted from Atomate2 MLFF MD workflow written by Aaron Kaplan and Yuan Chiang
https://github.com/materialsproject/atomate2/blob/main/src/atomate2/forcefields/md.py

atomate2 Copyright (c) 2015, The Regents of the University of
California, through Lawrence Berkeley National Laboratory (subject
to receipt of any required approvals from the U.S. Dept. of Energy).
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions
are met:

(1) Redistributions of source code must retain the above copyright
notice, this list of conditions and the following disclaimer.

(2) Redistributions in binary form must reproduce the above
copyright notice, this list of conditions and the following
disclaimer in the documentation and/or other materials provided with
the distribution.

(3) Neither the name of the University of California, Lawrence
Berkeley National Laboratory, U.S. Dept. of Energy nor the names of
its contributors may be used to endorse or promote products derived
from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
"AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
POSSIBILITY OF SUCH DAMAGE.

You are under no obligation whatsoever to provide any bug fixes,
patches, or upgrades to the features, functionality or performance
of the source code ("Enhancements") to anyone; however, if you
choose to make your Enhancements available either publicly, or
directly to Lawrence Berkeley National Laboratory or its
contributors, without imposing a separate written license agreement
for such Enhancements, then you hereby grant the following license:
a  non-exclusive, royalty-free perpetual license to install, use,
modify, prepare derivative works, incorporate into other computer
software, distribute, and sublicense such enhancements or derivative
works thereof, in binary and source code form.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime
from pathlib import Path
from typing import Literal

import numpy as np
from ase import Atoms, units
from ase.calculators.calculator import BaseCalculator
from ase.io import read
from ase.io.trajectory import Trajectory
from ase.md.andersen import Andersen
from ase.md.langevin import Langevin
from ase.md.md import MolecularDynamics
from ase.md.npt import NPT
from ase.md.nptberendsen import NPTBerendsen
from ase.md.nvtberendsen import NVTBerendsen
from ase.md.velocitydistribution import (
    MaxwellBoltzmannDistribution,
    Stationary,
    ZeroRotation,
)
from ase.md.verlet import VelocityVerlet
from prefect import task
from prefect.cache_policies import INPUTS, TASK_SOURCE
from prefect.runtime import task_run
from scipy.interpolate import interp1d
from scipy.linalg import schur
from tqdm.auto import tqdm

_valid_dynamics: dict[str, tuple[str, ...]] = {
    "nve": ("velocityverlet",),
    "nvt": ("nose-hoover", "langevin", "andersen", "berendsen"),
    "npt": ("nose-hoover", "berendsen"),
}

_preset_dynamics: dict = {
    "nve_velocityverlet": VelocityVerlet,
    "nvt_andersen": Andersen,
    "nvt_berendsen": NVTBerendsen,
    "nvt_langevin": Langevin,
    "nvt_nose-hoover": NPT,
    "npt_berendsen": NPTBerendsen,
    "npt_nose-hoover": NPT,
}


def _interpolate_quantity(values: Sequence | np.ndarray, n_pts: int) -> np.ndarray:
    """Interpolate temperature / pressure on a schedule."""
    n_vals = len(values)
    return np.interp(
        np.linspace(0, n_vals - 1, n_pts + 1),
        np.linspace(0, n_vals - 1, n_vals),
        values,
    )


def _get_ensemble_schedule(
    ensemble: Literal["nve", "nvt", "npt"] = "nvt",
    n_steps: int = 1000,
    temperature: float | Sequence | np.ndarray | None = 300.0,
    pressure: float | Sequence | np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    if ensemble == "nve":
        # Disable thermostat and barostat
        temperature = np.nan
        pressure = np.nan
        t_schedule = np.full(n_steps + 1, temperature)
        p_schedule = np.full(n_steps + 1, pressure)
        return t_schedule, p_schedule

    if isinstance(temperature, Sequence) or (
        isinstance(temperature, np.ndarray) and temperature.ndim == 1
    ):
        t_schedule = _interpolate_quantity(temperature, n_steps)
    # NOTE: In ASE Langevin dynamics, the temperature are normally
    # scalars, but in principle one quantity per atom could be specified by giving
    # an array. This is not implemented yet here.
    else:
        t_schedule = np.full(n_steps + 1, temperature)

    if ensemble == "nvt":
        pressure = np.nan
        p_schedule = np.full(n_steps + 1, pressure)
        return t_schedule, p_schedule

    if isinstance(pressure, Sequence) or (
        isinstance(pressure, np.ndarray) and pressure.ndim == 1
    ):
        p_schedule = _interpolate_quantity(pressure, n_steps)
    elif isinstance(pressure, np.ndarray) and pressure.ndim == 3:
        p_schedule = interp1d(np.arange(n_steps + 1), pressure, kind="linear")
        assert isinstance(p_schedule, np.ndarray)
    else:
        p_schedule = np.full(n_steps + 1, pressure)

    return t_schedule, p_schedule


def _get_ensemble_defaults(
    ensemble: Literal["nve", "nvt", "npt"],
    dynamics: str | MolecularDynamics,
    t_schedule: np.ndarray,
    p_schedule: np.ndarray,
    dynamics_kwargs: dict | None = None,
) -> dict:
    """Update ASE MD kwargs"""
    dynamics_kwargs = dynamics_kwargs or {}

    if ensemble == "nve":
        dynamics_kwargs.pop("temperature", None)
        dynamics_kwargs.pop("temperature_K", None)
        dynamics_kwargs.pop("externalstress", None)
    elif ensemble == "nvt":
        dynamics_kwargs["temperature_K"] = t_schedule[0]
        dynamics_kwargs.pop("externalstress", None)
    elif ensemble == "npt":
        dynamics_kwargs["temperature_K"] = t_schedule[0]
        dynamics_kwargs["externalstress"] = p_schedule[0]  # * 1e3 * units.bar

    if isinstance(dynamics, str) and dynamics.lower() == "langevin":
        dynamics_kwargs["friction"] = dynamics_kwargs.get(
            "friction",
            10.0 * 1e-3 / units.fs,  # Same default as in VASP: 10 ps^-1
        )

    return dynamics_kwargs


def _generate_task_run_name():
    task_name = task_run.task_name
    parameters = task_run.parameters

    atoms = parameters["atoms"]
    calculator = parameters["calculator"]

    return f"{task_name}: {atoms.get_chemical_formula()} - {calculator}"


@task(
    name="MD", task_run_name=_generate_task_run_name, cache_policy=TASK_SOURCE + INPUTS
)
def run(
    atoms: Atoms,
    calculator: BaseCalculator,
    ensemble: Literal["nve", "nvt", "npt"] = "nvt",
    dynamics: str | MolecularDynamics = "langevin",
    time_step: float | None = None,  # fs
    total_time: float = 1000,  # fs
    temperature: float | Sequence | np.ndarray | None = 300.0,  # K
    pressure: float | Sequence | np.ndarray | None = None,  # eV/A^3
    dynamics_kwargs: dict | None = None,
    velocity_seed: int | None = None,
    zero_linear_momentum: bool = True,
    zero_angular_momentum: bool = True,
    traj_file: str | Path | None = None,
    traj_interval: int = 1,
    restart: bool = True,
):
    """
    Run an ASE molecular dynamics (MD) simulation for the given Atoms and calculator.
    
    Summary:
        Execute an MD trajectory with configurable ensemble, dynamics, timestep, time-dependent temperature/pressure schedules, trajectory output, and optional restart from an existing trajectory.
    
    Parameters:
        atoms (Atoms): Atomic configuration to simulate; a copy is used and the provided calculator is set on it.
        calculator (BaseCalculator): Calculator used for forces/energies.
        ensemble (Literal["nve","nvt","npt"], optional): Thermodynamic ensemble. Default: "nvt".
        dynamics (str | MolecularDynamics, optional): Dynamics method name (e.g., "langevin", "andersen", "velocityverlet") or an ASE MolecularDynamics class. Must be compatible with the chosen ensemble. Default: "langevin".
        time_step (float | None, optional): Integration timestep in femtoseconds. If None, defaults to 0.5 fs when hydrogen isotopes are present, otherwise 2.0 fs.
        total_time (float, optional): Total simulation time in femtoseconds. Default: 1000.
        temperature (float | Sequence | np.ndarray | None, optional): Temperature schedule in Kelvin. May be a scalar (constant temperature), a 1D sequence/array to be linearly interpolated to simulation steps, or None/NaN to disable thermostat (used by "nve").
        pressure (float | Sequence | np.ndarray | None, optional): Pressure or external stress schedule in eV/Å³. May be a scalar, a 1D sequence/array to be interpolated, or a full 3×3 (or sequence of 3×3) ndarray for stepwise stress. NaN disables barostat.
        dynamics_kwargs (dict | None, optional): Extra keyword arguments forwarded to the ASE dynamics class. For "langevin" a default friction is added if not provided. Special key: "fraction_traceless" (float, default 1.0) is consumed to set NPT fraction traceless and removed from the kwargs.
        velocity_seed (int | None, optional): RNG seed for initializing Maxwell–Boltzmann velocities.
        zero_linear_momentum (bool, optional): If True, remove center-of-mass linear momentum after velocity initialization. Default: True.
        zero_angular_momentum (bool, optional): If True, remove net rotation after velocity initialization. Default: True.
        traj_file (str | Path | None, optional): Path to trajectory file to write frames. If provided, directory is created. If restart is True and the file exists, the last frame will be used to resume the run when readable.
        traj_interval (int, optional): Interval (in MD steps) between trajectory frame writes. Default: 1.
        restart (bool, optional): When True, attempt to resume from existing traj_file; if resume fails a new trajectory is started. Default: True.
    
    Returns:
        dict: {
            "atoms": Atoms,        # final Atoms object after the run
            "runtime": timedelta,  # elapsed wall-clock time for the MD run
            "n_steps": int         # number of steps performed (remaining steps computed from total_time and timestep)
        }
    
    Raises:
        ValueError: If the specified dynamics is invalid or incompatible with the chosen ensemble.
    
    Side effects:
        - May write to traj_file and create its parent directories.
        - For NPT dynamics, the atomic cell is converted to an upper-triangular form to satisfy ASE requirements.
    """

    atoms = atoms.copy()

    atoms.calc = calculator

    if time_step is None:
        # If a structure contains an isotope of hydrogen, set default `time_step`
        # to 0.5 fs, and 2 fs otherwise.
        has_h_isotope = "H" in atoms.get_chemical_symbols()
        time_step = 0.5 if has_h_isotope else 2.0

    n_steps = int(total_time / time_step)
    target_steps = n_steps

    t_schedule, p_schedule = _get_ensemble_schedule(
        ensemble=ensemble,
        n_steps=n_steps,
        temperature=temperature,
        pressure=pressure,
    )

    dynamics_kwargs = _get_ensemble_defaults(
        ensemble=ensemble,
        dynamics=dynamics,
        t_schedule=t_schedule,
        p_schedule=p_schedule,
        dynamics_kwargs=dynamics_kwargs,
    )

    if isinstance(dynamics, str):
        # Use known dynamics if `self.dynamics` is a str
        dynamics = dynamics.lower()
        if dynamics not in _valid_dynamics[ensemble]:
            raise ValueError(
                f"{dynamics} thermostat not available for {ensemble}."
                f"Available {ensemble} thermostats are:"
                " ".join(_valid_dynamics[ensemble])
            )
        if ensemble == "nve":
            dynamics = "velocityverlet"
        md_class = _preset_dynamics[f"{ensemble}_{dynamics}"]
    elif dynamics is MolecularDynamics:
        md_class = dynamics
    else:
        raise ValueError(f"Invalid dynamics: {dynamics}")

    if md_class is NPT:
        # Note that until md_func is instantiated, isinstance(md_func,NPT) is False
        # ASE NPT implementation requires upper triangular cell
        u, _ = schur(atoms.get_cell(complete=True), output="complex")
        atoms.set_cell(u.real, scale_atoms=True)

    last_step = 0

    if traj_file is not None:
        traj_file = Path(traj_file)
        traj_file.parent.mkdir(parents=True, exist_ok=True)

        if restart and traj_file.exists():
            try:
                last_atoms = read(traj_file, index="-1")
                assert isinstance(last_atoms, Atoms)
                last_step = last_atoms.info.get("step")
                n_steps -= last_step
                traj = Trajectory(traj_file, "a", atoms)
                atoms.set_positions(last_atoms.get_positions())
                atoms.set_momenta(last_atoms.get_momenta())
            except Exception:
                traj = Trajectory(traj_file, "w", atoms)

                if not np.isnan(t_schedule).any():
                    MaxwellBoltzmannDistribution(
                        atoms=atoms,
                        temperature_K=t_schedule[last_step],
                        rng=np.random.default_rng(seed=velocity_seed),
                    )

                if zero_linear_momentum:
                    Stationary(atoms)
                if zero_angular_momentum:
                    ZeroRotation(atoms)
        else:
            traj = Trajectory(traj_file, "w", atoms)

            if not np.isnan(t_schedule).any():
                MaxwellBoltzmannDistribution(
                    atoms=atoms,
                    temperature_K=t_schedule[last_step],
                    rng=np.random.default_rng(seed=velocity_seed),
                )

            if zero_linear_momentum:
                Stationary(atoms)
            if zero_angular_momentum:
                ZeroRotation(atoms)

    fraction_traceless = dynamics_kwargs.pop("fraction_traceless", 1.0)

    md_runner = md_class(
        atoms=atoms,
        timestep=time_step * units.fs,
        **dynamics_kwargs,
    )
    if md_class is NPT:
        md_runner.set_fraction_traceless(fraction_traceless)

    if traj_file is not None:
        md_runner.attach(traj.write, interval=traj_interval)

    with tqdm(total=n_steps, desc=f"MD {atoms.get_chemical_formula()}") as pbar:

        def _callback(dyn: MolecularDynamics = md_runner) -> None:
            """
            Update MD runner state and per-frame metadata during each MD step.
            
            This callback computes the current step as the sum of the saved last_step and the runner's nsteps, writes per-frame metadata into dyn.atoms.info (restart, datetime, step, and target_steps), updates the thermostat temperature when applicable, updates the barostat stress when applicable, and advances the progress bar.
            
            Parameters:
                dyn (MolecularDynamics): The active ASE MD runner whose atoms and state are being updated.
            
            Returns:
                None
            """
            step = last_step + dyn.nsteps
            dyn.atoms.info["restart"] = last_step
            dyn.atoms.info["datetime"] = datetime.now()
            dyn.atoms.info["step"] = step
            dyn.atoms.info["target_steps"] = target_steps
            if ensemble == "nve":
                return
            dyn.set_temperature(temperature_K=t_schedule[step])
            if ensemble == "nvt":
                return
            dyn.set_stress(p_schedule[step])
            pbar.update()

        md_runner.attach(_callback, interval=1)

        start_time = datetime.now()
        complete = md_runner.run(steps=n_steps) if n_steps > 0 else True
        end_time = datetime.now()

    if traj_file is not None:
        if complete and read(traj_file, index="-1").info.get("step") != target_steps:
            # If the MD run was completed but the last step in the trajectory
            # does not match the target steps, write the final frame.
            atoms.info["restart"] = last_step + md_runner.nsteps
            atoms.info["datetime"] = datetime.now()
            atoms.info["step"] = target_steps
            atoms.info["target_steps"] = target_steps
            traj.write(atoms)
        traj.close()

    return {
        "atoms": atoms,
        "runtime": end_time - start_time,
        "n_steps": n_steps,
    }