"""
Define molecular dynamics tasks.

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
from datetime import datetime, timedelta
from pathlib import Path
from typing import Literal

import numpy as np
from ase import Atoms, units
from ase.calculators.calculator import Calculator
from ase.calculators.mixing import SumCalculator
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
from prefect.tasks import task_input_hash
from scipy.interpolate import interp1d
from scipy.linalg import schur
from torch_dftd.torch_dftd3_calculator import TorchDFTD3Calculator
from tqdm.auto import tqdm

from mlip_arena.models import MLIPEnum
from mlip_arena.models.utils import get_freer_device

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
    elif isinstance(pressure, np.ndarray) and pressure.ndim == 4:
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
    ase_md_kwargs: dict | None = None,
) -> dict:
    """Update ASE MD kwargs"""
    ase_md_kwargs = ase_md_kwargs or {}

    if ensemble == "nve":
        ase_md_kwargs.pop("temperature", None)
        ase_md_kwargs.pop("temperature_K", None)
        ase_md_kwargs.pop("externalstress", None)
    elif ensemble == "nvt":
        ase_md_kwargs["temperature_K"] = t_schedule[0]
        ase_md_kwargs.pop("externalstress", None)
    elif ensemble == "npt":
        ase_md_kwargs["temperature_K"] = t_schedule[0]
        ase_md_kwargs["externalstress"] = p_schedule[0]  # * 1e3 * units.bar

    if isinstance(dynamics, str) and dynamics.lower() == "langevin":
        ase_md_kwargs["friction"] = ase_md_kwargs.get(
            "friction",
            10.0 * 1e-3 / units.fs,  # Same default as in VASP: 10 ps^-1
        )

    return ase_md_kwargs


@task(cache_key_fn=task_input_hash, cache_expiration=timedelta(days=1))
def run(
    atoms: Atoms,
    calculator_name: str | MLIPEnum,
    calculator_kwargs: dict | None,
    dispersion: str | None = None,
    dispersion_kwargs: dict | None = None,
    device: str | None = None,
    ensemble: Literal["nve", "nvt", "npt"] = "nvt",
    dynamics: str | MolecularDynamics = "langevin",
    time_step: float | None = None, # fs
    total_time: float = 1000,  # fs
    temperature: float | Sequence | np.ndarray | None = 300.0, # K
    pressure: float | Sequence | np.ndarray | None = None, # eV/A^3
    ase_md_kwargs: dict | None = None,
    md_velocity_seed: int | None = None,
    zero_linear_momentum: bool = True,
    zero_angular_momentum: bool = True,
    traj_file: str | Path | None = None,
    traj_interval: int = 1,
    restart: bool = True,
):
    device = device or str(get_freer_device())

    print(f"Using device: {device}")

    calculator_kwargs = calculator_kwargs or {}

    if isinstance(calculator_name, MLIPEnum) and calculator_name in MLIPEnum:
        assert issubclass(calculator_name.value, Calculator)
        calc = calculator_name.value(**calculator_kwargs)
    elif (
        isinstance(calculator_name, str) and calculator_name in MLIPEnum._member_names_
    ):
        calc = MLIPEnum[calculator_name].value(**calculator_kwargs)
    else:
        raise ValueError(f"Invalid calculator: {calculator_name}")

    print(f"Using calculator: {calc}")

    dispersion_kwargs = dispersion_kwargs or {}

    dispersion_kwargs.update({"device": device})

    if dispersion is not None:
        disp_calc = TorchDFTD3Calculator(
            **dispersion_kwargs,
        )
        calc = SumCalculator([calc, disp_calc])

        print(f"Using dispersion: {dispersion}")

    atoms.calc = calc

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

    ase_md_kwargs = _get_ensemble_defaults(
        ensemble=ensemble,
        dynamics=dynamics,
        t_schedule=t_schedule,
        p_schedule=p_schedule,
        ase_md_kwargs=ase_md_kwargs,
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
        #  Note that until md_func is instantiated, isinstance(md_func,NPT) is False
        # ASE NPT implementation requires upper triangular cell
        u, _ = schur(atoms.get_cell(complete=True), output="complex")
        atoms.set_cell(u.real, scale_atoms=True)

    last_step = 0

    if traj_file is not None:
        traj_file = Path(traj_file)
        traj_file.parent.mkdir(parents=True, exist_ok=True)

        if restart and traj_file.exists():
            try:
                traj = read(traj_file, index=":")
                last_atoms = traj[-1]
                assert isinstance(last_atoms, Atoms)
                last_step = last_atoms.info.get("step", len(traj) * traj_interval)
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
                        rng=np.random.default_rng(seed=md_velocity_seed),
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
                    rng=np.random.default_rng(seed=md_velocity_seed),
                )

            if zero_linear_momentum:
                Stationary(atoms)
            if zero_angular_momentum:
                ZeroRotation(atoms)

    md_runner = md_class(
        atoms=atoms,
        timestep=time_step * units.fs,
        **ase_md_kwargs,
    )

    if traj_file is not None:
        md_runner.attach(traj.write, interval=traj_interval)

    with tqdm(total=n_steps) as pbar:

        def _callback(dyn: MolecularDynamics = md_runner) -> None:
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
        md_runner.run(steps=n_steps)
        end_time = datetime.now()

    if traj_file is not None:
        traj.close()

    return {
        "atoms": atoms,
        "runtime": end_time - start_time,
        "n_steps": n_steps,
    }
