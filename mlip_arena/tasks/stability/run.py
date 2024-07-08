from __future__ import annotations

import datetime
from datetime import datetime
from pathlib import Path
from typing import Literal, Sequence

import numpy as np
import torch
from ase import Atoms, units
from ase.calculators.mixing import SumCalculator
from ase.io import read
from ase.io.trajectory import Trajectory
from ase.md.md import MolecularDynamics
from ase.md.npt import NPT
from ase.md.velocitydistribution import (
    MaxwellBoltzmannDistribution,
    Stationary,
    ZeroRotation,
)
from scipy.linalg import schur
from torch_dftd.torch_dftd3_calculator import TorchDFTD3Calculator
from tqdm.auto import tqdm

from mlip_arena.models.utils import EXTMLIPEnum, MLIPMap, external_ase_calculator
from mlip_arena.tasks.utils import (
    _get_ensemble_defaults,
    _get_ensemble_schedule,
    _preset_dynamics,
    _valid_dynamics,
)


def md(
    atoms: Atoms,
    calculator_name: str | EXTMLIPEnum,
    calculator_kwargs: dict | None,
    dispersion: str | None = None,
    dispersion_kwargs: dict | None = None,
    device: str | None = None,
    ensemble: Literal["nve", "nvt", "npt"] = "nvt",
    dynamics: str | MolecularDynamics = "langevin",
    time_step: float | None = None,
    total_time: float = 1000,
    temperature: float | Sequence | np.ndarray | None = 300.0,
    pressure: float | Sequence | np.ndarray | None = None,
    ase_md_kwargs: dict | None = None,
    mb_velocity_seed: int | None = None,
    zero_linear_momentum: bool = True,
    zero_angular_momentum: bool = True,
    traj_file: str | Path | None = None,
    traj_interval: int = 1,
    # ttime: float = 25 * units.fs,
    # pfactor: float = (75 * units.fs) ** 1 * units.GPa,
    # mask: np.ndarray | list[int] | None = None,
    # traceless: float = 1.0,
    restart: bool = True,
    # interval: int = 500,
    # device: str | None = None,
    # dtype: str = "float64",
):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Using device: {device}")

    calculator_kwargs = calculator_kwargs or {}

    if isinstance(calculator_name, EXTMLIPEnum) and calculator_name in EXTMLIPEnum:
        calc = external_ase_calculator(calculator_name, **calculator_kwargs)
    elif calculator_name in MLIPMap:
        calc = MLIPMap[calculator_name](**calculator_kwargs)

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

        if ensemble == "nve" and dynamics is None:
            dynamics = "velocityverlet"
        md_class = _preset_dynamics[f"{ensemble}_{dynamics}"]
    elif issubclass(dynamics, MolecularDynamics):
        md_class = dynamics

    if md_class is NPT:
        #  Note that until md_func is instantiated, isinstance(md_func,NPT) is False
        # ASE NPT implementation requires upper triangular cell
        u, _ = schur(atoms.get_cell(complete=True), output="complex")
        atoms.set_cell(u.real, scale_atoms=True)

    last_step = 0

    if traj_file is not None:
        traj_file = Path(traj_file)

        if restart and traj_file.exists():
            traj = read(traj_file, index=":")
            last_step = len(traj)
            n_steps -= len(traj)
            last_atoms = traj[-1]
            traj = Trajectory(traj_file, "a", atoms)
            atoms.set_positions(last_atoms.get_positions())
            atoms.set_momenta(last_atoms.get_momenta())
        else:
            traj = Trajectory(traj_file, "w", atoms)

            if not np.isnan(t_schedule).any():
                MaxwellBoltzmannDistribution(
                    atoms=atoms,
                    temperature_K=t_schedule[last_step],
                    rng=np.random.default_rng(seed=mb_velocity_seed),
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
            if ensemble == "nve":
                return
            dyn.set_temperature(temperature_K=t_schedule[last_step + dyn.nsteps])
            if ensemble == "nvt":
                return
            dyn.set_stress(p_schedule[last_step + dyn.nsteps] * 1e3 * units.bar)
            pbar.update()

        md_runner.attach(_callback, interval=1)

        start_time = datetime.now()
        md_runner.run(steps=n_steps)
        end_time = datetime.now()

        traj.close()

    return {"md_runtime": end_time - start_time}
