import os, glob
from pathlib import Path
from ase.io import read, write
from ase import units
from ase import Atoms, units
from ase.calculators.calculator import Calculator
from ase.data import chemical_symbols
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
from dask.distributed import Client
from dask_jobqueue import SLURMCluster
from jobflow import Maker
from prefect import flow, task
from prefect.tasks import task_input_hash
from prefect_dask import DaskTaskRunner
from pymatgen.io.ase import AseAtomsAdaptor
from scipy.interpolate import interp1d
from scipy.linalg import schur

from mlip_arena.models import MLIPCalculator
from mlip_arena.models.utils import EXTMLIPEnum, MLIPMap, external_ase_calculator
from torch_dftd.torch_dftd3_calculator import TorchDFTD3Calculator
from mp_api.client import MPRester

from fireworks import LaunchPad
from atomate2.vasp.flows.core import RelaxBandStructureMaker
from atomate2.vasp.flows.mp import MPGGADoubleRelaxStaticMaker
from atomate2.vasp.powerups import add_metadata_to_flow
from atomate2.forcefields.md import (
    CHGNetMDMaker,
    GAPMDMaker,
    M3GNetMDMaker,
    MACEMDMaker,
    NequipMDMaker,
)
from atomate2.forcefields.utils import MLFF
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.transformations.advanced_transformations import CubicSupercellTransformation
from jobflow.managers.fireworks import flow_to_workflow
from jobflow import run_locally, SETTINGS
from tqdm.auto import tqdm

from datetime import timedelta, datetime
from typing import Literal, Sequence, Tuple

import numpy as np
import torch
from pymatgen.core.structure import Structure

from ase.calculators.mixing import SumCalculator
from scipy.interpolate import interp1d

from ase.io.trajectory import Trajectory


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
        pressure: float | Sequence | np.ndarray | None = None
        ) -> Tuple[np.ndarray, np.ndarray]:
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
        p_schedule = interp1d(
            np.arange(n_steps + 1), pressure, kind="linear"
        )
        assert isinstance(p_schedule, np.ndarray)
    else:
        p_schedule = np.full(n_steps + 1, pressure)
    
    return t_schedule, p_schedule

def _get_ensemble_defaults(
        ensemble: Literal["nve", "nvt", "npt"],
        dynamics: str | MolecularDynamics,
        t_schedule: np.ndarray,
        p_schedule: np.ndarray,
        ase_md_kwargs: dict | None = None) -> dict:
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
        ase_md_kwargs["externalstress"] = p_schedule[0] * 1e3 * units.bar

    if isinstance(dynamics, str) and dynamics.lower() == "langevin":
        ase_md_kwargs["friction"] = ase_md_kwargs.get(
            "friction",
            10.0 * 1e-3 / units.fs,  # Same default as in VASP: 10 ps^-1
        )
        
    return ase_md_kwargs

