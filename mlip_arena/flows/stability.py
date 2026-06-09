"""Workflow for executing stability MD simulations."""

from __future__ import annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING

from ase import units
from ase.calculators.calculator import BaseCalculator
from dotenv import load_dotenv
from prefect import flow, task
from prefect.cache_policies import INPUTS, TASK_SOURCE
from prefect.futures import wait

from mlip_arena.models import MLIPEnum, REGISTRY
from mlip_arena.tasks import MD
from mlip_arena.tasks.stability.data import get_atoms_from_db
from mlip_arena.tasks.utils import get_calculator

if TYPE_CHECKING:
    from ase import Atoms

load_dotenv()

HF_TOKEN = os.environ.get("HF_TOKEN", None)


@task(cache_policy=TASK_SOURCE + INPUTS)
def nvt_heat_one(atoms: Atoms, model: MLIPEnum | BaseCalculator | str, run_dir: Path):
    """Run a 10 ps NVT MD simulation with linear heating schedule."""
    calculator = (
        get_calculator(
            model if isinstance(model, str) else model.name,
            calculator_kwargs=None,
        )
        if isinstance(model, MLIPEnum | str)
        else model
    )

    model_name = (
        model if isinstance(model, str) else (model.name if isinstance(model, MLIPEnum) else model.__class__.__name__)
    )

    return MD.with_options(refresh_cache=True)(
        atoms=atoms,
        calculator=calculator,
        ensemble="nvt",
        dynamics="nose-hoover",
        time_step=None,
        dynamics_kwargs=dict(
            ttime=25 * units.fs,
        ),
        total_time=1e4,  # 10 ps
        temperature=[300, 3000],
        pressure=None,
        traj_file=run_dir
        / f"{model_name}_{atoms.info.get('material_id', 'random')}_{atoms.get_chemical_formula()}_nvt.traj",
        traj_interval=10,
    )


@task(cache_policy=TASK_SOURCE + INPUTS)
def npt_compress_one(atoms: Atoms, model: MLIPEnum | BaseCalculator | str, run_dir: Path):
    """Run a 10 ps NPT MD simulation with linear pressure ramp."""
    calculator = (
        get_calculator(
            model if isinstance(model, str) else model.name,
            calculator_kwargs=None,
        )
        if isinstance(model, MLIPEnum | str)
        else model
    )

    model_name = (
        model if isinstance(model, str) else (model.name if isinstance(model, MLIPEnum) else model.__class__.__name__)
    )

    return MD.with_options(timeout_seconds=600, retries=2, refresh_cache=True)(
        atoms=atoms,
        calculator=calculator,
        ensemble="npt",
        dynamics="nose-hoover",
        time_step=None,
        dynamics_kwargs=dict(ttime=25 * units.fs, pfactor=((75 * units.fs) ** 2) * 1e2 * units.GPa),
        total_time=1e4,  # 10 ps
        temperature=[300, 3000],
        pressure=[0, 5e2 * units.GPa],  # 500 GPa
        traj_file=run_dir
        / f"{model_name}_{atoms.info.get('material_id', 'random')}_{atoms.get_chemical_formula()}_npt.traj",
        traj_interval=10,
    )


@flow
def heating(
    model: MLIPEnum | BaseCalculator | str,
    run_dir: Path | None = None,
    hf_token: str | None = HF_TOKEN,
):
    """Prefect flow to run NVT heating tasks for many database structures."""
    if isinstance(model, BaseCalculator):
        model_name = model.__class__.__name__
    elif isinstance(model, MLIPEnum):
        model_name = model.name
    elif isinstance(model, str) and hasattr(MLIPEnum, model):
        model_name = model
    else:
        model_name = str(model)

    family = REGISTRY[model_name]["family"] if hasattr(MLIPEnum, model_name) else "custom"
    out_dir = run_dir if run_dir is not None else Path.cwd() / "stability" / family
    out_dir.mkdir(parents=True, exist_ok=True)

    futures = []
    # To download the database automatically, `huggingface_hub login` or provide HF_TOKEN
    for i, atoms in enumerate(get_atoms_from_db("random-mixture.db", hf_token=hf_token, force_download=False)):
        if i >= 200:
            break
        future = nvt_heat_one.with_options(timeout_seconds=600, retries=2, refresh_cache=False).submit(
            atoms.copy(), model, out_dir
        )
        futures.append(future)

    wait(futures)

    return [f.result(timeout=None, raise_on_failure=False) for f in futures if f.state.is_completed()]


@flow
def compression(
    model: MLIPEnum | BaseCalculator | str,
    run_dir: Path | None = None,
    hf_token: str | None = HF_TOKEN,
):
    """Prefect flow to run NPT compression tasks for many database structures."""
    if isinstance(model, BaseCalculator):
        model_name = model.__class__.__name__
    elif isinstance(model, MLIPEnum):
        model_name = model.name
    elif isinstance(model, str) and hasattr(MLIPEnum, model):
        model_name = model
    else:
        model_name = str(model)

    family = REGISTRY[model_name]["family"] if hasattr(MLIPEnum, model_name) else "custom"
    out_dir = run_dir if run_dir is not None else Path.cwd() / "stability" / family
    out_dir.mkdir(parents=True, exist_ok=True)

    futures = []
    # To download the database automatically, `huggingface_hub login` or provide HF_TOKEN
    for i, atoms in enumerate(get_atoms_from_db("random-mixture.db", hf_token=hf_token, force_download=False)):
        if i >= 200:
            break
        future = npt_compress_one.with_options(timeout_seconds=600, retries=2, refresh_cache=False).submit(
            atoms.copy(), model, out_dir
        )
        futures.append(future)

    wait(futures)

    return [f.result(timeout=None, raise_on_failure=False) for f in futures if f.state.is_completed()]
