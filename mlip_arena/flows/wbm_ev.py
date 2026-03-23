from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from ase.db import connect
from prefect import flow, task
from prefect.cache_policies import INPUTS, TASK_SOURCE
from prefect.futures import wait
from prefect.runtime import task_run

from mlip_arena.models import MLIPEnum
from mlip_arena.tasks.utils import get_calculator

if TYPE_CHECKING:
    from ase import Atoms


@task
def load_wbm_structures(db_path: Path):
    """
    Load the WBM structures from an ASE database file.
    """
    with connect(db_path) as db:
        for row in db.select():
            yield row.toatoms(add_additional_information=True)


@task(
    name="E-V Scan",
    task_run_name=lambda: f"{task_run.task_name}: {task_run.parameters['atoms'].get_chemical_formula()} - {task_run.parameters['model_name']}",
    cache_policy=TASK_SOURCE + INPUTS,
)
def ev_scan_task(atoms: Atoms, model_name: str, out_dir: Path):
    calculator = get_calculator(model_name)

    wbm_id = atoms.info["key_value_pairs"]["wbm_id"]

    c0 = atoms.get_cell()
    max_abs_strain = 0.2
    npoints = 21
    volumes = []
    energies = []
    for uniaxial_strain in np.linspace(-max_abs_strain, max_abs_strain, npoints):
        cloned = atoms.copy()
        scale_factor = uniaxial_strain + 1
        cloned.set_cell(c0 * scale_factor, scale_atoms=True)
        cloned.calc = calculator
        volumes.append(cloned.get_volume())
        energies.append(cloned.get_potential_energy())

    data = {
        "method": model_name,
        "id": wbm_id,
        "eos": {"volumes": volumes, "energies": energies},
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    fpath = out_dir / f"{wbm_id}.json"

    df = pd.DataFrame([data])
    df.to_json(fpath)

    return df


@flow
def wbm_ev_scan(
    model: str,
    run_dir: Path | None = None,
    db_path: str = "benchmarks/wbm_structures.db",
):
    from ase.calculators.calculator import BaseCalculator

    if isinstance(model, BaseCalculator):
        model_name = model.__class__.__name__
    elif isinstance(model, str) and hasattr(MLIPEnum, model):
        model_name = model
    else:
        raise ValueError(f"Unsupported model: {model}")

    out_dir = run_dir if run_dir is not None else Path.cwd() / "wbm_ev" / model_name

    db_file = Path(db_path)
    if not db_file.exists():
        raise FileNotFoundError(f"Database not found at {db_file.absolute()}")

    futures = []
    for atoms in load_wbm_structures(db_file):
        future = ev_scan_task.with_options(refresh_cache=True).submit(
            atoms, model_name, out_dir
        )
        futures.append(future)

    wait(futures)
    return [f.result(raise_on_failure=False) for f in futures]
