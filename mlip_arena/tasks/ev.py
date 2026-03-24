from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from ase.calculators.calculator import BaseCalculator
from prefect import task
from prefect.cache_policies import INPUTS, TASK_SOURCE
from prefect.runtime import task_run

from mlip_arena.models import MLIPEnum
from mlip_arena.tasks.utils import get_calculator

if TYPE_CHECKING:
    from ase import Atoms


@task(
    name="E-V Scan",
    task_run_name=lambda: f"{task_run.task_name}: {task_run.parameters['atoms'].get_chemical_formula()} - {task_run.parameters.get('model_name', 'Unknown')}",
    cache_policy=TASK_SOURCE + INPUTS,
)
def run(
    atoms: Atoms,
    model: str | BaseCalculator,
    max_abs_strain: float = 0.2,
    npoints: int = 21,
):
    if isinstance(model, BaseCalculator):
        model_name = model.__class__.__name__
    elif isinstance(model, str) and hasattr(MLIPEnum, model):
        model_name = model
    else:
        raise ValueError(f"Unsupported model: {model}")

    wbm_id = atoms.info["key_value_pairs"]["wbm_id"]

    c0 = atoms.get_cell()
    volumes = []
    energies = []
    for uniaxial_strain in np.linspace(-max_abs_strain, max_abs_strain, npoints):
        cloned = atoms.copy()
        scale_factor = uniaxial_strain + 1
        cloned.set_cell(c0 * scale_factor, scale_atoms=True)
        cloned.calc = get_calculator(model)
        volumes.append(cloned.get_volume())
        energies.append(cloned.get_potential_energy())

    data = {
        "method": model_name,
        "id": wbm_id,
        "eos": {"volumes": volumes, "energies": energies},
    }

    return pd.DataFrame([data])
