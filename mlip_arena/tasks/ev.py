"""Energy-volume (E-V) curve calculation tasks."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from ase.calculators.calculator import BaseCalculator
from prefect import task
from prefect.runtime import task_run

from mlip_arena.models import MLIPEnum
from mlip_arena.tasks.utils import ARENA_TASK_CACHE_POLICY, get_calculator

if TYPE_CHECKING:
    from ase import Atoms


@task(
    name="E-V Scan",
    task_run_name=lambda: (
        f"{task_run.task_name}: {task_run.parameters['atoms'].get_chemical_formula()} - {task_run.parameters.get('model', 'Unknown')}"
    ),
    cache_policy=ARENA_TASK_CACHE_POLICY,
)
def run(
    atoms: Atoms,
    model: str | BaseCalculator,
    max_abs_strain: float = 0.2,
    npoints: int = 21,
):
    """Perform an energy-volume (E-V) scan by applying uniform strain to the
    cell.

    Args:
        atoms (Atoms): ASE Atoms object.
        model (str | BaseCalculator): Model name or ASE calculator.
        max_abs_strain (float, optional): Maximum absolute strain to apply. Defaults to 0.2.
        npoints (int, optional): Number of points in the E-V curve. Defaults to 21.

    Returns:
        pd.DataFrame: DataFrame containing model name, structure ID, and E-V data.
    """
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
