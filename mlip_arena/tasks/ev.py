"""Energy-volume (E-V) curve calculation tasks."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from ase.calculators.calculator import BaseCalculator
from prefect import task
from prefect.runtime import task_run

from mlip_arena.models import MLIPEnum
from mlip_arena.tasks.utils import ARENA_TASK_CACHE_POLICY, get_calculator, resolve_calculator_name

if TYPE_CHECKING:
    from ase import Atoms


def _generate_task_run_name():
    task_name = task_run.task_name
    parameters = task_run.parameters
    atoms = parameters["atoms"]
    calculator_name = resolve_calculator_name(parameters.get("calculator"))
    return f"{task_name}: {atoms.get_chemical_formula()} - {calculator_name}"


@task(
    name="E-V Scan",
    task_run_name=_generate_task_run_name,
    cache_policy=ARENA_TASK_CACHE_POLICY,
)
def run(
    atoms: Atoms,
    calculator: str | MLIPEnum | BaseCalculator | None = None,
    calculator_kwargs: dict | None = None,
    dispersion: bool = False,
    dispersion_kwargs: dict | None = None,
    max_abs_strain: float = 0.2,
    npoints: int = 21,
):
    """Perform an energy-volume (E-V) scan by applying uniform strain to the
    cell.

    Args:
        atoms (Atoms): ASE Atoms object.
        calculator (str | MLIPEnum | BaseCalculator, optional): The model or ASE calculator.
        calculator_kwargs (dict, optional): Keyword arguments to pass to the calculator. Defaults to None.
        dispersion (bool, optional): Whether to use dispersion correction. Defaults to False.
        dispersion_kwargs (dict, optional): Keyword arguments for dispersion correction.
        max_abs_strain (float, optional): Maximum absolute strain to apply. Defaults to 0.2.
        npoints (int, optional): Number of points in the E-V curve. Defaults to 21.

    Returns:
        pd.DataFrame: DataFrame containing model name, structure ID, and E-V data.
    """
    model_name = resolve_calculator_name(calculator)

    wbm_id = atoms.info["key_value_pairs"]["wbm_id"]

    c0 = atoms.get_cell()
    volumes = []
    energies = []

    # Instantiate calculator once for the scan to be fast
    calculator_obj = get_calculator(calculator, calculator_kwargs, dispersion, dispersion_kwargs)

    for uniaxial_strain in np.linspace(-max_abs_strain, max_abs_strain, npoints):
        cloned = atoms.copy()
        scale_factor = uniaxial_strain + 1
        cloned.set_cell(c0 * scale_factor, scale_atoms=True)
        cloned.calc = calculator_obj
        volumes.append(cloned.get_volume())
        energies.append(cloned.get_potential_energy())
        cloned.calc = None

    data = {
        "method": model_name,
        "id": wbm_id,
        "eos": {"volumes": volumes, "energies": energies},
    }

    return pd.DataFrame([data])
