"""
Define structure optimization tasks.
"""

from __future__ import annotations

from prefect import task
from prefect.cache_policies import INPUTS, TASK_SOURCE
from prefect.runtime import task_run

from ase import Atoms
from ase.constraints import FixSymmetry
from ase.filters import *  # type: ignore
from ase.filters import Filter
from ase.optimize import *  # type: ignore
from ase.optimize.optimize import Optimizer
from mlip_arena.models import MLIPEnum
from mlip_arena.tasks.utils import get_calculator

_valid_filters: dict[str, Filter] = {
    "Filter": Filter,
    "UnitCell": UnitCellFilter,
    "ExpCell": ExpCellFilter,
    "Strain": StrainFilter,
    "FrechetCell": FrechetCellFilter,
}  # type: ignore

_valid_optimizers: dict[str, Optimizer] = {
    "MDMin": MDMin,
    "FIRE": FIRE,
    "FIRE2": FIRE2,
    "LBFGS": LBFGS,
    "LBFGSLineSearch": LBFGSLineSearch,
    "BFGS": BFGS,
    "BFGSLineSearch": BFGSLineSearch,
    "QuasiNewton": QuasiNewton,
    "GPMin": GPMin,
    "CellAwareBFGS": CellAwareBFGS,
    "ODE12r": ODE12r,
}  # type: ignore


def _generate_task_run_name():
    task_name = task_run.task_name
    parameters = task_run.parameters

    atoms = parameters["atoms"]
    calculator_name = parameters["calculator_name"]

    return f"{task_name}: {atoms.get_chemical_formula()} - {calculator_name}"


@task(
    name="OPT",
    task_run_name=_generate_task_run_name,
    cache_policy=TASK_SOURCE + INPUTS
    # cache_key_fn=task_input_hash,
    # cache_expiration=timedelta(days=1)
)
def run(
    atoms: Atoms,
    calculator_name: str | MLIPEnum,
    calculator_kwargs: dict | None = None,
    dispersion: str | None = None,
    dispersion_kwargs: dict | None = None,
    device: str | None = None,
    optimizer: Optimizer | str = BFGSLineSearch,
    optimizer_kwargs: dict | None = None,
    filter: Filter | str | None = None,
    filter_kwargs: dict | None = None,
    criterion: dict | None = None,
    symmetry: bool = False,
):
    
    atoms.calc = get_calculator(
        calculator_name=calculator_name,
        calculator_kwargs=calculator_kwargs,
        dispersion=dispersion,
        dispersion_kwargs=dispersion_kwargs,
        device=device,
    )

    if isinstance(filter, str):
        if filter not in _valid_filters:
            raise ValueError(f"Invalid filter: {filter}")
        filter = _valid_filters[filter]

    if isinstance(optimizer, str):
        if optimizer not in _valid_optimizers:
            raise ValueError(f"Invalid optimizer: {optimizer}")
        optimizer = _valid_optimizers[optimizer]

    filter_kwargs = filter_kwargs or {}
    optimizer_kwargs = optimizer_kwargs or {}
    criterion = criterion or {}

    if symmetry:
        atoms.set_constraint(FixSymmetry(atoms))

    if isinstance(filter, type) and issubclass(filter, Filter):
        filter_instance = filter(atoms, **filter_kwargs)
        print(f"Using filter: {filter_instance}")

        optimizer_instance = optimizer(atoms, **optimizer_kwargs)
        print(f"Using optimizer: {optimizer_instance}")

        optimizer_instance.run(**criterion)

    elif filter is None:
        optimizer_instance = optimizer(atoms, **optimizer_kwargs)
        print(f"Using optimizer: {optimizer_instance}")
        optimizer_instance.run(**criterion)

    return {
        "atoms": atoms,
    }
