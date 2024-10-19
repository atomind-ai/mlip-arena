"""
Define equation of state flows.

https://github.com/materialsvirtuallab/matcalc/blob/main/matcalc/eos.py
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from ase import Atoms
from ase.filters import *  # type: ignore
from ase.optimize import *  # type: ignore
from ase.optimize.optimize import Optimizer
from prefect import flow
from prefect.futures import wait
from prefect.runtime import flow_run, task_run
from pymatgen.analysis.eos import BirchMurnaghan

from mlip_arena.models.utils import MLIPEnum
from mlip_arena.tasks.optimize import run as OPT

if TYPE_CHECKING:
    from ase.filters import Filter


def generate_flow_run_name():
    flow_name = flow_run.flow_name

    parameters = flow_run.parameters

    atoms = parameters["atoms"]
    calculator_name = parameters["calculator_name"]

    return f"{flow_name}: {atoms.get_chemical_formula()} - {calculator_name}"


def generate_task_run_name():
    task_name = task_run.task_name

    parameters = task_run.parameters

    atoms = parameters["atoms"]
    calculator_name = parameters["calculator_name"]

    return f"{task_name}: {atoms.get_chemical_formula()} - {calculator_name}"


# https://docs.prefect.io/3.0/develop/write-tasks#custom-retry-behavior
# @task(task_run_name=generate_task_run_name)
@flow(flow_run_name=generate_flow_run_name, validate_parameters=False)
def fit(
    atoms: Atoms,
    calculator_name: str | MLIPEnum,
    calculator_kwargs: dict | None,
    device: str | None = None,
    optimizer: Optimizer | str = "BFGSLineSearch",  # type: ignore
    optimizer_kwargs: dict | None = None,
    filter: Filter | str | None = None,
    filter_kwargs: dict | None = None,
    criterion: dict | None = None,
    max_abs_strain: float = 0.1,
    npoints: int = 11,
):
    """
    Compute the equation of state (EOS) for the given atoms and calculator.

    Args:
        atoms: The input atoms.
        calculator_name: The name of the calculator to use.
        calculator_kwargs: Additional kwargs to pass to the calculator.
        device: The device to use.
        optimizer: The optimizer to use.
        optimizer_kwargs: Additional kwargs to pass to the optimizer.
        filter: The filter to use.
        filter_kwargs: Additional kwargs to pass to the filter.
        criterion: The criterion to use.
        max_abs_strain: The maximum absolute strain to use.
        npoints: The number of points to sample.

    Returns:
        A dictionary containing the EOS data and the bulk modulus.
    """
    first_relax = OPT(
        atoms=atoms,
        calculator_name=calculator_name,
        calculator_kwargs=calculator_kwargs,
        device=device,
        optimizer=optimizer,
        optimizer_kwargs=optimizer_kwargs,
        filter=filter,
        filter_kwargs=filter_kwargs,
        criterion=criterion,
    )

    relaxed = first_relax["atoms"]

    # p0 = relaxed.get_positions()
    c0 = relaxed.get_cell()

    factors = np.linspace(1 - max_abs_strain, 1 + max_abs_strain, npoints) ** (1 / 3)

    futures = []
    for f in factors:
        atoms = relaxed.copy()
        atoms.set_cell(c0 * f, scale_atoms=True)

        future = OPT.submit(
            atoms=atoms,
            calculator_name=calculator_name,
            calculator_kwargs=calculator_kwargs,
            device=device,
            optimizer=optimizer,
            optimizer_kwargs=optimizer_kwargs,
            filter=None,
            filter_kwargs=None,
            criterion=criterion,
        )

        futures.append(future)

    wait(futures)

    volumes = [
        f.result()["atoms"].get_volume()
        for f in futures
        if isinstance(f.result(), dict)
    ]
    energies = [
        f.result()["atoms"].get_potential_energy()
        for f in futures
        if isinstance(f.result(), dict)
    ]

    volumes, energies = map(
        list,
        zip(
            *sorted(zip(volumes, energies, strict=True), key=lambda i: i[0]),
            strict=True,
        ),
    )

    bm = BirchMurnaghan(volumes=volumes, energies=energies)
    bm.fit()

    return {
        "eos": {"volumes": volumes, "energies": energies},
        "K": bm.b0_GPa,
    }
