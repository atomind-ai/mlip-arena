"""
Define equation of state task.

https://github.com/materialsvirtuallab/matcalc/blob/main/matcalc/eos.py
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
from prefect import task
from prefect.cache_policies import INPUTS, TASK_SOURCE
from prefect.futures import wait
from prefect.runtime import task_run
from prefect.states import State

from ase import Atoms
from ase.filters import *  # type: ignore
from ase.optimize import *  # type: ignore
from ase.optimize.optimize import Optimizer
from mlip_arena.models import MLIPEnum
from mlip_arena.tasks.optimize import run as OPT
from pymatgen.analysis.eos import BirchMurnaghan

if TYPE_CHECKING:
    from ase.filters import Filter


def _generate_task_run_name():
    task_name = task_run.task_name
    parameters = task_run.parameters

    atoms = parameters["atoms"]
    calculator_name = parameters["calculator_name"]

    return f"{task_name}: {atoms.get_chemical_formula()} - {calculator_name}"


@task(
    name="EOS",
    task_run_name=_generate_task_run_name,
    cache_policy=TASK_SOURCE + INPUTS
    # cache_key_fn=task_input_hash,
)
def run(
    atoms: Atoms,
    calculator_name: str | MLIPEnum,
    calculator_kwargs: dict | None = None,
    device: str | None = None,
    optimizer: Optimizer | str = "BFGSLineSearch",  # type: ignore
    optimizer_kwargs: dict | None = None,
    filter: Filter | str | None = "FrechetCell",  # type: ignore
    filter_kwargs: dict | None = None,
    criterion: dict | None = None,
    max_abs_strain: float = 0.1,
    npoints: int = 11,
    concurrent: bool = True,
    persist_opt: bool = True,
    cache_opt: bool = True,
) -> dict[str, Any] | State:
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
        concurrent: Whether to relax multiple structures concurrently.
        cache_opt: Whether to cache the intermediate optimization results.

    Returns:
        A dictionary containing the EOS data, bulk modulus, equilibrium volume, and equilibrium energy if successful. Otherwise, a prefect state object.
    """

    OPT_ = OPT.with_options(
        refresh_cache=not cache_opt,
        persist_result=persist_opt,     
    )

    state = OPT_(
        atoms=atoms,
        calculator_name=calculator_name,
        calculator_kwargs=calculator_kwargs,
        device=device,
        optimizer=optimizer,
        optimizer_kwargs=optimizer_kwargs,
        filter=filter,
        filter_kwargs=filter_kwargs,
        criterion=criterion,
        return_state=True,
    )

    if state.is_failed():
        return state
    elif state.is_completed() and state.name in ["Completed", "Cached"]:
        first_relax = state.result(raise_on_failure=False)
    elif state.is_completed() and state.name in ["Rollback"]:
        first_relax = state.result(raise_on_failure=False)
    
    assert isinstance(first_relax, dict)
    relaxed = first_relax["atoms"]

    # p0 = relaxed.get_positions()
    c0 = relaxed.get_cell()

    factors = np.linspace(1 - max_abs_strain, 1 + max_abs_strain, npoints) ** (1 / 3)

    if concurrent:
        futures = []
        for f in factors:
            atoms = relaxed.copy()
            atoms.set_cell(c0 * f, scale_atoms=True)

            future = OPT_.submit(
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

        results = [
            f.result(raise_on_failure=False)
            for f in futures
            if future.state.is_completed()
        ]
    else:
        states = []
        for f in factors:
            atoms = relaxed.copy()
            atoms.set_cell(c0 * f, scale_atoms=True)

            state = OPT_(
                atoms=atoms,
                calculator_name=calculator_name,
                calculator_kwargs=calculator_kwargs,
                device=device,
                optimizer=optimizer,
                optimizer_kwargs=optimizer_kwargs,
                filter=None,
                filter_kwargs=None,
                criterion=criterion,
                return_state=True,
            )
            states.append(state)
        results = [
            s.result(raise_on_failure=False) for s in states if s.is_completed()
        ]

    volumes = [f["atoms"].get_volume() for f in results]
    energies = [f["atoms"].get_potential_energy() for f in results]

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
        "atoms": relaxed,
        "calculator_name": calculator_name,
        "eos": {"volumes": volumes, "energies": energies},
        "K": bm.b0_GPa,
        "b0": bm.b0,
        "b1": bm.b1,
        "e0": bm.e0,
        "v0": bm.v0,
    }
