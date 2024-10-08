"""
Define equation of state flows.

https://github.com/materialsvirtuallab/matcalc/blob/main/matcalc/eos.py
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

import numpy as np
from ase import Atoms
from ase.filters import *  # type: ignore
from ase.optimize import *  # type: ignore
from ase.optimize.optimize import Optimizer
from prefect import flow
from pymatgen.analysis.eos import BirchMurnaghan

from prefect.futures import wait

from mlip_arena.models.utils import MLIPEnum
from mlip_arena.tasks.optimize import run as OPT

if TYPE_CHECKING:
    from ase.filters import Filter


@flow
def fit(
    atoms: Atoms,
    calculator_name: str | MLIPEnum,
    calculator_kwargs: dict | None,
    device: str | None = None,
    optimizer: Optimizer | str = BFGSLineSearch, # type: ignore
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
    result = OPT(
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

    relaxed = result["atoms"]

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

    volumes = [f.result()["atoms"].get_volume() for f in futures]
    energies = [f.result()["atoms"].get_potential_energy() for f in futures]

    bm = BirchMurnaghan(volumes=volumes, energies=energies)
    bm.fit()

    volumes, energies = map(list, zip(*sorted(zip(volumes, energies, strict=False), key=lambda i: i[0]), strict=False))

    return {
        "eos": {"volumes": volumes, "energies": energies},
        "K": bm.b0_GPa,
    }
