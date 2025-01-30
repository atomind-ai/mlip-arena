"""
Defines nudged elastic band (NEB) task

This module has been modified from MatCalc
https://github.com/materialsvirtuallab/matcalc/blob/main/src/matcalc/neb.py

https://github.com/materialsvirtuallab/matcalc/blob/main/LICENSE

BSD 3-Clause License

Copyright (c) 2023, Materials Virtual Lab

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this
   list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its
   contributors may be used to endorse or promote products derived from
   this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from ase import Atoms
from ase.calculators.calculator import BaseCalculator
from ase.filters import *  # type: ignore
from ase.mep.neb import NEB, NEBTools
from ase.optimize import *  # type: ignore
from ase.optimize.optimize import Optimizer
from ase.utils.forcecurve import fit_images
from prefect import task
from prefect.cache_policies import INPUTS, TASK_SOURCE
from prefect.runtime import task_run
from prefect.states import State

from mlip_arena.tasks.optimize import run as OPT
from mlip_arena.tasks.utils import logger, pformat
from pymatgen.io.ase import AseAtomsAdaptor

_valid_optimizers: dict[str, Optimizer] = {
    "MDMin": MDMin,
    "FIRE": FIRE,
    "FIRE2": FIRE2,
    "LBFGS": LBFGS,
    "LBFGSLineSearch": LBFGSLineSearch,
    "BFGS": BFGS,
    # "BFGSLineSearch": BFGSLineSearch, # NEB does not support BFGSLineSearch
    "QuasiNewton": QuasiNewton,
    "GPMin": GPMin,
    "CellAwareBFGS": CellAwareBFGS,
    "ODE12r": ODE12r,
}  # type: ignore


def _generate_task_run_name():
    task_name = task_run.task_name
    parameters = task_run.parameters

    if "images" in parameters:
        atoms = parameters["images"][0]
    elif "start" in parameters:
        atoms = parameters["start"]
    else:
        raise ValueError("No images or start atoms found in parameters")

    calculator_name = parameters["calculator"]

    return f"{task_name}: {atoms.get_chemical_formula()} - {calculator_name}"


@task(
    name="NEB from images",
    task_run_name=_generate_task_run_name,
    cache_policy=TASK_SOURCE + INPUTS,
)
def run(
    images: list[Atoms],
    calculator: BaseCalculator,
    optimizer: Optimizer | str = "MDMin",  # type: ignore
    optimizer_kwargs: dict | None = None,
    criterion: dict | None = None,
    interpolation: Literal["linear", "idpp"] = "idpp",
    climb: bool = True,
    traj_file: str | Path | None = None,
) -> dict[str, Any] | State:
    """Run the nudged elastic band (NEB) calculation.

    Args:
        images (list[Atoms]): The images.
        calculator_name (str | MLIPEnum): The calculator name.
        calculator_kwargs (dict, optional): The calculator kwargs. Defaults to None.
        dispersion (str, optional): The dispersion. Defaults to None.
        dispersion_kwargs (dict, optional): The dispersion kwargs. Defaults to None.
        device (str, optional): The device. Defaults to None.
        optimizer (Optimizer | str, optional): The optimizer. Defaults to "BFGSLineSearch".
        optimizer_kwargs (dict, optional): The optimizer kwargs. Defaults to None.
        criterion (dict, optional): The criterion. Defaults to None.
        interpolation (Literal['linear', 'idpp'], optional): The interpolation method. Defaults to "idpp".
        climb (bool, optional): Whether to use the climbing image. Defaults to True.
        traj_file (str | Path, optional): The trajectory file. Defaults to None.

    Returns:
        dict[str, Any] | State: The energy barrier.
    """

    images = [image.copy() for image in images]

    for image in images:
        assert isinstance(image, Atoms)
        image.calc = calculator

    neb = NEB(images, climb=climb, allow_shared_calculator=True)

    neb.interpolate(method=interpolation)

    if isinstance(optimizer, str):
        if optimizer not in _valid_optimizers:
            raise ValueError(f"Invalid optimizer: {optimizer}")
        optimizer = _valid_optimizers[optimizer]

    optimizer_kwargs = optimizer_kwargs or {}
    criterion = criterion or {}

    optimizer_instance = optimizer(neb, trajectory=traj_file, **optimizer_kwargs)  # type: ignore
    logger.info(f"Using optimizer: {optimizer_instance}")
    logger.info(pformat(optimizer_kwargs))
    logger.info(f"Criterion: {pformat(criterion)}")
    optimizer_instance.run(**criterion)

    neb_tool = NEBTools(neb.images)

    return {
        "barrier": neb_tool.get_barrier(),
        "images": neb.images,
        "forcefit": fit_images(neb.images),
    }


@task(
    name="NEB from endpoints",
    task_run_name=_generate_task_run_name,
    cache_policy=TASK_SOURCE + INPUTS,
)
def run_from_endpoints(
    start: Atoms,
    end: Atoms,
    n_images: int,
    calculator: BaseCalculator,
    optimizer: Optimizer | str = "BFGS",  # type: ignore
    optimizer_kwargs: dict | None = None,
    criterion: dict | None = None,
    relax_end_points: bool = True,
    interpolation: Literal["linear", "idpp"] = "idpp",
    climb: bool = True,
    traj_file: str | Path | None = None,
    cache_subtasks: bool = False,
) -> dict[str, Any] | State:
    """Run the nudged elastic band (NEB) calculation from end points.

    Args:
        start (Atoms): The start image.
        end (Atoms): The end image.
        n_images (int): The number of images.
        calculator_name (str | MLIPEnum): The calculator name.
        calculator_kwargs (dict, optional): The calculator kwargs. Defaults to None.
        dispersion (str, optional): The dispersion. Defaults to None.
        dispersion_kwargs (dict, optional): The dispersion kwargs. Defaults to None.
        device (str, optional): The device. Defaults to None.
        optimizer (Optimizer | str, optional): The optimizer. Defaults to "BFGSLineSearch".
        optimizer_kwargs (dict, optional): The optimizer kwargs. Defaults to None.
        criterion (dict, optional): The criterion. Defaults to None.
        interpolation (Literal['linear', 'idpp'], optional): The interpolation method. Defaults to "idpp".
        climb (bool, optional): Whether to use the climbing image. Defaults to True.
        traj_file (str | Path, optional): The trajectory file. Defaults to None.

    Returns:
        dict[str, Any] | State: The energy barrier.
    """

    if relax_end_points:
        relax = OPT.with_options(
            refresh_cache=not cache_subtasks,
        )(
            atoms=start.copy(),
            calculator=calculator,
            optimizer=optimizer,
            optimizer_kwargs=optimizer_kwargs,
            criterion=criterion,
        )
        start = relax["atoms"]

        relax = OPT.with_options(
            refresh_cache=not cache_subtasks,
        )(
            atoms=end.copy(),
            calculator=calculator,
            optimizer=optimizer,
            optimizer_kwargs=optimizer_kwargs,
            criterion=criterion,
        )
        end = relax["atoms"]

    path = (
        AseAtomsAdaptor()
        .get_structure(start)
        .interpolate(
            AseAtomsAdaptor().get_structure(end),
            nimages=n_images - 1,
            interpolate_lattices=False,
            pbc=False,
            autosort_tol=0.5,
        )
    )

    images = [s.to_ase_atoms(msonable=False) for s in path]

    return run.with_options(
        refresh_cache=not cache_subtasks,
    )(
        images,
        calculator=calculator,
        optimizer=optimizer,
        optimizer_kwargs=optimizer_kwargs,
        criterion=criterion,
        interpolation=interpolation,
        climb=climb,
        traj_file=traj_file,
    )
