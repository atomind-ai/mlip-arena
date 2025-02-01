"""
Defines the tasks for computing the elastic tensor.

This module has been modified from MatCalc
https://github.com/materialsvirtuallab/matcalc/blob/main/src/matcalc/elasticity.py

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

from typing import TYPE_CHECKING, Any

import numpy as np
from ase import Atoms
from ase.calculators.calculator import BaseCalculator
from ase.optimize.optimize import Optimizer
from numpy.typing import ArrayLike
from prefect import task
from prefect.cache_policies import INPUTS, TASK_SOURCE
from prefect.runtime import task_run
from prefect.states import State

from mlip_arena.tasks.optimize import run as OPT
from pymatgen.analysis.elasticity import DeformedStructureSet, ElasticTensor, Strain
from pymatgen.analysis.elasticity.elastic import get_strain_state_dict
from pymatgen.io.ase import AseAtomsAdaptor

if TYPE_CHECKING:
    from ase.filters import Filter


def _generate_task_run_name():
    task_name = task_run.task_name
    parameters = task_run.parameters

    atoms = parameters["atoms"]
    calculator_name = parameters["calculator"]

    return f"{task_name}: {atoms.get_chemical_formula()} - {calculator_name}"


@task(
    name="Elasticity",
    task_run_name=_generate_task_run_name,
    cache_policy=TASK_SOURCE + INPUTS,
    # cache_key_fn=task_input_hash,
)
def run(
    atoms: Atoms,
    calculator: BaseCalculator,
    optimizer: Optimizer | str = "BFGSLineSearch",  # type: ignore
    optimizer_kwargs: dict | None = None,
    filter: Filter | str | None = "FrechetCell",  # type: ignore
    filter_kwargs: dict | None = None,
    criterion: dict | None = None,
    normal_strains: list[float] | np.ndarray | None = np.linspace(-0.01, 0.01, 4),
    shear_strains: list[float] | np.ndarray | None = np.linspace(-0.06, 0.06, 4),
    persist_opt: bool = True,
    cache_opt: bool = False,
) -> dict[str, Any] | State:
    """
    Compute the elastic tensor for the given structure and calculator.

    Args:
        atoms (Atoms): The input structure.
        calculator (BaseCalculator): The calculator.
        optimizer (Optimizer | str, optional): The optimizer. Defaults to "BFGSLineSearch".
        optimizer_kwargs (dict, optional): The optimizer kwargs. Defaults to None.
        filter (Filter | str, optional): The filter. Defaults to "FrechetCell".
        filter_kwargs (dict, optional): The filter kwargs. Defaults to None.
        criterion (dict, optional): The criterion. Defaults to None.
        normal_strains (list[float] | np.ndarray, optional): The normal strains. Defaults to np.linspace(-0.01, 0.01, 4).
        shear_strains (list[float] | np.ndarray, optional): The shear strains. Defaults to np.linspace(-0.06, 0.06, 4).
        concurrent (bool, optional): Whether to run concurrently. Defaults to True.
        persist_opt (bool, optional): Whether to persist the optimizer results. Defaults to True.
        cache_opt (bool, optional): Whether to cache the optimizer results. Defaults to True.

    Returns:
        dict[str, Any] | State: The elastic tensor.
    """

    atoms = atoms.copy()

    OPT_ = OPT.with_options(
        refresh_cache=not cache_opt,
        persist_result=persist_opt,
    )

    first_relax = OPT_(
        atoms=atoms,
        calculator=calculator,
        optimizer=optimizer,
        optimizer_kwargs=optimizer_kwargs,
        filter=filter,
        filter_kwargs=filter_kwargs,
        criterion=criterion,
        return_state=True,
    )

    if first_relax.is_failed():
        return first_relax

    result = first_relax.result(raise_on_failure=False)

    assert isinstance(result, dict)
    relaxed = result["atoms"]

    if isinstance(normal_strains, np.ndarray):
        normal_strains = normal_strains.tolist()
    if isinstance(shear_strains, np.ndarray):
        shear_strains = shear_strains.tolist()

    assert isinstance(relaxed, Atoms)
    assert isinstance(normal_strains, list)
    assert isinstance(shear_strains, list)

    structure = AseAtomsAdaptor.get_structure(relaxed)  # type: ignore

    deformed_structure_set = DeformedStructureSet(
        structure,
        normal_strains,
        shear_strains,
    )

    stresses = []
    for deformed_structure in deformed_structure_set:
        atoms = deformed_structure.to_ase_atoms()
        atoms.calc = relaxed.calc
        stresses.append(atoms.get_stress(voigt=False))

    strains = [
        Strain.from_deformation(deformation)
        for deformation in deformed_structure_set.deformations
    ]

    fit = fit_elastic_tensor(
        strains, stresses, eq_stress=relaxed.get_stress(voigt=False)
    )

    return {
        "elastic_tensor": fit["elastic_tensor"],
        "residuals_sum": fit["residuals_sum"],
    }


@task
def fit_elastic_tensor(
    strains: ArrayLike,
    stresses: ArrayLike,
    eq_stress: ArrayLike | None = None,
    tolerance: float = 1e-7,
):
    """
    Compute the elastic tensor from the given strains and stresses.

    Args:
        strains (ArrayLike): The strains.
        stresses (ArrayLike): The stresses.
        tolerance (float, optional): The tolerance. Defaults to 1e-7.

    Returns:
        ElasticTensor: The elastic tensor.
    """

    strain_states = [tuple(ss) for ss in np.eye(6)]
    ss_dict = get_strain_state_dict(
        strains,
        stresses,
        eq_stress=eq_stress,
        add_eq=True if eq_stress is not None else False,
    )
    c_ij = np.zeros((6, 6))
    residuals_sum = 0.0
    for ii in range(6):
        strain = ss_dict[strain_states[ii]]["strains"]
        stress = ss_dict[strain_states[ii]]["stresses"]
        for jj in range(6):
            fit = np.polyfit(strain[:, ii], stress[:, jj], 1, full=True)
            c_ij[ii, jj] = fit[0][0]
            residuals_sum += fit[1][0] if len(fit[1]) > 0 else 0.0
    elastic_tensor = ElasticTensor.from_voigt(c_ij)

    return {
        "elastic_tensor": elastic_tensor.zeroed(tolerance),
        "residuals_sum": residuals_sum,
    }
