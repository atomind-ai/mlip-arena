"""
This module has been adapted from Quacc (https://github.com/Quantum-Accelerators/quacc). By using this software, you agree to the Quacc license agreement: https://github.com/Quantum-Accelerators/quacc/blob/main/LICENSE.md


BSD 3-Clause License

Copyright (c) 2025, Andrew S. Rosen.
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

- Redistributions of source code must retain the above copyright notice, this
  list of conditions and the following disclaimer.

- Redistributions in binary form must reproduce the above copyright notice,
  this list of conditions and the following disclaimer in the documentation
  and/or other materials provided with the distribution.

- Neither the name of the copyright holder nor the names of its
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

from pathlib import Path

import numpy as np
from ase import Atoms
from ase.calculators.calculator import BaseCalculator
from prefect import task
from prefect.cache_policies import INPUTS, TASK_SOURCE
from prefect.runtime import task_run

from mlip_arena.tasks.utils import logger

try:
    from phonopy import Phonopy
    from phonopy.structure.atoms import PhonopyAtoms
except ImportError as e:
    logger.warning(e)
    logger.warning(
        "Phonopy is not installed. Please install it following the instructions at https://phonopy.github.io/phonopy/install.html to use this module."
    )


@task(cache_policy=TASK_SOURCE + INPUTS)
def get_phonopy(
    atoms: Atoms,
    supercell_matrix: list[int] | None = None,
    min_lengths: float | tuple[float, float, float] | None = None,
    symprec: float = 1e-5,
    distance: float = 0.01,
    phonopy_kwargs: dict = {},
) -> Phonopy:
    if supercell_matrix is None and min_lengths is not None:
        supercell_matrix = np.diag(
            np.round(np.ceil(min_lengths / atoms.cell.lengths()))
        )

    phonon = Phonopy(
        PhonopyAtoms(
            symbols=atoms.get_chemical_symbols(),
            cell=atoms.get_cell(),
            scaled_positions=atoms.get_scaled_positions(wrap=True),
            masses=atoms.get_masses(),
        ),
        symprec=symprec,
        supercell_matrix=supercell_matrix,
        **phonopy_kwargs,
    )
    phonon.generate_displacements(distance=distance)

    return phonon


def _get_forces(
    phononpy_atoms: PhonopyAtoms,
    calculator: BaseCalculator,
) -> np.ndarray:
    atoms = Atoms(
        symbols=phononpy_atoms.symbols,
        cell=phononpy_atoms.cell,
        scaled_positions=phononpy_atoms.scaled_positions,
        pbc=True,
    )

    atoms.calc = calculator

    return atoms.get_forces()


def _generate_task_run_name():
    task_name = task_run.task_name
    parameters = task_run.parameters

    atoms = parameters["atoms"]
    calculator_name = parameters["calculator"]

    return f"{task_name}: {atoms.get_chemical_formula()} - {calculator_name}"


@task(
    name="PHONON",
    task_run_name=_generate_task_run_name,
    cache_policy=TASK_SOURCE + INPUTS,
)
def run(
    atoms: Atoms,
    calculator: BaseCalculator,
    supercell_matrix: list[int] | None = None,
    min_lengths: float | tuple[float, float, float] | None = None,
    symprec: float = 1e-5,
    distance: float = 0.01,
    phonopy_kwargs: dict = {},
    symmetry: bool = False,
    t_min: float = 0.0,
    t_max: float = 1000.0,
    t_step: float = 10.0,
    outdir: str | None = None,
):
    phonon = get_phonopy(
        atoms=atoms.copy(),
        supercell_matrix=supercell_matrix,
        min_lengths=min_lengths,
        symprec=symprec,
        distance=distance,
        phonopy_kwargs=phonopy_kwargs,
    )

    supercells_with_displacements = phonon.supercells_with_displacements

    phonon.forces = [
        _get_forces(supercell, calculator)
        for supercell in supercells_with_displacements
        if supercell is not None
    ]
    phonon.produce_force_constants()

    if symmetry:
        phonon.symmetrize_force_constants()
        phonon.symmetrize_force_constants_by_space_group()

    phonon.run_mesh(with_eigenvectors=True)
    phonon.run_total_dos()
    phonon.run_thermal_properties(t_step=t_step, t_max=t_max, t_min=t_min)  # type: ignore
    phonon.auto_band_structure(
        write_yaml=True if outdir is not None else False,
        filename=Path(outdir, "band.yaml") if outdir is not None else "band.yaml",
    )
    if outdir:
        phonon.save(Path(outdir, "phonopy.yaml"), settings={"force_constants": True})

    return {
        "phonon": phonon,
    }
