"""
Adapted from the k_SRME package, which is licensed under the GPL-3.0 license.
https://github.com/MPA2suite/k_SRME?tab=GPL-3.0-1-ov-file
"""

from k_srme import ID, NO_TILT_MASK, STRUCTURES, aseatoms2str, two_stage_relax
from k_srme.conductivity import (
    calculate_conductivity,
    get_fc2_and_freqs,
    get_fc3,
    init_phono3py,
)
from k_srme.utils import (
    check_imaginary_freqs,
    get_spacegroup_number,
    log_message,
    log_symmetry,
    symm_name_map,
)
from mlip_arena.models import MLIPEnum
from mlip_arena.tasks.optimize import run as OPT
from prefect import flow, task

from ase import Atoms
from ase.io import iread

from copy import deepcopy


@task
def get_atoms():
    for atoms in iread(STRUCTURES, format="extxyz", index=":"):
        yield atoms


@task
def get_thermal_conductivity(
    atoms: Atoms,
    calculator_name: str,
    calculator_kwargs: dict = {},
    steps_stage1=300,
    steps_stage2=300,
    enforce_symmetry: bool = True,
    symprec: float = 1e-5,
    conductivity_broken_symm: bool = False,
    symprec_tests: list[float] = [1e-5, 1e-4, 1e-3, 1e-1],
    # save_forces: bool = True,
):
    # TODO: move to flow
    # mat_id = atoms.info[ID]
    # init_info = deepcopy(atoms.info)
    # mat_name = atoms.info["name"]
    # mat_desc = f"{mat_name}-{symm_name_map[atoms.info['symm.no']]}"
    # info_dict = {
    #     "desc": mat_desc,
    #     "name": mat_name,
    #     "initial_space_group_number": atoms.info["symm.no"],
    #     "errors": [],
    #     "error_traceback": [],
    # }
    info_dict = {
        "errors": [],
        "error_traceback": [],
    }

    # Relaxation

    sym_init = log_symmetry(atoms, symprec, output=True)

    result1 = OPT(
        atoms=atoms,
        calculator_name=calculator_name,
        calculator_kwargs=calculator_kwargs,
        device=None,
        optimizer="FIRE",
        optimizer_kwargs={},
        filter="FrechetCell",
        filter_kwargs=dict(
            mask=[True, True, True, False, False, False],
        ),
        criterion=dict(fmax=1e-4, steps=steps_stage1),
        symmetry=True,
    )

    sym_stage1 = log_symmetry(atoms, symprec, output=True)
    max_stress_stage1 = atoms.get_stress().reshape((2, 3), order="C").max(axis=1)

    atoms = result1["atoms"].copy()
    atoms_stage1 = atoms.copy()
    atoms.constraints = None

    result2 = OPT(
        atoms=atoms,
        calculator_name=calculator_name,
        calculator_kwargs=calculator_kwargs,
        device=None,
        optimizer="FIRE",
        optimizer_kwargs={},
        filter="FrechetCell",
        filter_kwargs=dict(
            mask=[True, True, True, False, False, False],
        ),
        criterion=dict(fmax=1e-4, steps=steps_stage2),
        symmetry=False,
    )

    sym_stage2 = log_symmetry(atoms, symprec, output=True)

    # Test symmetries with various symprec if stage2 is different
    sym_tests = {}
    if sym_init.number != sym_stage2.number:
        for symprec_test in symprec_tests:
            log_message("Stage 2 Symmetry Test:", output=True)
            dataset_tests = log_symmetry(atoms, symprec_test, output=True)
            sym_tests[symprec_test] = dataset_tests.number

    max_stress_stage2 = atoms.get_stress().reshape((2, 3), order="C").max(axis=1)

    atoms = result1["atoms"].copy()
    atoms_stage2 = atoms.copy()

    if sym_stage1.number != sym_stage2.number and enforce_symmetry:
        redirected_to_symm = True
        atoms = atoms_stage1
        max_stress = max_stress_stage1
        sym_final = sym_stage1
        # warnings.warn(
        #     f"Symmetry is not kept after deleting FixSymmetry constraint, redirecting to structure with symmetry of material {mat_name}, in folder {os.getcwd()}"
        # )
        # log_message(
        #     f"Symmetry is not kept after deleting FixSymmetry constraint, redirecting to structure with symmetry of material {mat_name}, in folder {os.getcwd()}",
        #     output=log,
        # )
    else:
        redirected_to_symm = False
        sym_final = sym_stage2
        max_stress = max_stress_stage2

    reached_max_steps = (
        result1["nsteps"] == steps_stage1 or result2["nsteps"] == steps_stage2
    )

    relax_dict = {
        "max_stress": max_stress,
        "reached_max_steps": reached_max_steps,
        "relaxed_space_group_number": sym_final.number,
        "broken_symmetry": sym_final.number != sym_init.number,
        "symprec_tests": sym_tests,
        "redirected_to_symm": redirected_to_symm,
    }

    # Force calculation

    calc = MLIPEnum[calculator_name].value(**calculator_kwargs)

    try:
        ph3 = init_phono3py(atoms, log=False, symprec=symprec)

        ph3, fc2_set, freqs = get_fc2_and_freqs(
            ph3,
            calculator=calc,
            log=False,
            pbar_kwargs={"leave": False, "disable": True},
        )

        imaginary_freqs = check_imaginary_freqs(freqs)
        freqs_dict = {"imaginary_freqs": imaginary_freqs, "frequencies": freqs}

        # if conductivity condition is met, calculate fc3
        ltc_condition = not imaginary_freqs and (
            not relax_dict["broken_symmetry"] or conductivity_broken_symm
        )

        if ltc_condition:
            ph3, fc3_set = get_fc3(
                ph3,
                calculator=calc,
                log=False,
                pbar_kwargs={"leave": False, "disable": not True},
            )
        else:
            fc3_set = []

        if not ltc_condition:
            # warnings.warn(f"Material {mat_desc}, {mat_id} has imaginary frequencies.")
            return {}
    except Exception as exc:
        # warnings.warn(f"Failed to calculate force sets {mat_id}: {exc!r}")
        # traceback.print_exc()
        info_dict["errors"].append(f"ForceConstantError: {exc!r}")
        # info_dict["error_traceback"].append(traceback.format_exc())

    # Conductivity calculation

    try:
        ph3, kappa_dict = calculate_conductivity(ph3, log=False)

    except Exception as exc:
        # warnings.warn(f"Failed to calculate conductivity {mat_id}: {exc!r}")
        # traceback.print_exc()
        # info_dict["errors"].append(f"ConductivityError: {exc!r}")
        # info_dict["error_traceback"].append(traceback.format_exc())
        # kappa_results[mat_id] = info_dict | relax_dict | freqs_dict
        return {}

    return {
        "force": {"fc2_set": fc2_set, "fc3_set": fc3_set},
        "kappa": info_dict | relax_dict | freqs_dict | kappa_dict,
    }
