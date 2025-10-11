import itertools
from pathlib import Path

import numpy as np
from ase import Atom, Atoms
from ase.calculators.calculator import BaseCalculator
from ase.data import chemical_symbols, covalent_radii, vdw_alvarez
from ase.io import read, write
from prefect import flow, task
from tqdm.auto import tqdm

from mlip_arena.models import REGISTRY, MLIPEnum
from mlip_arena.tasks.utils import get_calculator


@task
def homonuclear_diatomic(symbol: str, calculator: BaseCalculator, out_dir: Path):
    """
    Compute and save the potential energy curve for a homonuclear diatomic by varying the interatomic distance.
    
    The distance range is determined from the element's covalent and van der Waals radii (minimum = 0.9 × covalent radius; maximum = 3.1 × vdw radius or 6 Å if vdw is unknown) with a fixed step of 0.01 Å. Results are appended to an extxyz trajectory file named "<symbol><symbol>.extxyz" in the provided output directory; if that file exists the calculation resumes from the last frame. The provided ASE calculator is used to evaluate potential energies and each geometry is saved as a frame in the trajectory.
    """

    atom = Atom(symbol)
    rmin = 0.9 * covalent_radii[atom.number]
    rvdw = (
        vdw_alvarez.vdw_radii[atom.number]
        if atom.number < len(vdw_alvarez.vdw_radii)
        else np.nan
    )
    rmax = 3.1 * rvdw if not np.isnan(rvdw) else 6
    rstep = 0.01
    npts = int((rmax - rmin) / rstep)

    rs = np.linspace(rmin, rmax, npts)
    es = np.zeros_like(rs)

    da = symbol + symbol

    out_dir.mkdir(parents=True, exist_ok=True)

    skip = 0

    a = 5 * rmax
    r = rs[0]

    positions = [
        [a / 2 - r / 2, a / 2, a / 2],
        [a / 2 + r / 2, a / 2, a / 2],
    ]

    traj_fpath = out_dir / f"{da!s}.extxyz"

    if traj_fpath.exists():
        traj = read(traj_fpath, index=":")
        skip = len(traj)
        atoms = traj[-1]
    else:
        # Create the unit cell with two atoms
        atoms = Atoms(
            da,
            positions=positions,
            # magmoms=magmoms,
            cell=[a, a + 0.001, a + 0.002],
            pbc=False,
        )

    atoms.calc = calculator

    for i, r in enumerate(tqdm(rs)):
        if i < skip:
            continue

        positions = [
            [a / 2 - r / 2, a / 2, a / 2],
            [a / 2 + r / 2, a / 2, a / 2],
        ]

        # atoms.set_initial_magnetic_moments(magmoms)
        atoms.set_positions(positions)
        es[i] = atoms.get_potential_energy()
        write(traj_fpath, atoms, append="a")


@flow
def homonuclear_diatomics(model: str | BaseCalculator, run_dir: Path):

    """
    Submit homonuclear diatomic potential curve tasks for a given model and collect their results.
    
    Parameters:
        model (str | BaseCalculator): Model identifier or calculator instance; if a string, it is mapped to the corresponding MLIPEnum.
        run_dir (Path): Base directory where per-model output directories will be created.
    
    Returns:
        list: Results returned by each submitted homonuclear diatomic task, in the same order as chemical symbols were processed. Each entry is the task's return value or an exception wrapper if the task failed.
    """
    model = MLIPEnum[model] if isinstance(model, str) else model
    model_name = model.name if isinstance(model, MLIPEnum) else model.__class__.__name__

    futures = []

    for symbol in chemical_symbols[1:]:
        out_dir = run_dir / model_name
        calculator = get_calculator(model)
        future = homonuclear_diatomic.submit(
            symbol,
            calculator,
            out_dir=out_dir,
        )
        futures.append(future)

    return [f.result(raise_on_failure=False) for f in futures]


if __name__ == "__main__":
    homonuclear_diatomics.with_options(
        # task_runner=DaskTaskRunner(address=client.scheduler.address),
        log_prints=True,
    )()