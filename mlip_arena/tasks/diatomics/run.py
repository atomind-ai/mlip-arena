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
def homonuclear_diatomics(symbol: str, calculator: BaseCalculator, out_dir: Path):
    """
    Calculate potential energy curves for homonuclear diatomic molecules.

    This function computes the potential energy of a diatomic molecule (two atoms of
    the same element) across a range of interatomic distances. The distance range is
    automatically determined from the covalent and van der Waals radii of the element.

    Args:
        symbol: Chemical symbol of the atom (e.g., 'H', 'O', 'Fe')
        calculator: ASE calculator object used to compute the potential energies. Could be VASP, MLIP, etc.

    Returns:
        None: Results are saved as trajectory files in a directory structure:
              /{model_family}/{element_pair}/{model_name}.extxyz

    Note:
        - Minimum distance is set to 0.9× the covalent radius
        - Maximum distance is set to 3.1× the van der Waals radius (or 6 Å if unknown)
        - Distance step size is fixed at 0.01 Å
        - If an existing trajectory file is found, the calculation will resume from where it left off
        - The atoms are placed in a periodic box large enough to avoid self-interaction
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
def submit_homonuclear_diatomics():
    futures = []
    for symbol, model in itertools.product(
        chemical_symbols[1:],
        MLIPEnum,
    ):
        if "homonuclear-diatomics" not in REGISTRY[model.name].get("gpu-tasks", []):
            continue

        out_dir = Path(__file__).parent / model.name

        calculator = get_calculator(model)

        # if not (out_dir / "homonuclear-diatomics.json").exists():
        future = homonuclear_diatomics.submit(
            symbol,
            calculator,
            out_dir=out_dir,
        )
        futures.append(future)

    return [f.result(raise_on_failure=False) for f in futures]


if __name__ == "__main__":
    submit_homonuclear_diatomics.with_options(
        # task_runner=DaskTaskRunner(address=client.scheduler.address),
        log_prints=True,
    )()
