from pathlib import Path

import numpy as np
import pandas as pd
from ase import Atom, Atoms
from ase.calculators.calculator import BaseCalculator
from ase.data import chemical_symbols, covalent_radii, vdw_alvarez
from ase.io import read, write
from prefect import flow, task
from prefect.futures import wait
from scipy import stats
from tqdm.auto import tqdm

from mlip_arena.models import MLIPEnum, REGISTRY
from mlip_arena.tasks.utils import get_calculator


@task
def homonuclear_diatomic(symbol: str, calculator: BaseCalculator, out_dir: Path):
    """
    Compute and save the potential energy curve for a homonuclear diatomic over a range of interatomic distances.
    
    The distance range is determined from element radii: minimum = 0.9 × covalent radius, maximum = 3.1 × van der Waals radius (or 6.0 Å if van der Waals radius is unavailable), sampled with a fixed step of 0.01 Å. Results are written to an extxyz trajectory file named "<symbol><symbol>.extxyz" in out_dir; if that file already exists the computation resumes by appending missing frames.
    
    Parameters:
        symbol (str): Chemical symbol of the atom (e.g., 'H', 'O', 'Fe').
        calculator (BaseCalculator): ASE calculator used to evaluate potential energies for each geometry.
        out_dir (Path): Directory where the trajectory file will be created or appended.
    
    Side effects:
        Writes or appends trajectory frames to out_dir/<symbol><symbol>.extxyz. The function does not return a value.
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

@task
def analyze(out_dir: Path):

    """
    Analyze homonuclear diatomic trajectory files in a directory and return a DataFrame of potential-energy-curve metrics.
    
    Parameters:
    	out_dir (Path): Directory containing per-diatomic trajectory files named like "HeHe.extxyz", "LiLi.extxyz", etc.
    
    Returns:
    	df (pandas.DataFrame): One row per analyzed diatomic with columns:
    		- name: diatomic identifier (e.g., "HeHe")
    		- method: (placeholder, may be set later)
    		- R, E, F, S^2: arrays of interatomic distances, energies, forces, and spin-squared values
    		- force-flip-times: number of sign changes in the force sequence
    		- force-total-variation: sum of absolute force differences
    		- force-jump: magnitude measure of force discontinuities
    		- energy-diff-flip-times: number of sign changes in successive energy differences
    		- energy-grad-norm-max: maximum absolute dE/dr
    		- energy-jump: magnitude measure of energy discontinuities
    		- energy-total-variation: total variation of the energy curve
    		- tortuosity: normalized energy total variation
    		- conservation-deviation: mean absolute value of (F + dE/dr)
    		- spearman-descending-force, spearman-ascending-force: Spearman correlations of R vs F on descending/ascending branches
    		- spearman-repulsion-energy, spearman-attraction-energy: Spearman correlations of R vs E on repulsive/attractive branches
    		- pbe-energy-mae, pbe-force-mae: placeholders for comparison error metrics (may be populated externally)
    
    Notes:
    	Files that are missing or unreadable are skipped; only successfully read trajectories contribute rows.
    """
    df = pd.DataFrame(
        columns=[
            "name",
            "method",
            "R",
            "E",
            "F",
            "S^2",
            "force-flip-times",
            "force-total-variation",
            "force-jump",
            "energy-diff-flip-times",
            "energy-grad-norm-max",
            "energy-jump",
            "energy-total-variation",
            "tortuosity",
            "conservation-deviation",
            "spearman-descending-force",
            "spearman-ascending-force",
            "spearman-repulsion-energy",
            "spearman-attraction-energy",
            "pbe-energy-mae",
            "pbe-force-mae",
        ]
    )

    for symbol in chemical_symbols[1:]:
        da = symbol + symbol
        traj_fpath = out_dir / f"{da!s}.extxyz"

        if not traj_fpath.exists():
            continue

        traj = read(traj_fpath, index=":")

        #
        # Extract PEC data
        #

        Rs, Es, Fs, S2s = [], [], [], []
        for atoms in traj:
            vec = atoms.positions[1] - atoms.positions[0]
            r = np.linalg.norm(vec)
            e = atoms.get_potential_energy()
            f = np.inner(vec / r, atoms.get_forces()[1])
            # s2 = np.mean(np.power(atoms.get_magnetic_moments(), 2))

            Rs.append(r)
            Es.append(e)
            Fs.append(f)
            # S2s.append(s2)

        rs = np.array(Rs)
        es = np.array(Es)
        fs = np.array(Fs)

        #
        # Sort interatomic distances and align to zero at far field
        #

        indices = np.argsort(rs)[::-1]
        rs = rs[indices]
        es = es[indices]
        eshift = es[0]
        es -= eshift
        fs = fs[indices]

        #
        # Metrics
        #

        iminf = np.argmin(fs)
        imine = np.argmin(es)

        de_dr = np.gradient(es, rs)
        # d2e_dr2 = np.gradient(de_dr, rs)

        rounded_fs = np.copy(fs)
        rounded_fs[np.abs(rounded_fs) < 1e-2] = 0  # 10 meV/A

        fs_sign = np.sign(rounded_fs)
        mask = fs_sign != 0
        rounded_fs = rounded_fs[mask]
        fs_sign = fs_sign[mask]
        # force sign changes
        f_flip = np.diff(fs_sign) != 0

        fdiff = np.diff(fs)
        fdiff_sign = np.sign(fdiff)
        mask = fdiff_sign != 0
        fdiff = fdiff[mask]
        fdiff_sign = fdiff_sign[mask]
        fdiff_flip = np.diff(fdiff_sign) != 0
        # force discontinuities
        fjump = (
            np.abs(fdiff[:-1][fdiff_flip]).sum() + np.abs(fdiff[1:][fdiff_flip]).sum()
        )

        ediff = np.diff(es)
        ediff[np.abs(ediff) < 1e-3] = 0  # 1 meV
        ediff_sign = np.sign(ediff)
        mask = ediff_sign != 0
        ediff = ediff[mask]
        ediff_sign = ediff_sign[mask]
        ediff_flip = np.diff(ediff_sign) != 0
        # energy discontinuities
        ejump = (
            np.abs(ediff[:-1][ediff_flip]).sum() + np.abs(ediff[1:][ediff_flip]).sum()
        )

        # conservation deviation
        conservation_deviation = np.mean(np.abs(fs + de_dr))

        # total variation (for tortuosity)
        etv = np.sum(np.abs(np.diff(es)))

        data = {
            "name": da,
            # "method": model_name,
            "R": rs,
            "E": es + eshift,
            "F": fs,
            "S^2": S2s,
            "force-flip-times": np.sum(f_flip),
            "force-total-variation": np.sum(np.abs(np.diff(fs))),
            "force-jump": fjump,
            "energy-diff-flip-times": np.sum(ediff_flip),
            "energy-grad-norm-max": np.max(np.abs(de_dr)),
            "energy-jump": ejump,
            # "energy-grad-norm-mean": np.mean(de_dr_abs),
            "energy-total-variation": etv,
            "tortuosity": etv / (abs(es[0] - es.min()) + (es[-1] - es.min())),
            "conservation-deviation": conservation_deviation,
            "spearman-descending-force": stats.spearmanr(
                rs[iminf:], fs[iminf:]
            ).statistic,
            "spearman-ascending-force": stats.spearmanr(
                rs[:iminf], fs[:iminf]
            ).statistic,
            "spearman-repulsion-energy": stats.spearmanr(
                rs[imine:], es[imine:]
            ).statistic,
            "spearman-attraction-energy": stats.spearmanr(
                rs[:imine], es[:imine]
            ).statistic,
        }

        df = pd.concat([df, pd.DataFrame([data])], ignore_index=True)

    return df


@flow
def homonuclear_diatomics(model: str | BaseCalculator, run_dir: Path | None = None):

    """
    Orchestrates homonuclear diatomic PEC computations for all elements, collects their results, and writes a summary JSON to the output directory.
    
    Parameters:
        model (str | BaseCalculator): Either a model key string (lookup in MLIPEnum/REGISTRY) or an instantiated ASE calculator; determines the calculator used for each per-element task.
        run_dir (Path | None): Optional path to write per-element trajectories and the aggregated "homonuclear-diatomics.json". If omitted, a default directory is created under the current working directory using the model family and name.
    
    Returns:
        list: Per-element task results in the same order as chemical_symbols[1:]. Each entry is the corresponding task's return value or an exception-like object if the task failed.
    """
    model_name = MLIPEnum[model].name if isinstance(model, str) else model.__class__.__name__
    family = REGISTRY[model_name]['family'] if hasattr(MLIPEnum, model_name) else "custom"

    out_dir = run_dir if run_dir is not None else Path.cwd() / family / model_name

    futures = []
    for symbol in chemical_symbols[1:]:
        calculator = get_calculator(model)
        future = homonuclear_diatomic.submit(
            symbol,
            calculator,
            out_dir=out_dir,
        )
        futures.append(future)
    wait(futures)

    df = analyze(out_dir)
    df["method"] = model_name
    df.to_json(out_dir / "homonuclear-diatomics.json", orient="records")

    return [f.result(raise_on_failure=False) for f in futures]