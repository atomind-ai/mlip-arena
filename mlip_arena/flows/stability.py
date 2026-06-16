from collections.abc import Generator, Iterable
import os
from pathlib import Path
from typing import Literal, TYPE_CHECKING
import numpy as np
import pandas as pd
from ase import Atoms, units
from ase.db import connect
from ase.io import read
from ase.calculators.calculator import BaseCalculator
from dotenv import load_dotenv
from huggingface_hub import HfApi, hf_hub_download
from loguru import logger
from prefect import flow, task
from prefect.cache_policies import INPUTS, TASK_SOURCE
from prefect.futures import wait
from tqdm.auto import tqdm

from mlip_arena.models import MLIPEnum, REGISTRY
from mlip_arena.tasks import MD
from mlip_arena.tasks.utils import resolve_calculator_name

if TYPE_CHECKING:
    pass

load_dotenv()

HF_TOKEN = os.environ.get("HF_TOKEN", None)


@task(cache_policy=TASK_SOURCE + INPUTS)
def nvt_heat_one(
    atoms: Atoms,
    calculator: str | MLIPEnum | BaseCalculator | None = None,
    calculator_kwargs: dict | None = None,
    dispersion: bool = False,
    dispersion_kwargs: dict | None = None,
    run_dir: Path | None = None,
):
    """Run a 10 ps NVT MD simulation with linear heating schedule."""
    model_name = resolve_calculator_name(calculator)

    return MD.with_options(refresh_cache=True)(
        atoms=atoms,
        calculator=calculator,
        calculator_kwargs=calculator_kwargs,
        dispersion=dispersion,
        dispersion_kwargs=dispersion_kwargs,
        ensemble="nvt",
        dynamics="nose-hoover",
        time_step=None,
        dynamics_kwargs=dict(
            ttime=25 * units.fs,
        ),
        total_time=1e4,  # 10 ps
        temperature=[300, 3000],
        pressure=None,
        traj_file=run_dir
        / f"{model_name}_{atoms.info.get('material_id', 'random')}_{atoms.get_chemical_formula()}_nvt.traj",
        traj_interval=10,
    )


@task(cache_policy=TASK_SOURCE + INPUTS)
def npt_compress_one(
    atoms: Atoms,
    calculator: str | MLIPEnum | BaseCalculator | None = None,
    calculator_kwargs: dict | None = None,
    dispersion: bool = False,
    dispersion_kwargs: dict | None = None,
    run_dir: Path | None = None,
):
    """Run a 10 ps NPT MD simulation with linear pressure ramp."""
    model_name = resolve_calculator_name(calculator)

    return MD.with_options(timeout_seconds=600, retries=2, refresh_cache=True)(
        atoms=atoms,
        calculator=calculator,
        calculator_kwargs=calculator_kwargs,
        dispersion=dispersion,
        dispersion_kwargs=dispersion_kwargs,
        ensemble="npt",
        dynamics="nose-hoover",
        time_step=None,
        dynamics_kwargs=dict(ttime=25 * units.fs, pfactor=((75 * units.fs) ** 2) * 1e2 * units.GPa),
        total_time=1e4,  # 10 ps
        temperature=[300, 3000],
        pressure=[0, 5e2 * units.GPa],  # 500 GPa
        traj_file=run_dir
        / f"{model_name}_{atoms.info.get('material_id', 'random')}_{atoms.get_chemical_formula()}_npt.traj",
        traj_interval=10,
    )


@flow
def heating(
    calculator: str | MLIPEnum | BaseCalculator | None = None,
    calculator_kwargs: dict | None = None,
    dispersion: bool = False,
    dispersion_kwargs: dict | None = None,
    run_dir: Path | None = None,
    hf_token: str | None = HF_TOKEN,
):
    """Prefect flow to run NVT heating tasks for many database structures."""
    model_name = resolve_calculator_name(calculator)

    family = REGISTRY[model_name]["family"] if hasattr(MLIPEnum, model_name) else "custom"
    out_dir = run_dir if run_dir is not None else Path.cwd() / "stability" / family
    out_dir.mkdir(parents=True, exist_ok=True)

    futures = []
    # To download the database automatically, `huggingface_hub login` or provide HF_TOKEN
    for i, atoms in enumerate(get_atoms_from_db("random-mixture.db", hf_token=hf_token, force_download=False)):
        if i >= 200:
            break
        future = nvt_heat_one.with_options(
            timeout_seconds=600, retries=2, persist_result=False, refresh_cache=True
        ).submit(
            atoms=atoms.copy(),
            calculator=calculator,
            calculator_kwargs=calculator_kwargs,
            dispersion=dispersion,
            dispersion_kwargs=dispersion_kwargs,
            run_dir=out_dir,
        )
        futures.append(future)

    wait(futures)

    return [f.result(timeout=None, raise_on_failure=False) for f in futures if f.state.is_completed()]


@flow
def compression(
    calculator: str | MLIPEnum | BaseCalculator | None = None,
    calculator_kwargs: dict | None = None,
    dispersion: bool = False,
    dispersion_kwargs: dict | None = None,
    run_dir: Path | None = None,
    hf_token: str | None = HF_TOKEN,
):
    """Prefect flow to run NPT compression tasks for many database structures."""
    model_name = resolve_calculator_name(calculator)

    family = REGISTRY[model_name]["family"] if hasattr(MLIPEnum, model_name) else "custom"
    out_dir = run_dir if run_dir is not None else Path.cwd() / "stability" / family
    out_dir.mkdir(parents=True, exist_ok=True)

    futures = []
    # To download the database automatically, `huggingface_hub login` or provide HF_TOKEN
    for i, atoms in enumerate(get_atoms_from_db("random-mixture.db", hf_token=hf_token, force_download=False)):
        if i >= 200:
            break
        future = npt_compress_one.with_options(
            timeout_seconds=600, retries=2, persist_result=False, refresh_cache=True
        ).submit(
            atoms=atoms.copy(),
            calculator=calculator,
            calculator_kwargs=calculator_kwargs,
            dispersion=dispersion,
            dispersion_kwargs=dispersion_kwargs,
            run_dir=out_dir,
        )
        futures.append(future)

    wait(futures)

    return [f.result(timeout=None, raise_on_failure=False) for f in futures if f.state.is_completed()]


def save_to_db(
    atoms_list: list[Atoms] | Iterable[Atoms] | Atoms,
    db_path: Path | str,
    upload: bool = True,
    hf_token: str | None = os.getenv("HF_TOKEN", None),
    repo_id: str = "atomind/mlip-arena",
    repo_type: str = "dataset",
    subfolder: str = "stability",
):
    """Save ASE Atoms objects to an ASE database and optionally upload to
    Hugging Face Hub.
    """
    if upload and hf_token is None:
        raise ValueError("HF_TOKEN is required to upload the database.")

    db_path = Path(db_path)

    if isinstance(atoms_list, Atoms):
        atoms_list = [atoms_list]

    with connect(db_path) as db:
        for atoms in atoms_list:
            if not isinstance(atoms, Atoms):
                raise ValueError("atoms_list must contain ASE Atoms objects.")
            db.write(atoms)

    if upload:
        api = HfApi(token=hf_token)
        api.upload_file(
            path_or_fileobj=db_path,
            path_in_repo=f"{subfolder}/{db_path.name}",
            repo_id=repo_id,
            repo_type=repo_type,
        )
        logger.info(f"{db_path.name} uploaded to {repo_id}/{subfolder}")

    return db_path


@task
def get_atoms_from_db(
    db_path: Path | str,
    hf_token: str | None = os.getenv("HF_TOKEN", None),
    repo_id: str = "atomind/mlip-arena",
    repo_type: str = "dataset",
    subfolder: str = "stability",
    force_download: bool = False,
) -> Generator[Atoms, None, None]:
    """Retrieve ASE Atoms objects from an ASE database."""
    db_path = Path(db_path)
    if not db_path.exists():
        db_path = hf_hub_download(
            repo_id=repo_id,
            repo_type=repo_type,
            subfolder=subfolder,
            # local_dir=db_path.parent,
            filename=db_path.name,
            token=hf_token,
            force_download=force_download,
        )
    with connect(db_path) as db:
        for row in db.select():
            yield row.toatoms()


def get_runtime_stats(traj: list[Atoms], atoms0: Atoms):
    """Compute runtime statistics for an ASE trajectory.

    Parameters
    ----------
    traj : list[ase.Atoms]
        Sequence of ASE Atoms objects representing trajectory frames. Each frame
        is expected to have an `info` dict containing at least the keys
        'restart', 'datetime', and 'step'. The first frame should provide
        'target_steps'.
    atoms0 : ase.Atoms
        Reference Atoms object (typically the first frame) used to compute
        center-of-mass drift and to determine the number of atoms.

    Returns:
    -------
    dict
        A dictionary containing the following keys:
        - 'natoms': int, number of atoms from atoms0.
        - 'total_time_seconds': float, total wall-clock time summed across
          unique restart blocks (seconds).
        - 'total_steps': int, total MD steps summed across unique restart blocks.
        - 'steps_per_second': float, throughput (0 if total_time_seconds == 0).
        - 'seconds_per_step': float, average seconds per step (inf if total_steps == 0).
        - 'seconds_per_step_per_atom': float, seconds per step normalized by atom count.
        - 'energies': list of potential energies for successfully parsed frames.
        - 'kinetic_energies': list of kinetic energies.
        - 'temperatures': list of temperatures.
        - 'pressures': list of mean pressures (may be empty if not available).
        - 'target_steps': target number of steps taken from traj[0].info.
        - 'final_step': last recorded step number (0 if no valid frames).
        - 'timestep': array of step numbers for valid frames.
        - 'com_drifts': list of center-of-mass drift vectors relative to atoms0.

    Notes:
    -----
    Frames that raise exceptions when querying potential energy are skipped.
    Unique restart blocks are identified by atoms.info['restart'] and used to
    compute contiguous time and step differences across restarts.
    """
    restarts = []
    steps, times = [], []
    Ts, Ps, Es, KEs = [], [], [], []
    com_drifts = []

    for atoms in traj:
        try:
            energy = atoms.get_potential_energy()
            assert np.isfinite(energy), f"invalid energy: {energy}"
        except Exception:
            continue

        restarts.append(atoms.info["restart"])
        times.append(atoms.info["datetime"])
        steps.append(atoms.info["step"])
        Es.append(energy)
        KEs.append(atoms.get_kinetic_energy())
        Ts.append(atoms.get_temperature())
        try:
            Ps.append(atoms.get_stress()[:3].mean())
        except Exception:
            Ps.append(np.nan)
        com_drifts.append((atoms.get_center_of_mass() - atoms0.get_center_of_mass()).tolist())

    restarts = np.array(restarts)
    times = np.array(times)
    steps = np.array(steps)

    # Identify unique blocks
    unique_restarts = np.unique(restarts)

    total_time_seconds = 0
    total_steps = 0

    # Iterate over unique blocks to calculate averages
    for block in unique_restarts:
        # Get the indices corresponding to the current block
        # indices = np.where(restarts == block)[0]
        indices = restarts == block
        # Extract the corresponding data values
        block_time = times[indices][-1] - times[indices][0]
        total_time_seconds += block_time.total_seconds()
        total_steps += steps[indices][-1] - steps[indices][0]

    target_steps = traj[1].info["target_steps"]
    natoms = len(atoms0)

    return {
        "natoms": natoms,
        "total_time_seconds": total_time_seconds,
        "total_steps": total_steps,
        "steps_per_second": total_steps / total_time_seconds if total_time_seconds != 0 else 0,
        "seconds_per_step": total_time_seconds / total_steps if total_steps != 0 else float("inf"),
        "seconds_per_step_per_atom": total_time_seconds / total_steps / natoms if total_steps != 0 else float("inf"),
        "energies": Es,
        "kinetic_energies": KEs,
        "temperatures": Ts,
        "pressures": Ps,
        "target_steps": target_steps,
        "final_step": steps[-1] if len(steps) != 0 else 0,
        "timestep": steps,
        "com_drifts": com_drifts,
    }


def gather_results(run_dir: Path, prefix: str, run_type: Literal["nvt", "npt"]) -> pd.DataFrame:
    """Gather stability analysis results from multiple runs into a single DataFrame.

    Args:
        run_dir (Path): Output directory containing the runs.
        prefix (str): Prefix of the target files to gather.
        run_type (Literal["nvt", "npt"]): MD ensemble type used.

    Returns:
        pd.DataFrame: Aggregated results table.
    """
    df = pd.DataFrame()

    run_dir = Path(run_dir)
    files = list(run_dir.glob(f"{prefix}_*{run_type}.traj"))

    for fpath in tqdm(files, desc=prefix):
        try:
            traj = read(fpath, index=":")
        except Exception as e:
            logger.warning(f"Error reading {fpath}: {e}")
            continue

        try:
            stats = get_runtime_stats(traj, atoms0=traj[0])
            df = pd.concat(
                [
                    df,
                    pd.DataFrame(
                        {
                            # "model": model_name,
                            "formula": traj[0].get_chemical_formula(),
                            "normalized_timestep": stats["timestep"] / stats["target_steps"],
                            "normalized_final_step": stats["final_step"] / stats["target_steps"],
                            "pressure": np.array(stats["pressures"]) / units.GPa,
                        }
                        | stats
                    ),
                ],
                ignore_index=True,
            )
        except Exception as e:
            logger.warning(f"Error processing {fpath}: {e}")
            continue
    return df
