from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd
from ase.db import connect
from prefect import flow, task
from prefect.cache_policies import INPUTS, TASK_SOURCE
from prefect.futures import wait
from prefect.runtime import task_run

from mlip_arena.models import MLIPEnum
from mlip_arena.tasks.utils import get_calculator

if TYPE_CHECKING:
    from ase import Atoms


@task
def load_wbm_structures(db_path: Path):
    """
    Load the WBM structures from an ASE database file.
    """
    with connect(db_path) as db:
        for row in db.select():
            yield row.toatoms(add_additional_information=True)


@task(
    name="EOS bulk - WBM",
    task_run_name=lambda: f"{task_run.task_name}: {task_run.parameters['atoms'].get_chemical_formula()} - {task_run.parameters['model_name']}",
    cache_policy=TASK_SOURCE + INPUTS,
)
def eos_bulk(atoms: Atoms, model_name: str, out_dir: Path):
    from mlip_arena.tasks.eos import run as EOS
    from mlip_arena.tasks.optimize import run as OPT

    calculator = get_calculator(model_name)

    result = OPT.with_options(
        refresh_cache=True,
    )(
        atoms,
        calculator,
        optimizer="FIRE",
        criterion=dict(
            fmax=0.1,
        ),
    )
    result = EOS.with_options(
        refresh_cache=True,
    )(
        atoms=result["atoms"],
        calculator=calculator,
        optimizer="FIRE",
        npoints=21,
        max_abs_strain=0.2,
        concurrent=False,
    )

    result["method"] = model_name
    result["id"] = atoms.info["key_value_pairs"]["wbm_id"]
    result.pop("atoms", None)

    out_dir.mkdir(parents=True, exist_ok=True)
    fpath = out_dir / f"{result['id']}.pkl"

    df = pd.DataFrame([result])
    df.to_pickle(fpath)

    return df


@flow
def wbm_eos_bulk(
    model: str,
    run_dir: Path | None = None,
    db_path: str = "benchmarks/wbm_structures.db",
):
    from ase.calculators.calculator import BaseCalculator

    if isinstance(model, BaseCalculator):
        model_name = model.__class__.__name__
    elif isinstance(model, str) and hasattr(MLIPEnum, model):
        model_name = model
    else:
        raise ValueError(f"Unsupported model: {model}")

    out_dir = run_dir if run_dir is not None else Path.cwd() / "eos_bulk" / model_name

    db_file = Path(db_path)
    if not db_file.exists():
        raise FileNotFoundError(f"Database not found at {db_file.absolute()}")

    futures = []
    for atoms in load_wbm_structures(db_file):
        future = eos_bulk.with_options(refresh_cache=True).submit(
            atoms, model_name, out_dir
        )
        futures.append(future)

    wait(futures)
    return [f.result(raise_on_failure=False) for f in futures]
