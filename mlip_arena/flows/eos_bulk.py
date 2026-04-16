from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from ase.calculators.calculator import BaseCalculator
from ase.db import connect
from huggingface_hub import hf_hub_download
from prefect import flow, task
from prefect.cache_policies import INPUTS, TASK_SOURCE
from prefect.runtime import task_run
from scipy import stats

from mlip_arena.models import MLIPEnum
from mlip_arena.tasks.eos import run as EOS
from mlip_arena.tasks.optimize import run as OPT
from mlip_arena.tasks.utils import get_calculator

if TYPE_CHECKING:
    from ase import Atoms


def calculate_metrics(res_eos: dict, b0: float, atoms: Atoms, model_name: str, structure_id: str) -> dict:
    """Calculate analysis metrics from E-V curves."""
    es = np.array(res_eos["energies"])
    vols = np.array(res_eos["volumes"])

    indices = np.argsort(vols)
    vols = vols[indices]
    es = es[indices]

    imine = len(es) // 2
    emin = es[imine]
    vol0 = vols[imine]

    interpolated_volumes = [(vols[i] + vols[i + 1]) / 2 for i in range(len(vols) - 1)]
    ediff = np.diff(es)
    ediff_sign = np.sign(ediff)
    mask = ediff_sign != 0
    ediff = ediff[mask]
    ediff_sign = ediff_sign[mask]
    ediff_flip = np.diff(ediff_sign) != 0

    etv = np.sum(np.abs(np.diff(es)))

    return {
        "model": model_name,
        "structure": structure_id,
        "formula": atoms.get_chemical_formula(),
        "missing": False,
        "volume-ratio": vols / vol0,
        "energy-delta-per-atom": (es - emin) / len(atoms),
        "energy-diff-flip-times": np.sum(ediff_flip).astype(int),
        "energy-delta-per-volume-b0": (es - emin) / (b0 * vol0) if b0 else None,
        "tortuosity": etv / (abs(es[0] - emin) + abs(es[-1] - emin)),
        "spearman-compression-energy": stats.spearmanr(vols[:imine], es[:imine]).statistic,
        "spearman-compression-derivative": stats.spearmanr(interpolated_volumes[:imine], ediff[:imine]).statistic,
        "spearman-tension-energy": stats.spearmanr(vols[imine:], es[imine:]).statistic,
    }


@task(
    name="EOS bulk",
    task_run_name=lambda: (
        f"{task_run.task_name}: {task_run.parameters['atoms'].get_chemical_formula()} - {task_run.parameters['model_name']}"
    ),
    cache_policy=TASK_SOURCE + INPUTS,
)
def run(atoms: Atoms, model_name: str, model: str | BaseCalculator):
    """Run EOS bulk task for a single structure and model.

    Args:
        atoms (Atoms): ASE Atoms structure.
        model_name (str): Human-readable name of the model.
        model (str | BaseCalculator): The model or ASE calculator.

    Returns:
        pd.DataFrame: A DataFrame containing the raw EOS results.
    """
    calculator = model if isinstance(model, BaseCalculator) else get_calculator(model)

    result = OPT(
        atoms,
        calculator,
        optimizer="FIRE",
        criterion=dict(
            fmax=0.1,
        ),
    )
    result = EOS(
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

    return pd.DataFrame([result])


@flow
def run_db(
    model: str | BaseCalculator,
    run_dir: Path | None = None,
    dataset: str = "atomind/mlip-arena",
    dataset_file: str = "wbm_subset.db",
):
    """Run bulk EOS calculations over a database of structures.

    Args:
        model (str | BaseCalculator): Model name or ASE calculator.
        run_dir (Path, optional): Directory to save outputs (parquet files). Defaults to None.
        dataset (str, optional): HuggingFace dataset ID. Defaults to "atomind/mlip-arena".
        dataset_file (str, optional): Database filename in the dataset. Defaults to "wbm_subset.db".

    Returns:
        pd.DataFrame: A DataFrame containing analyzed results for all structures in the database.
    """
    if isinstance(model, BaseCalculator):
        model_name = model.__class__.__name__
    elif isinstance(model, str) and hasattr(MLIPEnum, model):
        model_name = model
    else:
        raise ValueError(f"Unsupported model: {model}")

    out_dir = run_dir if run_dir is not None else Path.cwd() / "eos_bulk" / model_name
    out_dir.mkdir(parents=True, exist_ok=True)

    db_path = hf_hub_download(repo_id=dataset, filename=dataset_file, repo_type="dataset")

    futures = []

    with connect(db_path) as db:
        for row in db.select():
            atoms = row.toatoms(add_additional_information=True)
            future = run.submit(atoms, model_name, model)
            futures.append(future)

    results = [f.result(raise_on_failure=False) for f in futures]
    valid_results = [res for res in results if isinstance(res, pd.DataFrame)]

    if not valid_results:
        return pd.DataFrame()

    df_raw_results = pd.concat(valid_results, ignore_index=True)
    df_raw_results.to_parquet(out_dir / f"{model_name}.parquet")

    df_analyzed = pd.DataFrame(
        columns=[
            "model",
            "structure",
            "formula",
            "volume-ratio",
            "energy-delta-per-atom",
            "energy-diff-flip-times",
            "energy-delta-per-volume-b0",
            "tortuosity",
            "spearman-compression-energy",
            "spearman-compression-derivative",
            "spearman-tension-energy",
            "missing",
        ]
    )

    with connect(db_path) as db:
        for row in db.select():
            atoms = row.toatoms(add_additional_information=True)
            structure_id = atoms.info["key_value_pairs"]["wbm_id"]

            try:
                row_results = df_raw_results.loc[df_raw_results["id"] == structure_id]
                if row_results.empty:
                    raise ValueError(f"No results found for {structure_id}")

                res_eos = row_results["eos"].iloc[0]
                b0 = row_results["b0"].iloc[0]
                data = calculate_metrics(res_eos, b0, atoms, model_name, structure_id)
            except Exception:
                data = {
                    "model": model_name,
                    "structure": structure_id,
                    "formula": atoms.get_chemical_formula(),
                    "missing": True,
                    "volume-ratio": None,
                    "energy-delta-per-atom": None,
                    "energy-diff-flip-times": None,
                    "energy-delta-per-volume-b0": None,
                    "tortuosity": None,
                    "spearman-compression-energy": None,
                    "spearman-compression-derivative": None,
                    "spearman-tension-energy": None,
                }

            df_analyzed = pd.concat([df_analyzed, pd.DataFrame([data])], ignore_index=True)

    df_analyzed.to_parquet(out_dir / f"{model_name}_processed.parquet")

    return df_analyzed
