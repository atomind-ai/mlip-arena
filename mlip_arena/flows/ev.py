from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from ase.calculators.calculator import BaseCalculator
from ase.db import connect
from huggingface_hub import hf_hub_download
from prefect import flow
from scipy import stats

from mlip_arena.models import MLIPEnum
from mlip_arena.tasks.ev import run as ev_scan

if TYPE_CHECKING:
    from ase import Atoms


def calculate_metrics(res_eos: dict, wbm_struct: Atoms, model_name: str, structure_id: str) -> dict:
    """Calculate analysis metrics from E-V curves."""
    es = np.array(res_eos["energies"])
    vols = np.array(res_eos["volumes"])
    vol0 = wbm_struct.get_volume()

    indices = np.argsort(vols)
    vols = vols[indices]
    es = es[indices]

    imine = len(es) // 2
    emin = es[imine]

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
        "formula": wbm_struct.get_chemical_formula(),
        "missing": False,
        "volume-ratio": vols / vol0,
        "energy-delta-per-atom": (es - emin) / len(wbm_struct),
        "energy-diff-flip-times": np.sum(ediff_flip).astype(int),
        "tortuosity": etv / (abs(es[0] - emin) + abs(es[-1] - emin)),
        "spearman-compression-energy": stats.spearmanr(vols[:imine], es[:imine]).statistic,
        "spearman-compression-derivative": stats.spearmanr(interpolated_volumes[:imine], ediff[:imine]).statistic,
        "spearman-tension-energy": stats.spearmanr(vols[imine:], es[imine:]).statistic,
    }


@flow
def run_db(
    model: str | BaseCalculator,
    run_dir: Path | None = None,
    dataset: str = "atomind/mlip-arena",
    dataset_file: str = "wbm_subset.db",
):
    """Run bulk E-V scan calculations over a database.

    Args:
        model (str | BaseCalculator): Model name or ASE calculator.
        run_dir (Path, optional): Directory to save outputs. Defaults to None.
        dataset (str, optional): HuggingFace dataset ID. Defaults to "atomind/mlip-arena".
        dataset_file (str, optional): Database filename. Defaults to "wbm_subset.db".

    Returns:
        pd.DataFrame: Analyzed results for all structures in the database.
    """
    if isinstance(model, BaseCalculator):
        model_name = model.__class__.__name__
    elif isinstance(model, str) and hasattr(MLIPEnum, model):
        model_name = model
    else:
        raise ValueError(f"Unsupported model: {model}")

    out_dir = run_dir if run_dir is not None else Path.cwd() / Path(__file__).stem

    out_dir.mkdir(parents=True, exist_ok=True)

    db_path = hf_hub_download(repo_id=dataset, filename=dataset_file, repo_type="dataset")

    futures = []

    with connect(db_path) as db:
        for row in db.select():
            atoms = row.toatoms(add_additional_information=True)
            future = ev_scan.submit(atoms, model)
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
            "tortuosity",
            "spearman-compression-energy",
            "spearman-compression-derivative",
            "spearman-tension-energy",
            "missing",
        ]
    )

    with connect(db_path) as db:
        for row in db.select():
            wbm_struct = row.toatoms(add_additional_information=True)
            structure_id = wbm_struct.info["key_value_pairs"]["wbm_id"]

            try:
                row_results = df_raw_results.loc[df_raw_results["id"] == structure_id]
                if row_results.empty:
                    raise ValueError(f"No results found for {structure_id}")

                res_eos = row_results["eos"].iloc[0]
                data = calculate_metrics(res_eos, wbm_struct, model_name, structure_id)
            except Exception:
                data = {
                    "model": model_name,
                    "structure": structure_id,
                    "formula": wbm_struct.get_chemical_formula(),
                    "missing": True,
                    "volume-ratio": None,
                    "energy-delta-per-atom": None,
                    "energy-diff-flip-times": None,
                    "tortuosity": None,
                    "spearman-compression-energy": None,
                    "spearman-compression-derivative": None,
                    "spearman-tension-energy": None,
                }

            df_analyzed = pd.concat([df_analyzed, pd.DataFrame([data])], ignore_index=True)

    df_analyzed.to_parquet(out_dir / f"{model_name}_processed.parquet")

    return df_analyzed
