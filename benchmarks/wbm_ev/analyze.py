from pathlib import Path

import numpy as np
import pandas as pd
from ase.db import connect
from scipy import stats

from mlip_arena.models import REGISTRY, MLIPEnum

DATA_DIR = Path(__file__).parent.absolute()


def load_wbm_structures():
    """
    Load the WBM structures from a ASE DB file.
    """
    with connect(DATA_DIR.parent / "wbm_structures.db") as db:
        for row in db.select():
            yield row.toatoms(add_additional_information=True)

def gather_results():
    for model in MLIPEnum:
        if "wbm_ev" not in REGISTRY[model.name].get("gpu-tasks", []):
            continue

        if (DATA_DIR / f"{model.name}.parquet").exists():
            continue

        all_data = []

        for atoms in load_wbm_structures():
            fpath = Path(model.name) / f"{atoms.info['key_value_pairs']['wbm_id']}.json"
            if not fpath.exists():
                continue

            all_data.append(pd.read_json(fpath))

        df = pd.concat(all_data, ignore_index=True)
        df.to_parquet(DATA_DIR / f"{model.name}.parquet")


def summarize():
    summary_table = pd.DataFrame(
        columns=[
            "model",
            "energy-diff-flip-times",
            "tortuosity",
            "spearman-compression-energy",
            "spearman-compression-derivative",
            "spearman-tension-energy",
            "missing",
        ]
    )


    for model in MLIPEnum:
        fpath = DATA_DIR / f"{model.name}.parquet"
        if not fpath.exists():
            continue
        df_raw_results = pd.read_parquet(fpath)

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

        for wbm_struct in load_wbm_structures():
            structure_id = wbm_struct.info["key_value_pairs"]["wbm_id"]

            try:
                results = df_raw_results.loc[df_raw_results["id"] == structure_id]
                results = results["eos"].values[0]
                es = np.array(results["energies"])
                vols = np.array(results["volumes"])
                vol0 = wbm_struct.get_volume()

                indices = np.argsort(vols)
                vols = vols[indices]
                es = es[indices]

                imine = len(es) // 2
                # min_center_val = np.min(es[imid - 1 : imid + 2])
                # imine = np.where(es == min_center_val)[0][0]
                emin = es[imine]

                interpolated_volumes = [
                    (vols[i] + vols[i + 1]) / 2 for i in range(len(vols) - 1)
                ]
                ediff = np.diff(es)
                ediff_sign = np.sign(ediff)
                mask = ediff_sign != 0
                ediff = ediff[mask]
                ediff_sign = ediff_sign[mask]
                ediff_flip = np.diff(ediff_sign) != 0

                etv = np.sum(np.abs(np.diff(es)))

                data = {
                    "model": model.name,
                    "structure": structure_id,
                    "formula": wbm_struct.get_chemical_formula(),
                    "missing": False,
                    "volume-ratio": vols / vol0,
                    "energy-delta-per-atom": (es - emin) / len(wbm_struct),
                    "energy-diff-flip-times": np.sum(ediff_flip).astype(int),
                    "tortuosity": etv / (abs(es[0] - emin) + abs(es[-1] - emin)),
                    "spearman-compression-energy": stats.spearmanr(
                        vols[:imine], es[:imine]
                    ).statistic,
                    "spearman-compression-derivative": stats.spearmanr(
                        interpolated_volumes[:imine], ediff[:imine]
                    ).statistic,
                    "spearman-tension-energy": stats.spearmanr(
                        vols[imine:], es[imine:]
                    ).statistic,
                }

            except Exception:
                data = {
                    "model": model.name,
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

        df_analyzed.to_parquet(DATA_DIR / f"{model.name}_processed.parquet")
        # json_fpath = DATA_DIR / f"EV_scan_analyzed_{model.name}.json"

        # df_analyzed.to_json(json_fpath, orient="records")

        valid_results = df_analyzed[df_analyzed["missing"] == False]

        analysis_summary = {
            "model": model.name,
            "energy-diff-flip-times": valid_results["energy-diff-flip-times"].mean(),
            "tortuosity": valid_results["tortuosity"].mean(),
            "spearman-compression-energy": valid_results[
                "spearman-compression-energy"
            ].mean(),
            "spearman-compression-derivative": valid_results[
                "spearman-compression-derivative"
            ].mean(),
            "spearman-tension-energy": valid_results["spearman-tension-energy"].mean(),
            "missing": len(df_analyzed[df_analyzed["missing"] == True]),
        }
        summary_table = pd.concat(
            [summary_table, pd.DataFrame([analysis_summary])], ignore_index=True
        )


    flip_rank = (
        (summary_table["energy-diff-flip-times"] - 1)
        .abs()
        .rank(ascending=True, method="min")
    )
    tortuosity_rank = summary_table["tortuosity"].rank(ascending=True, method="min")
    spearman_compression_energy_rank = summary_table["spearman-compression-energy"].rank(
        method="min"
    )
    spearman_compression_derivative_rank = summary_table[
        "spearman-compression-derivative"
    ].rank(ascending=False, method="min")
    spearman_tension_energy_rank = summary_table["spearman-tension-energy"].rank(
        ascending=False, method="min"
    )
    missing_rank = summary_table["missing"].rank(ascending=True, method="min")

    rank_aggr = (
        flip_rank
        + tortuosity_rank
        + spearman_compression_energy_rank
        + spearman_compression_derivative_rank
        + spearman_tension_energy_rank
        + missing_rank
    )
    rank = rank_aggr.rank(method="min")

    summary_table.insert(1, "rank", rank.astype(int))
    summary_table.insert(2, "rank-aggregation", rank_aggr.astype(int))
    summary_table = summary_table.sort_values(by="rank", ascending=True)
    summary_table = summary_table.reset_index(drop=True)

    summary_table.to_csv(DATA_DIR / "summary.csv", index=False)
    summary_table.to_latex(DATA_DIR / "summary.tex", index=False)

    return summary_table

if __name__ == "__main__":
    gather_results()
    summarize()
