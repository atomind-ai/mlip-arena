# benchmarks/stability/aggregate.py

import warnings
from pathlib import Path

DATA_DIR = Path(__file__).parent.absolute()


def summarize(model_name: str | None = None):
    """Summarize stability MD results.

    If model_name is provided, processes results only for that model.
    Otherwise, processes results for all registered models.
    """
    # Dynamic registry lookup from registry.yaml to avoid importing heavy modules if not needed
    family_map = {}
    try:
        from mlip_arena.models import REGISTRY

        for k, v in REGISTRY.items():
            family_map[k] = v.get("family", "custom").lower()
    except Exception:
        pass

    if model_name:
        models = [model_name]
    else:
        # Fallback to importing MLIPEnum if summarizing everything
        from mlip_arena.models import MLIPEnum

        models = [m.name for m in MLIPEnum]

    # First check if there is anything to do for the requested models
    models_to_process = []
    for mname in models:
        family = family_map.get(mname, "custom")
        run_dir = DATA_DIR / family
        nvt_files = list(run_dir.glob(f"{mname}_*nvt.traj"))
        npt_files = list(run_dir.glob(f"{mname}_*npt.traj"))
        if nvt_files or npt_files:
            models_to_process.append((mname, family, run_dir, nvt_files, npt_files))
        elif model_name:
            # If specifically requested a single model and files are missing, warn
            warnings.warn(
                f"No stability trajectory files (*.traj) found for {mname} in {run_dir}. Skipping stability analysis."
            )

    if not models_to_process:
        return

    # Defer heavy imports until we are sure we have files to process
    from mlip_arena.flows.stability import gather_results, get_atoms_from_db

    compositions = []
    try:
        for atoms in get_atoms_from_db("random-mixture.db"):
            if len(atoms) == 0:
                continue
            compositions.append(atoms.get_chemical_formula())
    except Exception as e:
        warnings.warn(f"Failed to load random-mixture.db: {e}")
        return

    for mname, family, run_dir, nvt_files, npt_files in models_to_process:
        # Heating (nvt)
        if nvt_files:
            try:
                df_heat = gather_results(run_dir, prefix=mname, run_type="nvt")
                df_heat = df_heat[df_heat["formula"].isin(compositions[:120])].copy()
                if len(df_heat) > 0:
                    df_heat.to_parquet(run_dir / f"{mname}-heating.parquet", index=False)
                    print(f"Generated {mname}-heating.parquet")
                else:
                    warnings.warn(f"No valid heating results found for {mname}.")
            except Exception as e:
                warnings.warn(f"Error processing heating for {mname}: {e}")
        elif model_name:
            warnings.warn(f"No heating trajectory files found for {mname}. Skipping heating analysis.")

        # Compression (npt)
        if npt_files:
            try:
                df_comp = gather_results(run_dir, prefix=mname, run_type="npt")
                df_comp = df_comp[df_comp["formula"].isin(compositions[:80])].copy()
                if len(df_comp) > 0:
                    df_comp.to_parquet(run_dir / f"{mname}-compression.parquet", index=False)
                    print(f"Generated {mname}-compression.parquet")
                else:
                    warnings.warn(f"No valid compression results found for {mname}.")
            except Exception as e:
                warnings.warn(f"Error processing compression for {mname}: {e}")
        elif model_name:
            warnings.warn(f"No compression trajectory files found for {mname}. Skipping compression analysis.")


if __name__ == "__main__":
    summarize()
