# benchmarks/report.py

import argparse
import shutil
import warnings
from pathlib import Path

from eos_bulk.aggregate import summarize as summarize_eos_bulk
from ev.aggregate import summarize as summarize_ev
from stability.aggregate import summarize as summarize_stability


def get_model_name(calculator) -> str:
    """Helper to resolve a string model name from various calculator inputs."""
    if isinstance(calculator, str):
        return calculator
    if isinstance(calculator, type):
        return calculator.__name__
    if hasattr(calculator, "__class__"):
        return calculator.__class__.__name__
    return str(calculator)


def summarize(model_name):
    model_name = get_model_name(model_name)

    benchmarks_dir = Path(__file__).parent

    # Check which output files exist first to enable quick warning & early exit
    p_eos = benchmarks_dir / "eos_bulk" / f"{model_name}_processed.parquet"
    p_ev = benchmarks_dir / "ev" / f"{model_name}_processed.parquet"

    family = "custom"
    try:
        from mlip_arena.models import REGISTRY

        if model_name in REGISTRY:
            family = REGISTRY[model_name].get("family", "custom").lower()
    except Exception:
        pass

    run_dir_stability = benchmarks_dir / "stability" / family
    nvt_files = list(run_dir_stability.glob(f"{model_name}_*nvt.traj"))
    npt_files = list(run_dir_stability.glob(f"{model_name}_*npt.traj"))

    p_combustion = benchmarks_dir / "combustion" / family / f"{model_name}_H256O128.json"

    has_eos = p_eos.exists()
    has_ev = p_ev.exists()
    has_stability = bool(nvt_files or npt_files)
    has_combustion = p_combustion.exists()

    if not (has_eos or has_ev or has_stability or has_combustion):
        warnings.warn(
            f"All expected benchmark output files are missing for model '{model_name}'. "
            f"Expected files at: {p_eos}, {p_ev}, {p_combustion}, or stability traj files in {run_dir_stability}. "
            "Bailing out of report generation."
        )
        return

    # 1. EOS Bulk
    if has_eos:
        dest_eos = p_eos.with_name(f"{model_name}_results.parquet")
        shutil.copy2(p_eos, dest_eos)
        print(f"Copied {p_eos.name} to {dest_eos.name}")
    else:
        warnings.warn(f"EOS Bulk processed file is missing for {model_name} at {p_eos}. Skipping EOS Bulk report.")

    # 2. EV
    if has_ev:
        dest_ev = p_ev.with_name(f"{model_name}_results.parquet")
        shutil.copy2(p_ev, dest_ev)
        print(f"Copied {p_ev.name} to {dest_ev.name}")
    else:
        warnings.warn(f"E-V processed file is missing for {model_name} at {p_ev}. Skipping E-V report.")

    # 3. Stability MD
    if has_stability:
        summarize_stability(model_name)
    else:
        warnings.warn(f"No stability trajectory files found for {model_name}. Skipping stability analysis.")

    # 4. Combustion
    if has_combustion:
        print(f"Combustion results JSON found for {model_name} at {p_combustion}.")
    else:
        warnings.warn(
            f"Combustion processed file is missing for {model_name} at {p_combustion}. Skipping combustion report check."
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Gather benchmark results and generate reports.")
    parser.add_argument(
        "--model",
        type=str,
        help="Optional name of the model to generate report/results for.",
    )
    args = parser.parse_args()

    benchmarks_dir = Path(__file__).parent

    if args.model:
        print(f"Generating reports and processed results for model: {args.model}")
        summarize(args.model)
    else:
        # Legacy behavior: copy all processed parquets to results parquets and run global summaries
        print("Gathering results and generating global summaries for all models...")
        for subdir in ["eos_bulk", "ev"]:
            dir_path = benchmarks_dir / subdir
            for p in dir_path.glob("*_processed.parquet"):
                dest = p.with_name(p.name.replace("_processed.parquet", "_results.parquet"))
                shutil.copy2(p, dest)

        print("Reporting EOS Bulk Results...")
        summarize_eos_bulk()

        print("\nReporting Energy-Volume (EV) Results...")
        summarize_ev()

        print("\nReporting Stability MD Results...")
        summarize_stability()
