# benchmarks/report.py

import argparse
import shutil
import warnings
from pathlib import Path

from stability.aggregate import summarize as summarize_stability

BENCH_DIR = Path(__file__).parent


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

    family = "custom"
    try:
        from mlip_arena.models import REGISTRY

        if model_name in REGISTRY:
            family = REGISTRY[model_name].get("family", "custom").lower()
    except Exception:
        pass

    # Check which output files exist first to enable quick warning & early exit
    p_eos = BENCH_DIR / "eos_bulk" / f"{model_name}_processed.parquet"
    p_ev = BENCH_DIR / "ev" / f"{model_name}_processed.parquet"

    p_stability = BENCH_DIR / "stability" / family
    nvt_files = list(p_stability.glob(f"{model_name}_*nvt.traj"))
    npt_files = list(p_stability.glob(f"{model_name}_*npt.traj"))

    p_combustion = BENCH_DIR / "combustion" / family / f"{model_name}_H256O128.json"

    has_eos = p_eos.exists()
    has_ev = p_ev.exists()
    has_stability = bool(nvt_files or npt_files)
    has_combustion = p_combustion.exists()

    if not (has_eos or has_ev or has_stability or has_combustion):
        warnings.warn(
            f"All expected benchmark output files are missing for model '{model_name}'. "
            f"Expected files at: {p_eos}, {p_ev}, {p_combustion}, or stability traj files in {p_stability}. "
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
        required=True,
        help="Name of the model to generate report/results for.",
    )
    args = parser.parse_args()

    print(f"Generating reports and processed results for model: {args.model}")
    summarize(args.model)
