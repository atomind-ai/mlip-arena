from pathlib import Path

from ase.calculators.calculator import BaseCalculator
from dask.distributed import Client
from dask_jobqueue import SLURMCluster
from huggingface_hub import hf_hub_download
from prefect import flow
from prefect.context import FlowRunContext
from prefect_dask import DaskTaskRunner

from mlip_arena.flows.diatomics import homonuclear_diatomics
from mlip_arena.flows.eos_bulk import run_db as EOSFlow
from mlip_arena.flows.ev import run_db as EVFlow
from mlip_arena.flows.conservation import differential_entropy_along_nve_trajectory
from mlip_arena.flows.stability import compression, heating
from report import summarize, get_model_name


@flow
def asymptotic_behaviors(calculator: str | BaseCalculator, calculator_kwargs: dict | None = None):
    ctx = FlowRunContext.get()
    parent_task_runner = ctx.task_runner

    model_name = get_model_name(calculator)

    from mlip_arena.tasks.utils import get_calculator

    # Use string directly for registered models to avoid client-side unpickleable PyTorch objects
    model = calculator if isinstance(calculator, str) else get_calculator(calculator, calculator_kwargs)

    # 1. Diatomics
    print(f"Starting homonuclear diatomics benchmark for {model_name}...")
    from mlip_arena.models import REGISTRY, MLIPEnum

    if isinstance(calculator, str) and hasattr(MLIPEnum, calculator):
        family = REGISTRY[calculator]["family"]
    elif isinstance(calculator, BaseCalculator) and hasattr(MLIPEnum, calculator.__class__.__name__):
        family = REGISTRY[calculator.__class__.__name__]["family"]
    else:
        family = "custom"
    family = family.lower()
    out_dir_diatomics = Path(__file__).parent / "diatomics" / family / model_name
    homonuclear_diatomics.with_options(name=f"diatomics-{model_name}", task_runner=parent_task_runner)(
        calculator=model, calculator_kwargs=calculator_kwargs, run_dir=out_dir_diatomics
    )

    # 2. EOS Bulk
    print(f"Starting EOS bulk benchmark for {model_name}...")
    out_dir_eos = Path(__file__).parent / "eos_bulk"
    EOSFlow.with_options(name=f"eos_bulk-{model_name}", task_runner=parent_task_runner)(
        model=model, run_dir=out_dir_eos, dataset_file="wbm_subset.db"
    )

    # 3. E-V
    print(f"Starting E-V scan benchmark for {model_name}...")
    out_dir_ev = Path(__file__).parent / "ev"
    EVFlow.with_options(name=f"ev-{model_name}", task_runner=parent_task_runner)(
        model=model, run_dir=out_dir_ev, dataset_file="wbm_subset.db"
    )


@flow
def distribution_shifts(calculator: str | BaseCalculator, calculator_kwargs: dict | None = None):
    ctx = FlowRunContext.get()
    parent_task_runner = ctx.task_runner

    work_dir = Path(__file__).parent / "energy_conservation"

    dH, sampled_structures = differential_entropy_along_nve_trajectory.with_options(
        task_runner=parent_task_runner,
    )(
        calculator=calculator,
        calculator_kwargs=calculator_kwargs,
        input_path=Path(__file__).parent / "mptrj_eq_test.extxyz",
        reference_path=hf_hub_download(
            repo_id="atomind/mlip-arena", filename="mptrj_subset.extxyz", repo_type="dataset"
        ),
        start_idx=0,
        end_idx=-1,
        step=100,
        work_dir=work_dir,
    )


@flow
def stability(calculator: str | BaseCalculator, calculator_kwargs: dict | None = None):
    ctx = FlowRunContext.get()
    parent_task_runner = ctx.task_runner

    model_name = get_model_name(calculator)

    from mlip_arena.models import REGISTRY, MLIPEnum

    if hasattr(MLIPEnum, model_name):
        family = REGISTRY[model_name]["family"]
    else:
        family = "custom"
    family = family.lower()

    run_dir_stability = Path(__file__).parent / "stability" / family

    heating.with_options(
        name=f"stability-heating-{model_name}",
        task_runner=parent_task_runner,
    )(calculator, run_dir_stability)

    compression.with_options(
        name=f"stability-compression-{model_name}",
        task_runner=parent_task_runner,
    )(calculator, run_dir_stability)

    # 3. Analysis / gather results
    print(f"Analyzing stability results for {model_name}...")
    from mlip_arena.flows.stability import gather_results, get_atoms_from_db
    from loguru import logger

    compositions = []
    for atoms in get_atoms_from_db("random-mixture.db"):
        if len(atoms) == 0:
            continue
        compositions.append(atoms.get_chemical_formula())

    try:
        df = gather_results(run_dir_stability, prefix=model_name, run_type="nvt")
        df = df[df["formula"].isin(compositions[:120])].copy()  # tentatively we only take the first 120 structures
        assert len(df) > 0
        df.to_parquet(run_dir_stability / f"{model_name}-heating.parquet", index=False)
    except Exception as e:
        logger.warning(f"Error processing model {model_name} (heating): {e}")

    try:
        df = gather_results(run_dir_stability, prefix=model_name, run_type="npt")
        df = df[df["formula"].isin(compositions[:80])].copy()  # tentatively we only take the first 80 structures
        assert len(df) > 0
        df.to_parquet(run_dir_stability / f"{model_name}-compression.parquet", index=False)
    except Exception as e:
        logger.warning(f"Error processing model {model_name} (compression): {e}")


# ==============================================================================
# 1. JOB CONFIGURATION
# ==============================================================================

# Example A: Registered string model (e.g., "MACE-MP(M)", "CHGNet")
calculator = "NequIP-OAM-L"

# Example B: Custom ASE Calculator class and arguments (e.g., custom MLIP)
# from ase.calculators.lj import LennardJones
# calculator = LennardJones
# calculator_kwargs = dict(rc=10.0, sigma=1.0, epsilon=0.01)

# SLURM environment configuration
SLURM_CONFIG = {
    "nodes": 1,
    "gpus_per_node": 1,
    "account": "matgen",
    "qos": "preempt",
    "walltime": "02:00:00",
    # Example virutal environment (must modify)
    "job_script_prologue": [
        "source ~/.bashrc",
        "module load python",
        "module load cudatoolkit/12.4",
        "source activate /pscratch/sd/c/cyrusyc/.conda/mlip-arena",
    ],
}

job_model_name = get_model_name(calculator)

cluster_kwargs = dict(
    cores=1,
    memory="64 GB",
    processes=1,
    shebang="#!/bin/bash",
    account=SLURM_CONFIG["account"],
    walltime=SLURM_CONFIG["walltime"],
    job_script_prologue=SLURM_CONFIG["job_script_prologue"],
    job_directives_skip=["-n", "--cpus-per-task", "-J"],
    job_extra_directives=[
        f"-J benchmark_{job_model_name}",
        f"-q {SLURM_CONFIG['qos']}",
        f"-N {SLURM_CONFIG['nodes']}",
        "-C gpu",
        f"-G {SLURM_CONFIG['gpus_per_node']}",
        "-o /dev/null",  # Redirect stdout to null
        "-e /dev/null",  # Redirect stderr to null
    ],
)

if __name__ == "__main__":
    cluster = SLURMCluster(**cluster_kwargs)
    print("--------------------------------------------------------------------------------")
    print(f"Generating SLURM cluster jobs with script:\n{cluster.job_script()}")
    print("--------------------------------------------------------------------------------")

    cluster.adapt(minimum_jobs=1, maximum_jobs=50)
    client = Client(cluster)

    print(f"Dask dashboard available at: {client.dashboard_link}")

    # ==============================================================================
    # 2. JOB EXECUTION
    # ==============================================================================

    asymptotic_behaviors.with_options(
        task_runner=DaskTaskRunner(address=client.scheduler.address),
        log_prints=True,
        persist_result=False,
    )(
        calculator=calculator,
        # calculator_kwargs=calculator_kwargs # Uncomment for custom ASE Calculator class
    )

    # distribution_shifts.with_options(
    #     task_runner=DaskTaskRunner(address=client.scheduler.address),
    #     log_prints=True,
    #     persist_result=False,
    # )(
    #     calculator=calculator,
    #     # calculator_kwargs=calculator_kwargs # Uncomment for custom ASE Calculator class
    # )

    stability.with_options(
        task_runner=DaskTaskRunner(address=client.scheduler.address),
        log_prints=True,
        persist_result=False,
    )(
        calculator=calculator,
        # calculator_kwargs=calculator_kwargs # Uncomment for custom ASE Calculator class
    )

    # 3. Report Generation
    print(f"Generating reports and processed results for {calculator}...")
    summarize(calculator)

    # TODO: Add Reactivity
