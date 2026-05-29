from __future__ import annotations

from pathlib import Path

from ase.calculators.calculator import BaseCalculator
from loguru import logger
from prefect import flow
from prefect.context import FlowRunContext

from mlip_arena.flows.stability import compression, heating
from mlip_arena.models import REGISTRY, MLIPEnum
from mlip_arena.tasks.stability.analysis import gather_results
from mlip_arena.tasks.stability.data import get_atoms_from_db


@flow
def random_mixtures(calculator: str | BaseCalculator, calculator_kwargs: dict | None = None):
    ctx = FlowRunContext.get()
    parent_task_runner = ctx.task_runner if ctx else None

    if isinstance(calculator, BaseCalculator):
        model_name = calculator.__class__.__name__
        model = calculator
        family = "custom"
    elif isinstance(calculator, str) and hasattr(MLIPEnum, calculator):
        model_name = calculator
        model = MLIPEnum[calculator]
        family = REGISTRY[model_name]["family"]
    else:
        raise ValueError(f"Unsupported model: {calculator}")

    run_dir = Path(__file__).parent / "stability" / family

    # 1. Run heating
    print(f"Starting stability heating simulation for {model_name}...")
    heating.with_options(
        name=f"stability-heating-{model_name}",
        task_runner=parent_task_runner,
        log_prints=True,
    )(model, run_dir)

    # 2. Run compression
    print(f"Starting stability compression simulation for {model_name}...")
    compression.with_options(
        name=f"stability-compression-{model_name}",
        task_runner=parent_task_runner,
        log_prints=True,
    )(model, run_dir)

    # 3. Analysis / gather results
    print(f"Analyzing stability results for {model_name}...")
    compositions = []
    for atoms in get_atoms_from_db("random-mixture.db"):
        if len(atoms) == 0:
            continue
        compositions.append(atoms.get_chemical_formula())

    try:
        df = gather_results(run_dir, prefix=model_name, run_type="nvt")
        df = df[df["formula"].isin(compositions[:120])].copy()  # tentatively we only take the first 120 structures
        assert len(df) > 0
        df.to_parquet(run_dir / f"{model_name}-heating.parquet", index=False)
    except Exception as e:
        logger.warning(f"Error processing model {model_name} (heating): {e}")

    try:
        df = gather_results(run_dir, prefix=model_name, run_type="npt")
        df = df[df["formula"].isin(compositions[:80])].copy()  # tentatively we only take the first 80 structures
        assert len(df) > 0
        df.to_parquet(run_dir / f"{model_name}-compression.parquet", index=False)
    except Exception as e:
        logger.warning(f"Error processing model {model_name} (compression): {e}")


if __name__ == "__main__":
    from dask.distributed import Client
    from dask_jobqueue import SLURMCluster
    from prefect_dask import DaskTaskRunner

    nodes_per_alloc = 1
    gpus_per_alloc = 4

    cluster_kwargs = dict(
        cores=1,
        memory="64 GB",
        processes=1,
        shebang="#!/bin/bash",
        account="matgen",
        walltime="04:00:00",
        job_mem="0",
        job_script_prologue=[
            "source ~/.bashrc",
            "module load python",
            "source activate /pscratch/sd/c/cyrusyc/.conda/mlip-arena",
        ],
        job_directives_skip=["-n", "--cpus-per-task", "-J"],
        job_extra_directives=[
            "-J arena-stability",
            "-q preempt",
            "--time-min=00:30:00",
            "--comment=12:00:00",
            f"-N {nodes_per_alloc}",
            "-C gpu",
            f"-G {gpus_per_alloc}",
        ],
    )

    cluster = SLURMCluster(**cluster_kwargs)
    print(cluster.job_script())
    cluster.adapt(minimum_jobs=10, maximum_jobs=50)
    client = Client(cluster)

    for model in MLIPEnum:
        random_mixtures.with_options(
            task_runner=DaskTaskRunner(address=client.scheduler.address),
            log_prints=True,
        )(model.name)
