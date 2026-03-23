import argparse
from pathlib import Path

from dask.distributed import Client
from dask_jobqueue import SLURMCluster
from prefect import flow
from prefect_dask import DaskTaskRunner

from mlip_arena.flows.diatomics import homonuclear_diatomics
from mlip_arena.flows.eos_bulk import wbm_eos_bulk
from mlip_arena.flows.wbm_ev import wbm_ev_scan
from mlip_arena.models import REGISTRY, MLIPEnum


@flow
def benchmark_model(model_name: str, db_path: str):
    """Run all core benchmarks sequentially for a given model."""
    if model_name not in MLIPEnum.__members__:
        print(f"Model {model_name} is not in MLIPEnum natively. Using directly.")

    family = REGISTRY[model_name]["family"] if model_name in REGISTRY else "custom"

    # 1. Diatomics
    print(f"Starting homonuclear diatomics benchmark for {model_name}...")
    # NOTE: diatomics flow does not iterate over db_path, it generates molecules
    # and internally writes {model_name}.json to the parents of the output directory
    # to maintain compatibility with `serve/rank/homonuclear-diatomics.py`
    out_dir_diatomics = Path.cwd() / "diatomics" / family / model_name
    homonuclear_diatomics.with_options(
        name=f"diatomics-{model_name}", refresh_cache=True
    )(model=model_name, run_dir=out_dir_diatomics)

    # 2. EOS Bulk
    print(f"Starting EOS bulk benchmark for {model_name}...")
    out_dir_eos = Path.cwd() / "eos_bulk" / model_name
    wbm_eos_bulk.with_options(name=f"eos_bulk-{model_name}", refresh_cache=True)(
        model=model_name, run_dir=out_dir_eos, db_path=db_path
    )

    # 3. WBM E-V
    print(f"Starting WBM E-V scan benchmark for {model_name}...")
    out_dir_ev = Path.cwd() / "wbm_ev" / model_name
    wbm_ev_scan.with_options(name=f"wbm_ev-{model_name}", refresh_cache=True)(
        model=model_name, run_dir=out_dir_ev, db_path=db_path
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Submit MLIP Arena benchmarks sequentially for a model."
    )
    parser.add_argument(
        "--model", type=str, required=True, help="Registered model name in MLIPEnum."
    )
    parser.add_argument(
        "--db-path",
        type=str,
        default="wbm_structures.db",
        help="Path to wbm_structures.db. Must reside under benchmarks folder typically.",
    )
    parser.add_argument("--nodes", type=int, default=1, help="Number of nodes.")
    parser.add_argument("--gpus", type=int, default=1, help="Number of GPUs per node.")
    parser.add_argument(
        "--account", type=str, default="m3828", help="SLURM account allocation."
    )
    parser.add_argument(
        "--qos", type=str, default="regular", help="SLURM QoS (regular/debug)."
    )
    parser.add_argument(
        "--walltime", type=str, default="04:00:00", help="Job walltime."
    )

    args = parser.parse_args()

    cluster_kwargs = dict(
        cores=1,
        memory="64 GB",
        processes=1,
        shebang="#!/bin/bash",
        account=args.account,
        walltime=args.walltime,
        job_script_prologue=[
            "source ~/.bashrc",
            "module load python",
            "module load cudatoolkit/12.4",
            "source activate /pscratch/sd/c/cyrusyc/.conda/mlip-arena",
        ],
        job_directives_skip=["-n", "--cpus-per-task", "-J"],
        job_extra_directives=[
            f"-J benchmark_{args.model}",
            f"-q {args.qos}",
            f"-N {args.nodes}",
            "-C gpu",
            f"-G {args.gpus}",
            # "--exclusive",
        ],
    )

    cluster = SLURMCluster(**cluster_kwargs)
    print("--------------------------------------------------")
    print(f"Generating SLURM cluster jobs with script:\n{cluster.job_script()}")
    print("--------------------------------------------------")

    # Scale appropriately to minimum jobs or let it adapt
    cluster.adapt(minimum_jobs=1, maximum_jobs=50)
    client = Client(cluster)

    print(f"Dask dashboard available at: {client.dashboard_link}")

    # Launch Prefect flow using Dask task runner
    # The run_dir logic expects this script to be run from `mlip-arena/benchmarks/` root
    benchmark_model.with_options(
        task_runner=DaskTaskRunner(address=client.scheduler.address),
        log_prints=True,
    )(model_name=args.model, db_path=args.db_path)
