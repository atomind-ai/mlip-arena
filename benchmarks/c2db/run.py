from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
from dask.distributed import Client
from dask_jobqueue import SLURMCluster
from mlip_arena.models import MLIPEnum
from mlip_arena.tasks import ELASTICITY, OPT, PHONON
from mlip_arena.tasks.optimize import run as OPT
from mlip_arena.tasks.utils import get_calculator
from numpy import linalg as LA
from prefect import flow, task
from prefect_dask import DaskTaskRunner
from tqdm.auto import tqdm

from ase.db import connect

select_models = [
    "ALIGNN",
    "CHGNet",
    "M3GNet",
    "MACE-MP(M)",
    "MACE-MPA",
    "MatterSim",
    "ORBv2",
    "SevenNet",
]

def elastic_tensor_to_voigt(C):
    """
    Convert a rank-4 (3x3x3x3) elastic tensor into a rank-2 (6x6) tensor using Voigt notation.

    Parameters:
    C (numpy.ndarray): A 3x3x3x3 elastic tensor.

    Returns:
    numpy.ndarray: A 6x6 elastic tensor in Voigt notation.
    """
    # voigt_map = {
    #     (0, 0): 0, (1, 1): 1, (2, 2): 2,  # Normal components
    #     (1, 2): 3, (2, 1): 3,  # Shear components
    #     (0, 2): 4, (2, 0): 4,
    #     (0, 1): 5, (1, 0): 5
    # }
    voigt_map = {
        (0, 0): 0,
        (1, 1): 1,
        (2, 2): -1,  # Normal components
        (1, 2): -1,
        (2, 1): -1,  # Shear components
        (0, 2): -1,
        (2, 0): -1,
        (0, 1): 2,
        (1, 0): 2,
    }

    C_voigt = np.zeros((3, 3))

    for i in range(3):
        for j in range(3):
            for k in range(3):
                for l in range(3):
                    alpha = voigt_map[(i, j)]
                    beta = voigt_map[(k, l)]

                    if alpha == -1 or beta == -1:
                        continue

                    factor = 1
                    # if alpha in [3, 4, 5]:
                    if alpha == 2:
                        factor = factor * (2**0.5)
                    if beta == 2:
                        factor = factor * (2**0.5)

                    C_voigt[alpha, beta] = C[i, j, k, l] * factor

    return C_voigt


# -


@task
def run_one(model, row):
    if Path(f"{model.name}.pkl").exists():
        df = pd.read_pickle(f"{model.name}.pkl")

        # if row.key_value_pairs.get('uid', None) in df['uid'].unique():
        #     pass
    else:
        df = pd.DataFrame(columns=["model", "uid", "eigenvalues", "frequencies"])

    atoms = row.toatoms()
    # print(data := row.key_value_pairs)

    calc = get_calculator(model)

    result_opt = OPT(
        atoms,
        calc,
        optimizer="FIRE",
        criterion=dict(fmax=0.05, steps=500),
        symmetry=True,
    )

    atoms = result_opt["atoms"]

    result_elastic = ELASTICITY(
        atoms,
        calc,
        optimizer="FIRE",
        criterion=dict(fmax=0.05, steps=500),
        pre_relax=False,
    )

    elastic_tensor = elastic_tensor_to_voigt(result_elastic["elastic_tensor"])
    eigenvalues, eigenvectors = LA.eig(elastic_tensor)

    outdir = Path(f"{model.name}") / row.key_value_pairs.get(
        "uid", atoms.get_chemical_formula()
    )
    outdir.mkdir(parents=True, exist_ok=True)

    np.savez(outdir / "elastic.npz", tensor=elastic_tensor, eigenvalues=eigenvalues)

    result_phonon = PHONON(
        atoms,
        calc,
        supercell_matrix=(2, 2, 1),
        outdir=outdir,
    )

    frequencies = result_phonon["phonon"].get_frequencies(q=(0, 0, 0))

    new_row = pd.DataFrame(
        [
            {
                "model": model.name,
                "uid": row.key_value_pairs.get("uid", None),
                "eigenvalues": eigenvalues,
                "frequencies": frequencies,
            }
        ]
    )

    df = pd.concat([df, new_row], ignore_index=True)
    df.drop_duplicates(subset=["model", "uid"], keep="last", inplace=True)

    df.to_pickle(f"{model.name}.pkl")


@flow
def run_all():
    import random

    random.seed(0)

    futures = []
    with connect("c2db.db") as db:
        random_indices = random.sample(range(1, len(db) + 1), 1000)
        for row, model in tqdm(
            product(db.select(filter=lambda r: r["id"] in random_indices), MLIPEnum)
        ):
            if model.name not in select_models:
                continue
            future = run_one.submit(model, row)
            futures.append(future)
    return [f.result(raise_on_failure=False) for f in futures]


# +


if __name__ == "__main__":
    nodes_per_alloc = 1
    gpus_per_alloc = 1
    ntasks = 1

    cluster_kwargs = dict(
        cores=1,
        memory="64 GB",
        processes=1,
        shebang="#!/bin/bash",
        account="matgen",
        walltime="00:30:00",
        # job_cpu=128,
        job_mem="0",
        job_script_prologue=[
            "source ~/.bashrc",
            "module load python",
            "source activate /pscratch/sd/c/cyrusyc/.conda/dev",
        ],
        job_directives_skip=["-n", "--cpus-per-task", "-J"],
        job_extra_directives=[
            "-J c2db",
            "-q regular",
            f"-N {nodes_per_alloc}",
            "-C gpu",
            f"-G {gpus_per_alloc}",
        ],
    )

    cluster = SLURMCluster(**cluster_kwargs)
    print(cluster.job_script())
    cluster.adapt(minimum_jobs=25, maximum_jobs=50)
    client = Client(cluster)
    # -

    run_all.with_options(
        task_runner=DaskTaskRunner(address=client.scheduler.address), log_prints=True
    )()
