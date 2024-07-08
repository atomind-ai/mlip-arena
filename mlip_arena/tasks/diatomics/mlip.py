from datetime import timedelta
from typing import Union

# import covalent as ct
import numpy as np
import pandas as pd
import torch
from ase import Atoms
from ase.calculators.calculator import Calculator
from ase.data import chemical_symbols
from dask.distributed import Client
from dask_jobqueue import SLURMCluster
from prefect import flow, task
from prefect.tasks import task_input_hash
from prefect_dask import DaskTaskRunner

from mlip_arena.models import MLIPCalculator
from mlip_arena.models.utils import EXTMLIPEnum, MLIPMap, external_ase_calculator

cluster_kwargs = {
    "cores": 4,
    "memory": "64 GB",
    "shebang": "#!/bin/bash",
    "account": "m3828",
    "walltime": "00:10:00",
    "job_mem": "0",
    "job_script_prologue": ["source ~/.bashrc"],
    "job_directives_skip": ["-n", "--cpus-per-task"],
    "job_extra_directives": ["-q debug", "-C gpu"],
}

cluster = SLURMCluster(**cluster_kwargs)
cluster.scale(jobs=10)
client = Client(cluster)


@task(cache_key_fn=task_input_hash, cache_expiration=timedelta(hours=1))
def calculate_single_diatomic(
    calculator_name: str | EXTMLIPEnum,
    calculator_kwargs: dict | None,
    atom1: str,
    atom2: str,
    rmin: float = 0.1,
    rmax: float = 6.5,
    npts: int = int(1e3),
):

    calculator_kwargs = calculator_kwargs or {}

    if isinstance(calculator_name, EXTMLIPEnum) and calculator_name in EXTMLIPEnum:
        calc = external_ase_calculator(calculator_name, **calculator_kwargs)
    elif calculator_name in MLIPMap:
        calc = MLIPMap[calculator_name](**calculator_kwargs)

    a = 2 * rmax

    rs = np.linspace(rmin, rmax, npts)
    e = np.zeros_like(rs)
    f = np.zeros_like(rs)

    da = atom1 + atom2

    for i, r in enumerate(rs):

        positions = [
            [0, 0, 0],
            [r, 0, 0],
        ]

        # Create the unit cell with two atoms
        atoms = Atoms(da, positions=positions, cell=[a, a, a])

        atoms.calc = calc

        e[i] = atoms.get_potential_energy()
        f[i] = np.inner(np.array([1, 0, 0]), atoms.get_forces()[1])

    return {"r": rs, "E": e, "F": f, "da": da}


@flow
def calculate_multiple_diatomics(calculator_name, calculator_kwargs):

    futures = []
    for symbol in chemical_symbols:
        if symbol == "X":
            continue
        futures.append(
            calculate_single_diatomic.submit(
                calculator_name, calculator_kwargs, symbol, symbol
            )
        )

    return [i for future in futures for i in future.result()]


@flow(task_runner=DaskTaskRunner(address=client.scheduler.address), log_prints=True)
def calculate_homonuclear_diatomics(calculator_name, calculator_kwargs):

    curves = calculate_multiple_diatomics(calculator_name, calculator_kwargs)

    pd.DataFrame(curves).to_csv(f"homonuclear-diatomics-{calculator_name}.csv")


# with plt.style.context("default"):

#     SMALL_SIZE = 6
#     MEDIUM_SIZE = 8
#     LARGE_SIZE = 10

#     LINE_WIDTH = 1

#     plt.rcParams.update(
#         {
#             "pgf.texsystem": "pdflatex",
#             "font.family": "sans-serif",
#             "text.usetex": True,
#             "pgf.rcfonts": True,
#             "figure.constrained_layout.use": True,
#             "axes.labelsize": MEDIUM_SIZE,
#             "axes.titlesize": MEDIUM_SIZE,
#             "legend.frameon": False,
#             "legend.fontsize": MEDIUM_SIZE,
#             "legend.loc": "best",
#             "lines.linewidth": LINE_WIDTH,
#             "xtick.labelsize": SMALL_SIZE,
#             "ytick.labelsize": SMALL_SIZE,
#         }
#     )

#     fig, ax = plt.subplots(layout="constrained", figsize=(3, 2), dpi=300)

#     color = "tab:red"
#     ax.plot(rs, e, color=color, zorder=1)

#     ax.axhline(ls="--", color=color, alpha=0.5, lw=0.5 * LINE_WIDTH)

#     ylo, yhi = ax.get_ylim()
#     ax.set(xlabel=r"r [$\AA]$", ylim=(max(-7, ylo), min(5, yhi)))
#     ax.set_ylabel(ylabel="E [eV]", color=color)
#     ax.tick_params(axis="y", labelcolor=color)
#     ax.text(0.8, 0.85, da, fontsize=LARGE_SIZE, transform=ax.transAxes)

#     color = "tab:blue"

#     at = ax.twinx()
#     at.plot(rs, f, color=color, zorder=0, lw=0.5 * LINE_WIDTH)

#     at.axhline(ls="--", color=color, alpha=0.5, lw=0.5 * LINE_WIDTH)

#     ylo, yhi = at.get_ylim()
#     at.set(
#         xlabel=r"r [$\AA]$",
#         ylim=(max(-20, ylo), min(20, yhi)),
#     )
#     at.set_ylabel(ylabel="F [eV/$\AA$]", color=color)
#     at.tick_params(axis="y", labelcolor=color)

#     plt.show()


if __name__ == "__main__":
    calculate_homonuclear_diatomics(
        EXTMLIPEnum.MACE, dict(model="medium", device="cuda")
    )
