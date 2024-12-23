from functools import partial
from pathlib import Path

import pandas as pd
from huggingface_hub import hf_hub_download
from prefect import Task, flow, task
from prefect.client.schemas.objects import TaskRun
from prefect.futures import wait
from prefect.states import State

from ase.db import connect
from mlip_arena.data.local import SafeHDFStore
from mlip_arena.models import REGISTRY, MLIPEnum
from mlip_arena.tasks.eos import run as EOS


@task
def get_atoms_from_db(db_path: Path | str):
    db_path = Path(db_path)
    if not db_path.exists():
        db_path = hf_hub_download(
            repo_id="atomind/mlip-arena",
            repo_type="dataset",
            subfolder=f"{Path(__file__).parent.name}",
            filename=str(db_path),
        )
    with connect(db_path) as db:
        for row in db.select():
            yield row.toatoms()


def save_to_hdf(
    tsk: Task, run: TaskRun, state: State, fpath: Path | str, table_name: str
):
    """
    Define a hook on completion of EOS task to save results to HDF5 file.
    """

    if run.state.is_failed():
        return

    result = run.state.result(raise_on_failure=False)

    if not isinstance(result, dict):
        return

    try:
        atoms = result["atoms"]
        calculator_name = (
            run.task_inputs["calculator_name"] or result["calculator_name"]
        )

        energies = [float(e) for e in result["eos"]["energies"]]

        formula = atoms.get_chemical_formula()

        df = pd.DataFrame(
            {
                "method": calculator_name,
                "formula": formula,
                "total_run_time": run.total_run_time,
                "v0": result["v0"],
                "e0": result["e0"],
                "b0": result["b0"],
                "b1": result["b1"],
                "volume": result["eos"]["volumes"],
                "energy": energies,
            }
        )

        fpath = Path(fpath)
        fpath = fpath.with_stem(fpath.stem + f"_{calculator_name}")

        family_path = Path(__file__).parent / REGISTRY[calculator_name]["family"]
        family_path.mkdir(parents=True, exist_ok=True)

        df.to_json(family_path / f"{calculator_name}_{formula}.json", indent=2)

        with SafeHDFStore(fpath, mode="a") as store:
            store.append(
                table_name,
                df,
                format="table",
                data_columns=True,
                min_itemsize={"formula": 50, "method": 20},
            )
    except Exception as e:
        print(e)


@flow(
    name="EOS Alloy"
)
def run(
    db_path: Path | str,
    out_path: Path | str,
    table_name: str,
    optimizer="FIRE",
    optimizer_kwargs=None,
    filter="FrechetCell",
    filter_kwargs=None,
    criterion=dict(fmax=0.1, steps=1000),
    max_abs_strain=0.20,
    concurrent=False,
    cache=True,
):
    EOS_ = EOS.with_options(
        on_completion=[partial(save_to_hdf, fpath=out_path, table_name=table_name)],
        refresh_cache=not cache,
    )

    futures = []
    for atoms in get_atoms_from_db(db_path):
        for mlip in MLIPEnum:
            if not REGISTRY[mlip.name]["npt"]:
                continue
            if Path(__file__).parent.name not in (
                REGISTRY[mlip.name].get("cpu-tasks", [])
                + REGISTRY[mlip.name].get("gpu-tasks", [])
            ):
                continue
            future = EOS_.submit(
                atoms=atoms,
                calculator_name=mlip.name,
                calculator_kwargs=dict(),
                optimizer=optimizer,
                optimizer_kwargs=optimizer_kwargs,
                filter=filter,
                filter_kwargs=filter_kwargs,
                criterion=criterion,
                max_abs_strain=max_abs_strain,
                concurrent=concurrent,
                persist_opt=cache,
                cache_opt=cache,
                # return_state=True
            )
            futures.append(future)

    wait(futures)

    return [
        f.result(timeout=None, raise_on_failure=False)
        for f in futures
        if f.state.is_completed()
    ]
