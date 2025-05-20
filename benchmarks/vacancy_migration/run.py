import pickle
from functools import partial
from pathlib import Path

from ase import Atoms
from prefect import Task, flow, task
from prefect.client.schemas.objects import TaskRun
from prefect.results import ResultRecord
from prefect.states import State

from mlip_arena.models import REGISTRY, MLIPEnum
from mlip_arena.tasks.eos import run as EOS
from mlip_arena.tasks.neb import run_from_endpoints as NEB
from mlip_arena.tasks.vacancy_migration.input import get_fcc_pristine, get_hcp_pristine

MP_API_KEY = None

test_models = ["MACE-MP(M)", "MatterSim", "ORBv2", "CHGNet", "M3GNet", "SevenNet"]


def save_to_pickle(
    tsk: Task, run: TaskRun, state: State, crystal: str
):
    result = run.state.result(raise_on_failure=False)

    pristine = result["pristine"]
    calculator_name = result["calculator_name"]
    calculator_name = calculator_name.name if isinstance(calculator_name, MLIPEnum) else calculator_name

    family_path = Path(REGISTRY[calculator_name]["family"])
    family_path.mkdir(parents=True, exist_ok=True)

    with open(family_path / f"{calculator_name}-{crystal}-{pristine.get_chemical_formula()}.pkl", "wb") as f:
        pickle.dump(result, f)

    # with open(family_path / f"{crystal}-{pristine.get_chemical_formula()}.json", 'w') as f:
    #     json.dump(result, f)

@task
def calculate_vacancy_migration(
    pristine: Atoms,
    istart: int,
    iend: int,
    calculator_name: MLIPEnum | str,
    optimizer: str,
    criterion: dict = {}
):

    eos = EOS.with_options(refresh_cache=True, persist_result=True)(
        atoms=pristine,
        calculator_name=calculator_name,
        optimizer=optimizer,
        criterion=criterion,
        concurrent=False,
    )

    if isinstance(eos, ResultRecord):
        eos = eos.result

    if isinstance(eos, dict):
        pristine = eos["atoms"]
    else:
        return eos

    atoms = pristine.copy()
    del atoms[istart]
    start = atoms.copy()

    atoms = pristine.copy()
    del atoms[iend]
    end = atoms.copy()


    neb = NEB.with_options(refresh_cache=True, persist_result=True)(
        start, end, n_images=7,
        calculator_name=calculator_name,
        optimizer=optimizer,
        criterion=criterion,
        relax_end_points=True
    )

    e_defect = 0.5 * (neb["images"][0].get_potential_energy() + neb["images"][-1].get_potential_energy())
    e_pristine = pristine.get_potential_energy()

    e_vacform = e_defect - (len(neb["images"][0]) / len(pristine)) * e_pristine

    e_vacmig = neb["barrier"][0]
    asymmetry = abs(neb["barrier"][1] / e_vacmig)

    # TODO: temporary solution to pickling problem of mattersim
    pristine.calc = None
    for image in neb["images"]:
        image.calc = None
    eos["atoms"].calc = None

    return {
        "pristine": pristine,
        "calculator_name": calculator_name,
        "e_vacform": e_vacform,
        "e_vacmig": e_vacmig,
        "asymmetry": asymmetry,
        "neb": neb,
        "eos": eos
    }

@flow(persist_result=True, result_serializer="pickle")
def run_fcc():

    futures = []
    for atoms in get_fcc_pristine(MP_API_KEY):
        for model in MLIPEnum:
            if model.name not in test_models:
                continue
            try:
                result = calculate_vacancy_migration.with_options(
                    refresh_cache=True, persist_result=True,
                    on_completion=[partial(save_to_pickle, crystal="fcc")]
                )(
                    pristine=atoms,
                    istart=0,
                    iend=1,
                    calculator_name=model,
                    optimizer="BFGS",
                    criterion=dict(fmax=0.05, steps=500),
                )
            except Exception:
                continue
            futures.append(result)

    return futures
    # wait(futures)
    # return [f.result(raise_on_failure=False) for f in futures if f.state.is_completed()]

@flow(persist_result=True, result_serializer="pickle")
def run_hcp():
    futures = []
    for i, atoms in enumerate(get_hcp_pristine(MP_API_KEY)):
        if i <= 30:
            continue
        for model in MLIPEnum:
            if model.name not in test_models:
                continue
            try:
                result = calculate_vacancy_migration.with_options(
                    refresh_cache=True, persist_result=True,
                    on_completion=[partial(save_to_pickle, crystal="hcp")]
                )(
                    pristine=atoms,
                    istart=0,
                    iend=1,
                    calculator_name=model,
                    optimizer="BFGS",
                    criterion=dict(fmax=0.05, steps=500),
                )
                # calculator_name = model.name if isinstance(model, MLIPEnum) else model

                # family_path = Path(REGISTRY[calculator_name]['family'])
                # family_path.mkdir(parents=True, exist_ok=True)
                # with open(family_path / f"{'hcp'}-{atoms.get_chemical_formula()}.pkl", 'wb') as f:
                #     pickle.dump(result, f)
            except Exception:
                continue
            futures.append(result)

    return futures
    # wait(futures)
    # return [f.result(raise_on_failure=False) for f in futures if f.state.is_completed()]
