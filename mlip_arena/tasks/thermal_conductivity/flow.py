
from pathlib import Path

from prefect import flow
from prefect.futures import wait
from mlip_arena.models import MLIPEnum

from .input import get_atoms_from_db, get_atoms_from_extxyz
from .task import get_thermal_conductivity


@flow(
    name="Thermal Conductivity",
)
def run(
    db_path: str | Path = "phononDB-PBE-structures.extxyz"
):
    """Run the thermal conductivity task on all structures in the database."""

    db_path = Path(db_path)
    states = []
    for model in MLIPEnum:
        for atoms in get_atoms_from_extxyz(db_path):

            state = get_thermal_conductivity.submit(
                atoms=atoms,
                calculator_name=model.name,
                calculator_kwargs={},
                return_state=True,
            )
            states.append(state)

    wait(states)

    return [s.result(raise_on_failure=False) for s in states if s.is_completed()]
