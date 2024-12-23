
from .task import (
    get_atoms,
    get_thermal_conductivity
)
from mlip_arena.models import MLIPEnum
from prefect import flow



@flow(
    name="Thermal Conductivity",
)
def run():

    states = []
    for model in MLIPEnum:
        for atoms in get_atoms():

            state = get_thermal_conductivity(
                atoms=atoms,
                calculator_name=model.name,
                calculator_kwargs={},
                return_state=True,
            )
            states.append(state)

    return [s.result(raise_on_failure=False) for s in states if s.is_completed()]
