import pytest
from mlip_arena.models import MLIPEnum
from mlip_arena.tasks.mof.flow import widom_insertion
from mlip_arena.tasks.mof.input import get_atoms_from_db
from prefect.testing.utilities import prefect_test_harness

from ase.build import molecule


@pytest.mark.parametrize("model", [MLIPEnum["MACE-MP(M)"]])
def test_widom_insertion(model: MLIPEnum):
    atoms = get_atoms_from_db("mof.db")
    with prefect_test_harness():
        result = widom_insertion.with_options(
            refresh_cache=True,
        )(
            structure=atoms,
            gas=molecule("CO2"),
            calculator_name=model.name,
            num_insertions=10,
        )
        assert isinstance(result, dict)
        assert isinstance(result["henry_coefficient"][0], float)
        assert isinstance(result["averaged_interaction_energy"][0], float)
        assert isinstance(result["heat_of_adsorption"][0], float)