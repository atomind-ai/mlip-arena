import sys

import pytest
from ase.build import molecule
from prefect.testing.utilities import prefect_test_harness

from mlip_arena.models import MLIPEnum
from mlip_arena.tasks.mof.flow import widom_insertion
from mlip_arena.tasks.mof.input import get_atoms_from_db
from mlip_arena.tasks.utils import get_calculator


@pytest.fixture(autouse=True, scope="session")
def prefect_test_fixture():
    with prefect_test_harness():
        yield


@pytest.mark.skipif(
    sys.version_info[:2] != (3, 11),
    reason="avoid prefect race condition on concurrent tasks",
)
@pytest.mark.parametrize("model", [MLIPEnum["MACE-MP(M)"]])
def test_widom_insertion(model: MLIPEnum):
    # with prefect_test_harness():
    for atoms in get_atoms_from_db("mofs.db"):
        result = widom_insertion.with_options(
            refresh_cache=True,
        )(
            structure=atoms,
            gas=molecule("CO2"),
            calculator=get_calculator(
                model,
                dispersion=True,
            ),
            num_insertions=10,
            fold=2,
        )
        assert isinstance(result, dict)
        assert isinstance(result["henry_coefficient"][0], float)
        assert isinstance(result["averaged_interaction_energy"][0], float)
        assert isinstance(result["heat_of_adsorption"][0], float)
        break  # only test one MOF
