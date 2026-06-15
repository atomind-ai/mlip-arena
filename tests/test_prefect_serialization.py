# ruff: noqa: E402
import os

os.environ["PREFECT_SERVER_EPHEMERAL_STARTUP_TIMEOUT_SECONDS"] = "240"

import pytest
from ase.build import molecule
from prefect import flow
from prefect.testing.utilities import prefect_test_harness

from mlip_arena.tasks.mof.flow import widom_insertion
from mlip_arena.tasks.mof.input import get_atoms_from_db


@pytest.fixture(autouse=True, scope="session")
def prefect_test_fixture():
    with prefect_test_harness():
        yield


@pytest.mark.nequip
def test_prefect_serialization_nequip():
    """Verify that submitting a Prefect task/flow with a NequIP calculator
    (unpickleable on client side) works correctly by passing string identifier
    so that instantiation occurs inside the task execution boundary.
    """

    @flow
    def test_flow():
        for atoms in get_atoms_from_db("mofs.db"):
            # Submit the task to trigger Prefect's serialization/pickling logic
            future = widom_insertion.submit(
                structure=atoms,
                gas=molecule("CO2"),
                calculator="NequIP-OAM-L",
                num_insertions=2,
                fold=1,
            )
            return future.result()

    result = test_flow()
    assert isinstance(result, dict)
    assert isinstance(result["henry_coefficient"][0], float)
    assert isinstance(result["averaged_interaction_energy"][0], float)
    assert isinstance(result["heat_of_adsorption"][0], float)
