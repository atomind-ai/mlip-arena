import sys

import pytest
from ase.build import bulk
from prefect import flow
from prefect.testing.utilities import prefect_test_harness

from mlip_arena.models import MLIPEnum
from mlip_arena.tasks.eos import run as EOS


@flow
def single_eos_flow(calculator_name):
    atoms = bulk("Cu", "fcc", a=3.6)

    return EOS(
        atoms=atoms,
        calculator_name=calculator_name,
        calculator_kwargs={},
        device=None,
        optimizer="BFGSLineSearch",
        optimizer_kwargs=None,
        filter="FrechetCell",
        filter_kwargs=None,
        criterion=dict(
            fmax=0.1,
        ),
        max_abs_strain=0.1,
        npoints=6,
    )


@pytest.mark.skipif(
    sys.version_info[:2] != (3, 11),
    reason="avoid prefect race condition on concurrent tasks",
)
@pytest.mark.parametrize("model", [MLIPEnum["MACE-MP(M)"]])
def test_eos(model: MLIPEnum):
    """
    Test EOS prefect workflow with a simple cubic lattice.
    """

    with prefect_test_harness():
        result = single_eos_flow(
            calculator_name=model.name,
        )

        assert isinstance(result["b0"], float)
