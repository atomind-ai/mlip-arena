import sys

import pytest
from mlip_arena.models import MLIPEnum
from mlip_arena.tasks.neb import run as NEB
from mlip_arena.tasks.neb import run_from_end_points as NEB
from prefect.testing.utilities import prefect_test_harness

from ase.spacegroup import crystal

pristine = crystal(
    "Al", [(0, 0, 0)], spacegroup=225, cellpar=[4.05, 4.05, 4.05, 90, 90, 90]
) * (3, 3, 3)

atoms = pristine.copy()
del atoms[0]
start = atoms.copy()

atoms = pristine.copy()
del atoms[1]
end = atoms.copy()


@pytest.mark.skipif(
    sys.version_info[:2] != (3, 11),
    reason="avoid prefect race condition on concurrent tasks",
)
@pytest.mark.parametrize("model", [MLIPEnum["MACE-MP(M)"]])
def test_neb(model: MLIPEnum):
    """
    Test NEB prefect workflow with a simple cubic lattice.
    """

    with prefect_test_harness():
        result = NEB(
            start=start.copy(),
            end=end.copy(),
            n_images=5,
            calculator_name=model.name,
            optimizer="FIRE2",
        )

        assert isinstance(result, dict)
        assert isinstance(result["barrier"][0], float)
        assert isinstance(result["barrier"][1], float)
