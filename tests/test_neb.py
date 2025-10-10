import sys

import pytest
from ase.spacegroup import crystal
from prefect.testing.utilities import prefect_test_harness

from mlip_arena.models import MLIPEnum
from mlip_arena.tasks import NEB_FROM_ENDPOINTS
from mlip_arena.tasks.utils import get_calculator

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
        result = NEB_FROM_ENDPOINTS(
            start=start.copy(),
            end=end.copy(),
            n_images=5,
            calculator=get_calculator(
                model.name,
            ),
            optimizer="BFGS",
        )

        assert isinstance(result, dict)
        assert isinstance(result["barrier"][0], float)
        assert isinstance(result["barrier"][1], float)
