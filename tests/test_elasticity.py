import sys

import numpy as np
import pytest
from mlip_arena.models import MLIPEnum
from mlip_arena.tasks.elasticity import run as ELASTICITY
from mlip_arena.tasks.utils import get_calculator
from prefect.testing.utilities import prefect_test_harness

from ase.build import bulk


@pytest.mark.skipif(
    sys.version_info[:2] != (3, 11),
    reason="avoid prefect race condition on concurrent tasks",
)
@pytest.mark.parametrize("model", [MLIPEnum["MACE-MP(M)"]])
def test_elasticity(model: MLIPEnum):
    """
    Test elasticity prefect workflow with a simple cubic lattice.
    """

    with prefect_test_harness():
        result = ELASTICITY(
            atoms=bulk("Cu", "fcc", a=3.6),
            calculator=get_calculator(
                calculator_name=model.name,
            ),
            optimizer="BFGSLineSearch",
            optimizer_kwargs=None,
            filter="FrechetCell",
            filter_kwargs=None,
            criterion=None,
            persist_opt=False,
            cache_opt=False,
        )
        assert isinstance(result, dict)
        assert isinstance(result["elastic_tensor"], np.ndarray)
        assert result["elastic_tensor"].shape == (3, 3, 3, 3)
        assert isinstance(result["elastic_tensor"][0, 0, 0, 0], float)
