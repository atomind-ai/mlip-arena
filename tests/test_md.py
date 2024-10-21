
import sys

import pytest
from ase.build import bulk

from mlip_arena.models import MLIPEnum
from mlip_arena.tasks.md import run as MD

atoms = bulk("Cu", "fcc", a=3.6)

@pytest.mark.skipif(sys.version_info[:2] != (3,11), reason="avoid prefect race condition on concurrent tasks")
@pytest.mark.parametrize("model", [MLIPEnum["MACE-MP(M)"]])
def test_nve(model: MLIPEnum):

    result = MD.fn(
        atoms,
        calculator_name=model.name,
        calculator_kwargs={},
        ensemble="nve",
        dynamics="velocityverlet",
        total_time=3,
    )

    assert isinstance(result["atoms"].get_potential_energy(), float)
