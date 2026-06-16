import sys

import pytest
from ase.build import bulk

from mlip_arena.models import MLIPEnum
from mlip_arena.tasks.md import run as MD

atoms = bulk("Cu", "fcc", a=3.6)


@pytest.mark.skipif(sys.version_info[:2] != (3, 12), reason="avoid prefect race condition on concurrent tasks")
@pytest.mark.mace
@pytest.mark.parametrize("model", [MLIPEnum["MACE-MP(M)"]])
def test_nve(model: MLIPEnum):
    result = MD.fn(
        atoms,
        calculator=model.name,
        ensemble="nve",
        dynamics="velocityverlet",
        total_time=10,
        time_step=2,
        dynamics_kwargs={},
    )

    res_atoms = result["atoms"]
    from mlip_arena.tasks.utils import get_calculator

    res_atoms.calc = get_calculator(model.name)
    assert isinstance(res_atoms.get_potential_energy(), float)
