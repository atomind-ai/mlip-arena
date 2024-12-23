import sys
import numpy as np

import pytest
from mlip_arena.models import MLIPEnum
from mlip_arena.tasks.thermal_conductivity.task import get_thermal_conductivity
from mlip_arena.tasks.thermal_conductivity.input import get_atoms_from_extxyz
from prefect.testing.utilities import prefect_test_harness


@pytest.mark.skipif(
    sys.version_info[:2] != (3, 11),
    reason="avoid prefect race condition on concurrent tasks",
)
@pytest.mark.parametrize("model", [MLIPEnum["MACE-MP(M)"]])
def test_tc_task(model: MLIPEnum):
    """Test thermal conductivity task on single structure"""

    pytest.importorskip("phono3py")

    # Phono3py will raise FloatingPointError in pytest environment if not ignored
    np.seterr(all="ignore")

    atoms = next(get_atoms_from_extxyz("phononDB-PBE-structures.extxyz"))

    with prefect_test_harness():
        result = get_thermal_conductivity.with_options(
            refresh_cache=True,
        )(
            atoms=atoms,
            calculator_name=model.name,
        )
        assert isinstance(result, dict)
        mode_kappa_tot_scratch = result["kappa"]["mode_kappa_TOT"]
        assert isinstance(mode_kappa_tot_scratch[0, 0, 0, 0], float)
