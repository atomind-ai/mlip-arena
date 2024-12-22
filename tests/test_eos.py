import sys

import pytest
from ase.build import bulk
from prefect import flow
from prefect.testing.utilities import prefect_test_harness

from mlip_arena.models import MLIPEnum
from mlip_arena.tasks.eos import run as EOS


@flow
def single_eos_flow(calculator_name, concurrent=True, cache=False):
    atoms = bulk("Cu", "fcc", a=3.6)

    EOS_ = EOS.with_options(
        refresh_cache=not cache,
    )
    
    return EOS_(
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
        concurrent=concurrent,
        cache_opt=cache
    )


@pytest.mark.skipif(
    sys.version_info[:2] != (3, 11),
    reason="avoid prefect race condition on concurrent tasks",
)
@pytest.mark.parametrize("concurrent", [False])
@pytest.mark.parametrize("model", [MLIPEnum["MACE-MP(M)"]])
def test_eos(model: MLIPEnum, concurrent: bool):
    """
    Test EOS prefect workflow with a simple cubic lattice.
    """

    with prefect_test_harness():
        result = single_eos_flow(
            calculator_name=model.name,
            concurrent=concurrent,
            cache=False,
        )
        assert isinstance(b0_scratch := result["b0"], float)

        # @pytest.mark.dependency(depends=["test_eos"])
        # @pytest.mark.parametrize("model", [MLIPEnum["MACE-MP(M)"]])
        # def test_eos_cache(model: MLIPEnum):

        result = single_eos_flow(
            calculator_name=model.name,
            concurrent=concurrent,
            cache=True,
        )
        assert isinstance(b0_cache := result["b0"], float)
        assert b0_scratch == pytest.approx(b0_cache, rel=1e-6)
