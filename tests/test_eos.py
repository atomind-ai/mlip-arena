import sys

import pytest
from ase.build import bulk
from prefect import flow
from prefect.testing.utilities import prefect_test_harness

from mlip_arena.models import MLIPEnum
from mlip_arena.tasks.eos import run as EOS
from mlip_arena.tasks.utils import get_calculator



@flow(persist_result=True)
def single_eos_flow(calculator_name, concurrent=True, cache=False):
    """
    Run an equation-of-state (EOS) Prefect flow for an FCC copper bulk using the specified calculator.
    
    Parameters:
    	calculator_name (str): Name or identifier of the calculator to obtain via get_calculator.
    	concurrent (bool): If True, allow the EOS task to run with concurrency enabled.
    	cache (bool): If True, use cached EOS results when available; if False, force a refresh.
    
    Returns:
    	result (dict): Results produced by the EOS task. Contains keys produced by the EOS implementation (tests expect a `'b0'` key with a float value).
    """
    atoms = bulk("Cu", "fcc", a=3.6)

    return EOS.with_options(
        refresh_cache=not cache,
    )(
        atoms=atoms,
        calculator=get_calculator(
            calculator_name,
        ),
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
        assert b0_scratch == pytest.approx(b0_cache, rel=1e-5)