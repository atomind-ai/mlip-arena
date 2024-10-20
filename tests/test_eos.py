import pytest
from ase.build import bulk

from mlip_arena.models import MLIPEnum
from mlip_arena.tasks.eos.run import fit as EOS

atoms = bulk("Cu", "fcc", a=3.6)


@pytest.mark.parametrize("model", [MLIPEnum["MACE-MP(M)"]])
def test_eos(model: MLIPEnum):
    """
    Test EOS prefect workflow with a simple cubic lattice.
    """

    result = EOS(
        atoms=atoms,
        calculator_name=model.name,
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

    print(result)

    assert isinstance(result["K"], float)
