import numpy as np
from ase.build import bulk

from mlip_arena.tasks.eos.run import fit as EOS

atoms = bulk("Cu", "fcc", a=3.6)

def test_eos():
    """
    Test EOS prefect workflow with a simple cubic lattice.
    """

    result = EOS(
        atoms=atoms,
        calculator_name="MACE-MP(M)",
        calculator_kwargs={},
        device=None,
        optimizer="BFGSLineSearch",
        optimizer_kwargs=None,
        filter="FrechetCell",
        filter_kwargs=None,
        criterion=dict(
            fmax=0.01,
        ),
        max_abs_strain=0.1,
        npoints=11,
    )

    print(result)

    assert isinstance(result["K"], float)
