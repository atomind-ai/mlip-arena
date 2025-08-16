import numpy as np
from ase.build import bulk
from ase.calculators.lj import LennardJones

from mlip_arena.tasks.viscosity import run as VISCOSITY


def test_task_run():
    """Test the viscosity task."""
    atoms = bulk("Cu", "fcc", a=3.6, cubic=True) * (2, 2, 2)

    atoms.positions = atoms.positions + 0.5 * np.random.rand(*atoms.positions.shape)

    calculator = LennardJones(
        epsilon=0.01, sigma=(atoms.get_volume() / len(atoms)) ** (1 / 3)
    )

    result = VISCOSITY(
        atoms,
        calculator=calculator,
        temperature=1250,
        pressure=0,
        npt_eq_time=100,
        nve_eq_time=100,
        prod_time=100,
    )
    assert isinstance(result, dict)
    assert "viscosity" in result
    assert isinstance(result["viscosity"]["average"], float)