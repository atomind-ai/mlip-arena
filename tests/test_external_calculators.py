import pytest
from ase import Atoms

from mlip_arena.models.utils import MLIPEnum


@pytest.mark.parametrize("model", MLIPEnum)
def test_calculate(model):
    calc = MLIPEnum[model.name].value()

    atoms = Atoms(
        "OO",
        positions=[[0, 0, 0], [1.5, 0, 0]],
        cell=[10, 10 + 0.001, 10 + 0.002],
        pbc=True,
    )

    atoms.calc = calc

    print(atoms.get_potential_energy())
    assert isinstance(atoms.get_potential_energy(), float)
