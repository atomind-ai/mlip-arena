import pytest
from ase import Atoms

from mlip_arena.models import MLIPEnum

from huggingface_hub.errors import LocalTokenNotFoundError

@pytest.mark.parametrize("model", MLIPEnum)
def test_calculate(model: MLIPEnum):

    if model.name == "ALIGNN":
        pytest.xfail("ALIGNN has poor file download mechanism")

    try:
        calc = MLIPEnum[model.name].value()

    except LocalTokenNotFoundError:
        # Gracefully skip the test if HF_TOKEN is not available
        pytest.skip("Skipping test because HF_TOKEN is not available for downloading the model.")

    atoms = Atoms(
        "OO",
        positions=[[0, 0, 0], [1.5, 0, 0]],
        cell=[10, 10 + 0.001, 10 + 0.002],
        pbc=True,
    )

    atoms.calc = calc

    print(atoms.get_potential_energy())
    assert isinstance(atoms.get_potential_energy(), float)
