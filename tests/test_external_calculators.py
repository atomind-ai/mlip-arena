import numpy as np
import pytest
from ase import Atoms
from ase.calculators.calculator import PropertyNotImplementedError
from httpx import HTTPStatusError
from huggingface_hub.errors import GatedRepoError, LocalTokenNotFoundError
from requests import HTTPError

from mlip_arena.models import MLIPEnum


@pytest.mark.parametrize("model", MLIPEnum)
def test_calculate(model: MLIPEnum):
    try:
        calc = MLIPEnum[model.name].value()
    except (
        LocalTokenNotFoundError,
        GatedRepoError,
        HTTPError,
        HTTPStatusError,
        FileNotFoundError,
    ) as e:
        pytest.skip(str(e))
    except Exception as e:
        if model.name == "ALIGNN":
            pytest.xfail("ALIGNN has poor file download mechanism")
        elif model.name == "ORB":
            pytest.xfail(
                "Orbital Materials deprecated the model a month after its premature release in favor of ORBv2"
            )
        elif model.name == "M3GNet":
            pytest.xfail("Cache sometimes fails")
        elif model.name == "NequIP-OAM-L":
            pytest.xfail("CPU model is not available at the moment")
        else:
            pytest.fail(f"Failed to initialize model {model.name}: {e}")

    atoms = Atoms(
        "OO",
        positions=[[0, 0, 0], [1.5, 0, 0]],
        cell=[10, 10 + 0.001, 10 + 0.002],
        pbc=True,
    )

    atoms.calc = calc

    energy = atoms.get_potential_energy()

    assert isinstance(energy, (float, np.float64, np.float32))

    forces = atoms.get_forces()
    assert isinstance(forces, (np.ndarray, list))
    assert len(forces) == len(atoms)

    try:
        stress = atoms.get_stress()
    except PropertyNotImplementedError:
        stress = None

    if stress is None:
        pytest.xfail("Stress calculation is not supported by the model")
    else:
        assert isinstance(stress, (np.ndarray, list))
