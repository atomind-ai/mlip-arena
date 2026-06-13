# tests/test_external_calculators.py

import numpy as np
import pytest
from ase import Atoms
from ase.calculators.calculator import PropertyNotImplementedError
from httpx import HTTPStatusError
from huggingface_hub.errors import GatedRepoError
from requests import HTTPError

from mlip_arena.models import MLIPEnum


model_params = []
for model in MLIPEnum:
    marks = []
    if model.name == "SevenNet":
        marks.append(pytest.mark.sevennet)
    elif model.name == "NequIP-OAM-L":
        marks.append(pytest.mark.nequip)
    elif "MACE" in model.name:
        marks.append(pytest.mark.mace)
    elif model.value.get("family") == "fairchem":
        marks.append(pytest.mark.fairchem)

    if marks:
        model_params.append(pytest.param(model, marks=marks, id=model.name))
    else:
        model_params.append(pytest.param(model, id=model.name))


@pytest.mark.parametrize("model", model_params)
def test_calculate(model: MLIPEnum):
    try:
        calc = MLIPEnum[model.name].load()
    except (
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
            pytest.xfail("Orbital Materials deprecated the model a month after its premature release in favor of ORBv2")
        elif model.name == "M3GNet":
            pytest.xfail("Cache sometimes fails")
        elif model.name == "UMA-S-1P1":
            pytest.xfail("The model fails CI on CPU for RuntimeError: expected scalar type Double but found Float")
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

    assert isinstance(energy, float | np.float64 | np.float32)

    forces = atoms.get_forces()
    assert isinstance(forces, np.ndarray | list)
    assert len(forces) == len(atoms)

    try:
        stress = atoms.get_stress()
    except PropertyNotImplementedError:
        stress = None

    if stress is None:
        pytest.xfail("Stress calculation is not supported by the model")
    else:
        assert isinstance(stress, np.ndarray | list)
