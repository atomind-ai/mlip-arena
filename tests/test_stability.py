import tempfile
from pathlib import Path
from unittest.mock import patch

from ase.build import bulk
from ase.calculators.lj import LennardJones

from mlip_arena.flows.stability import (
    compression,
    heating,
    npt_compress_one,
    nvt_heat_one,
    resolve_model_name,
)
from mlip_arena.models import MLIPEnum


def test_resolve_model_name():
    # Test string input
    assert resolve_model_name("custom-model") == "custom-model"

    # Test MLIPEnum member input
    assert resolve_model_name(MLIPEnum["MACE-MP(M)"]) == "MACE-MP(M)"

    # Test class input
    assert resolve_model_name(LennardJones) == "LennardJones"

    # Test instance input
    calc = LennardJones(rc=1.0)
    assert resolve_model_name(calc) == "LennardJones"


def test_nvt_heat_one():
    atoms = bulk("Cu", "fcc", a=3.6)
    calc = LennardJones(rc=5.0)

    with tempfile.TemporaryDirectory() as tmpdir:
        run_dir = Path(tmpdir)
        # Use .fn to bypass Prefect tasks runner locally
        result = nvt_heat_one.fn(atoms, calc, run_dir)

        # Verify the trajectory file was created
        expected_traj = run_dir / "LennardJones_random_Cu4_nvt.traj"
        assert expected_traj.exists()
        assert "atoms" in result


def test_npt_compress_one():
    atoms = bulk("Cu", "fcc", a=3.6)
    calc = LennardJones(rc=5.0)

    with tempfile.TemporaryDirectory() as tmpdir:
        run_dir = Path(tmpdir)
        # Use .fn to bypass Prefect tasks runner locally
        result = npt_compress_one.fn(atoms, calc, run_dir)

        # Verify the trajectory file was created
        expected_traj = run_dir / "LennardJones_random_Cu4_npt.traj"
        assert expected_traj.exists()
        assert "atoms" in result


def test_heating_flow():
    calc = LennardJones(rc=5.0)
    mock_atoms = [bulk("Cu", "fcc", a=3.6)]

    with tempfile.TemporaryDirectory() as tmpdir:
        run_dir = Path(tmpdir)
        with patch("mlip_arena.flows.stability.get_atoms_from_db", return_value=mock_atoms):
            # Run the heating flow
            results = heating(model=calc, run_dir=run_dir)
            assert len(results) == 1

            # Check file was generated in the custom directory
            expected_traj = run_dir / "custom" / "LennardJones_random_Cu4_nvt.traj"
            assert expected_traj.exists()


def test_compression_flow():
    calc = LennardJones(rc=5.0)
    mock_atoms = [bulk("Cu", "fcc", a=3.6)]

    with tempfile.TemporaryDirectory() as tmpdir:
        run_dir = Path(tmpdir)
        with patch("mlip_arena.flows.stability.get_atoms_from_db", return_value=mock_atoms):
            # Run the compression flow
            results = compression(model=calc, run_dir=run_dir)
            assert len(results) == 1

            # Check file was generated in the custom directory
            expected_traj = run_dir / "custom" / "LennardJones_random_Cu4_npt.traj"
            assert expected_traj.exists()
