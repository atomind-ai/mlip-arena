import sys
import tempfile
from pathlib import Path
from unittest.mock import patch
import pytest

from prefect.testing.utilities import prefect_test_harness

from ase.build import bulk
from ase.calculators.lj import LennardJones

from mlip_arena.flows.stability import (
    compression,
    heating,
    npt_compress_one,
    nvt_heat_one,
)
from mlip_arena.tasks.utils import resolve_calculator_name
from mlip_arena.models import MLIPEnum


def mock_md_run(*args, **kwargs):
    atoms = kwargs.get("atoms")
    traj_file = kwargs.get("traj_file")
    if traj_file:
        from ase.io.trajectory import Trajectory

        with Trajectory(traj_file, "w", atoms) as traj:
            traj.write(atoms)
    return {"atoms": atoms, "runtime": 0.0, "n_steps": 0}


def test_resolve_calculator_name():
    # Test string input
    assert resolve_calculator_name("custom-model") == "custom-model"

    # Test MLIPEnum member input
    assert resolve_calculator_name(MLIPEnum["MACE-MP(M)"]) == "MACE-MP(M)"

    # Test class input
    assert resolve_calculator_name(LennardJones) == "LennardJones"

    # Test instance input
    calc = LennardJones(rc=1.0)
    assert resolve_calculator_name(calc) == "LennardJones"


def test_nvt_heat_one():
    atoms = bulk("Cu", "fcc", a=3.6, cubic=True)
    calc = LennardJones(rc=5.0)

    with tempfile.TemporaryDirectory() as tmpdir:
        run_dir = Path(tmpdir)
        with patch("mlip_arena.flows.stability.MD") as mock_md:
            mock_md.with_options.return_value = mock_md_run
            # Use .fn to bypass Prefect tasks runner locally
            result = nvt_heat_one.fn(atoms=atoms, calculator=calc, run_dir=run_dir)

        # Verify the trajectory file was created
        expected_traj = run_dir / "LennardJones_random_Cu4_nvt.traj"
        assert expected_traj.exists()
        assert "atoms" in result


def test_npt_compress_one():
    atoms = bulk("Cu", "fcc", a=3.6, cubic=True)
    calc = LennardJones(rc=5.0)

    with tempfile.TemporaryDirectory() as tmpdir:
        run_dir = Path(tmpdir)
        with patch("mlip_arena.flows.stability.MD") as mock_md:
            mock_md.with_options.return_value = mock_md_run
            # Use .fn to bypass Prefect tasks runner locally
            result = npt_compress_one.fn(atoms=atoms, calculator=calc, run_dir=run_dir)

        # Verify the trajectory file was created
        expected_traj = run_dir / "LennardJones_random_Cu4_npt.traj"
        assert expected_traj.exists()
        assert "atoms" in result


@pytest.mark.skipif(
    sys.version_info[:2] != (3, 12),
    reason="avoid prefect race condition on concurrent tasks",
)
def test_heating_flow():
    calc = LennardJones(rc=5.0)
    mock_atoms = [bulk("Cu", "fcc", a=3.6, cubic=True)]

    with prefect_test_harness():
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = Path(tmpdir)
            with (
                patch("mlip_arena.flows.stability.get_atoms_from_db", return_value=mock_atoms),
                patch("mlip_arena.flows.stability.MD") as mock_md,
            ):
                mock_md.with_options.return_value = mock_md_run
                # Run the heating flow
                results = heating(calculator=calc, run_dir=run_dir)
                assert len(results) == 1

                # Check file was generated in the custom directory
                expected_traj = run_dir / "LennardJones_random_Cu4_nvt.traj"
                assert expected_traj.exists()


@pytest.mark.skipif(
    sys.version_info[:2] != (3, 12),
    reason="avoid prefect race condition on concurrent tasks",
)
def test_compression_flow():
    calc = LennardJones(rc=5.0)
    mock_atoms = [bulk("Cu", "fcc", a=3.6, cubic=True)]

    with prefect_test_harness():
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = Path(tmpdir)
            with (
                patch("mlip_arena.flows.stability.get_atoms_from_db", return_value=mock_atoms),
                patch("mlip_arena.flows.stability.MD") as mock_md,
            ):
                mock_md.with_options.return_value = mock_md_run
                # Run the compression flow
                results = compression(calculator=calc, run_dir=run_dir)
                assert len(results) == 1

                # Check file was generated in the custom directory
                expected_traj = run_dir / "LennardJones_random_Cu4_npt.traj"
                assert expected_traj.exists()
