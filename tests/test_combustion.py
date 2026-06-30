import sys
import tempfile
from pathlib import Path
from datetime import datetime
from unittest.mock import patch
import pytest

from ase import Atoms
from prefect.testing.utilities import prefect_test_harness

from mlip_arena.flows.combustion import (
    identify_water_molecules,
    get_runtime_stats,
    hydrogen_combustion,
)
from ase.calculators.lj import LennardJones
from ase.calculators.singlepoint import SinglePointCalculator


def test_identify_water_molecules():
    # 1. Test single water molecule
    water = Atoms(symbols="OHH", positions=[[0.0, 0.0, 0.0], [0.0, 0.757, 0.586], [0.0, -0.757, 0.586]])
    assert identify_water_molecules(water) == 1

    # 2. Test isolated atoms (no water molecules)
    isolated = Atoms(symbols="OHH", positions=[[0.0, 0.0, 0.0], [10.0, 0.0, 0.0], [0.0, 10.0, 0.0]])
    assert identify_water_molecules(isolated) == 0


def test_get_runtime_stats():
    # Create a mock trajectory with 3 frames + 1 invalid frame
    atoms0 = Atoms(symbols="H", positions=[[0.0, 0.0, 0.0]])
    atoms0.calc = LennardJones()

    frame1 = atoms0.copy()
    frame1.calc = LennardJones()
    # Frame 1 has no info keys (tests robust parsing - defaults to step 0 and first_valid_info datetime)

    frame2 = atoms0.copy()
    frame2.calc = LennardJones()
    frame2.info["restart"] = 0
    frame2.info["datetime"] = datetime(2026, 6, 20, 12, 0, 0)
    frame2.info["step"] = 1000
    frame2.info["target_steps"] = 2000000

    frame3 = atoms0.copy()
    frame3.calc = LennardJones()
    frame3.info["restart"] = 0
    frame3.info["datetime"] = datetime(2026, 6, 20, 12, 0, 10)
    frame3.info["step"] = 2000
    frame3.info["target_steps"] = 2000000

    frame_invalid = atoms0.copy()  # No calculator, will raise exception and be skipped

    traj = [frame1, frame2, frame3, frame_invalid]
    stats = get_runtime_stats.fn(traj, atoms0)

    assert stats["natoms"] == 1
    assert stats["total_time_seconds"] == 10.0
    assert stats["total_steps"] == 2000
    assert len(stats["energies"]) == 3  # frame1 included (defaults to step 0), frame_invalid skipped
    assert stats["target_steps"] == 2000000


@pytest.mark.skipif(
    sys.version_info[:2] != (3, 12),
    reason="avoid prefect race condition on concurrent tasks",
)
def test_hydrogen_combustion_flow():
    calc = LennardJones(rc=5.0)

    # Retrieve path to the local structure file
    real_atoms_path = Path(__file__).parents[1] / "benchmarks" / "combustion" / "H256O128.extxyz"

    class MockFuture:
        def __init__(self, val):
            self.val = val

        def result(self, **kwargs):
            return self.val

    def mock_md_submit(*args, **kwargs):
        traj_file = kwargs.get("traj_file")
        atoms = kwargs.get("atoms")
        if traj_file:
            from ase.io.trajectory import Trajectory

            # Write 2001 frames so postprocessing doesn't fail assertion (len(traj) >= 2000)
            with Trajectory(traj_file, "w", atoms) as traj:
                for i in range(2001):
                    frame = atoms.copy()
                    frame.calc = SinglePointCalculator(frame, energy=0.0)
                    if i > 0:
                        frame.info["restart"] = 0
                        frame.info["datetime"] = datetime.now()
                        frame.info["step"] = i * 1000
                        frame.info["target_steps"] = 2000000
                    traj.write(frame)
        return MockFuture({"atoms": atoms, "runtime": 0.0, "n_steps": 2000})

    with prefect_test_harness():
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = Path(tmpdir)
            with (
                patch("mlip_arena.flows.combustion.hf_hub_download", return_value=str(real_atoms_path)),
                patch("mlip_arena.flows.combustion.MD") as mock_md,
            ):
                mock_md.submit = mock_md_submit

                # Execute the flow
                result = hydrogen_combustion(run_dir=run_dir, calculator=calc)

                # Verify that files were created
                model_name = "LennardJones"
                expected_traj = run_dir / f"{model_name}_H256O128.traj"
                expected_json = run_dir / f"{model_name}_H256O128.json"

                assert expected_traj.exists()
                assert expected_json.exists()
                assert result["n_steps"] == 2000
