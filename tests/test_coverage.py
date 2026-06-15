import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch
import pandas as pd
from ase.build import bulk
from ase.calculators.lj import LennardJones
from ase.db import connect
from prefect.testing.utilities import prefect_test_harness

from mlip_arena.flows.conservation import (
    run_nve_md,
    run_simulations,
    differential_entropy_along_nve_trajectory,
)
from mlip_arena.flows.eos_bulk import run as eos_bulk_run
from mlip_arena.flows.eos_bulk import run_db as eos_bulk_run_db
from mlip_arena.tasks.ev import run as ev_run
from mlip_arena.flows.ev import run_db as ev_run_db
from mlip_arena.flows.diatomics import homonuclear_diatomic, homonuclear_diatomics
from mlip_arena.tasks.eos_alloy.flow import save_to_hdf

# 1. Test Conservation


def test_conservation_coverage():
    atoms = bulk("Cu", "fcc", a=3.6)
    calc = LennardJones(rc=5.0)
    with prefect_test_harness():
        with tempfile.TemporaryDirectory() as tmpdir:
            traj_file = Path(tmpdir) / "nve.traj"
            run_nve_md(
                atoms=atoms,
                calculator=calc,
                traj_file=traj_file,
            )
            assert traj_file.exists()

            # Test run_simulations
            run_simulations(
                calculator=calc,
                structures=[atoms],
                out_dir=Path(tmpdir),
            )


def test_differential_entropy_flow_coverage():
    calc = LennardJones(rc=5.0)
    atoms = bulk("Cu", "fcc", a=3.6)
    with prefect_test_harness():
        with tempfile.TemporaryDirectory() as tmpdir:
            with (
                patch("mlip_arena.flows.conservation.read", return_value=[atoms]),
                patch("mlip_arena.flows.conservation.get_trajectory_entropy", return_value=(0.0, [atoms])),
            ):
                dH, sampled = differential_entropy_along_nve_trajectory(
                    calculator=calc,
                    input_path=Path("dummy.extxyz"),
                    reference_path=Path("dummy_ref.extxyz"),
                    start_idx=0,
                    end_idx=-1,
                    step=1,
                    work_dir=Path(tmpdir),
                )
                assert dH == 0.0


# 2. Test EOS Bulk


def test_eos_bulk_coverage():
    atoms = bulk("Cu", "fcc", a=3.6)
    atoms.info["key_value_pairs"] = {"wbm_id": "wbm-1"}
    calc = LennardJones(rc=5.0)

    with prefect_test_harness():
        # Test run task
        res = eos_bulk_run(atoms=atoms, calculator=calc)
        assert isinstance(res, pd.DataFrame)

        # Test run_db flow
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"
            with connect(db_path) as db:
                db.write(atoms)

            with patch("mlip_arena.flows.eos_bulk.hf_hub_download", return_value=db_path):
                df = eos_bulk_run_db(
                    calculator=calc,
                    run_dir=Path(tmpdir),
                    dataset="dummy",
                    dataset_file="test.db",
                )
                assert isinstance(df, pd.DataFrame)


# 3. Test EV Scan


def test_ev_coverage():
    atoms = bulk("Cu", "fcc", a=3.6)
    atoms.info["key_value_pairs"] = {"wbm_id": "wbm-1"}
    calc = LennardJones(rc=5.0)

    with prefect_test_harness():
        # Test tasks.ev.run
        res = ev_run(atoms=atoms, calculator=calc, npoints=3)
        assert isinstance(res, pd.DataFrame)

        # Test flows.ev.run_db
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"
            with connect(db_path) as db:
                db.write(atoms)

            with patch("mlip_arena.flows.ev.hf_hub_download", return_value=db_path):
                df = ev_run_db(
                    calculator=calc,
                    run_dir=Path(tmpdir),
                    dataset="dummy",
                    dataset_file="test.db",
                )
                assert isinstance(df, pd.DataFrame)


# 4. Test Diatomics


def test_diatomics_coverage():
    calc = LennardJones(rc=5.0)
    with prefect_test_harness():
        with tempfile.TemporaryDirectory() as tmpdir:
            # Test homonuclear_diatomic
            homonuclear_diatomic(
                symbol="Cu",
                calculator=calc,
                out_dir=Path(tmpdir),
            )

            # Test homonuclear_diatomics flow with chemical_symbols patched
            with patch("mlip_arena.flows.diatomics.chemical_symbols", ["", "Cu"]):
                res = homonuclear_diatomics(
                    calculator=calc,
                    run_dir=Path(tmpdir),
                )
                assert len(res) == 1


# 5. Test EOS Alloy Hook


def test_save_to_hdf_coverage():
    mock_task = MagicMock()

    # Case A: run has parameters
    mock_run_params = MagicMock()
    mock_run_params.parameters = {"calculator": "LennardJones"}
    if hasattr(mock_run_params, "task_inputs"):
        del mock_run_params.task_inputs

    # Case B: run has task_inputs
    mock_run_inputs = MagicMock()
    if hasattr(mock_run_inputs, "parameters"):
        del mock_run_inputs.parameters
    mock_run_inputs.task_inputs = {"calculator": ["LennardJones"]}

    mock_state = MagicMock()
    atoms = bulk("Cu", "fcc", a=3.6)
    mock_state.result.return_value = {
        "atoms": atoms,
        "calculator_name": "LennardJones",
        "eos": {"energies": [0.0], "volumes": [1.0]},
        "b0": 1.0,
        "b0_GPa": 1.0,
        "v0": 1.0,
        "e0": 1.0,
    }

    with tempfile.TemporaryDirectory() as tmpdir:
        out_path = Path(tmpdir) / "test.h5"

        # Call hook with parameters path
        save_to_hdf(mock_task, mock_run_params, mock_state, out_path, "table_name")
        assert out_path.exists()

        # Call hook with task_inputs path
        out_path.unlink()
        save_to_hdf(mock_task, mock_run_inputs, mock_state, out_path, "table_name")
        assert out_path.exists()
