import tempfile
from pathlib import Path
from unittest.mock import patch
import pandas as pd
from ase.build import bulk
from ase.calculators.lj import LennardJones
from ase.db import connect
from prefect.testing.utilities import prefect_test_harness

from mlip_arena.flows.eos_bulk import run, run_db


def test_eos_bulk_flow():
    atoms = bulk("Cu", "fcc", a=3.6)
    atoms.info["key_value_pairs"] = {"wbm_id": "wbm-1"}
    calc = LennardJones(rc=5.0)

    with prefect_test_harness():
        # Test run task
        res = run(atoms=atoms, calculator=calc)
        assert isinstance(res, pd.DataFrame)

        # Test run_db flow
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"
            with connect(db_path) as db:
                db.write(atoms)

            with patch("mlip_arena.flows.eos_bulk.hf_hub_download", return_value=db_path):
                df = run_db(
                    calculator=calc,
                    run_dir=Path(tmpdir),
                    dataset="dummy",
                    dataset_file="test.db",
                )
                assert isinstance(df, pd.DataFrame)
