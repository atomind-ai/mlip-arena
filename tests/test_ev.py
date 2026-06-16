import tempfile
from pathlib import Path
from unittest.mock import patch
import pandas as pd
from ase.build import bulk
from ase.calculators.lj import LennardJones
from ase.db import connect
from prefect.testing.utilities import prefect_test_harness

from mlip_arena.tasks.ev import run as ev_run
from mlip_arena.flows.ev import run_db as ev_run_db


def test_ev_scan_flow():
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
