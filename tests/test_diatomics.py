import tempfile
from pathlib import Path
from unittest.mock import patch
from ase.calculators.lj import LennardJones
from prefect.testing.utilities import prefect_test_harness

from mlip_arena.flows.diatomics import homonuclear_diatomic, homonuclear_diatomics


def test_diatomics_flow():
    calc = LennardJones(rc=5.0)

    with prefect_test_harness():
        with tempfile.TemporaryDirectory() as tmpdir:
            # Test homonuclear_diatomic task
            homonuclear_diatomic(
                symbol="Cu",
                calculator=calc,
                out_dir=Path(tmpdir),
            )

            # Test homonuclear_diatomics flow with chemical_symbols patched to avoid looping all 118 elements
            with patch("mlip_arena.flows.diatomics.chemical_symbols", ["", "Cu"]):
                res = homonuclear_diatomics(
                    calculator=calc,
                    run_dir=Path(tmpdir),
                )
                assert len(res) == 1
