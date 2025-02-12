import numpy as np
from mlip_arena.models import MLIPCalculator
from mlip_arena.models.classicals.zbl import ZBL

from ase.build import bulk


def test_zbl():
    calc = MLIPCalculator(model=ZBL(), cutoff=6.0)

    energies = []
    forces = []
    stresses = []

    lattice_constants = [1, 3, 5, 7]

    for a in lattice_constants:
        atoms = bulk("Cu", "fcc", a=a) * (2, 2, 2)
        atoms.calc = calc

        energies.append(atoms.get_potential_energy())
        forces.append(atoms.get_forces())
        stresses.append(atoms.get_stress(voigt=False))

    # test energy monotonicity
    assert all(np.diff(energies) <= 0), "Energy is not monotonically decreasing with increasing lattice constant"

    # test force vectors are all zeros due to symmetry
    for f in forces:
        assert np.allclose(f, 0), "Forces should be zero due to symmetry"

    # test trace of stress is monotonically increasing (less negative) and zero beyond cutoff
    traces = [np.trace(s) for s in stresses]

    assert all(np.diff(traces) >= 0), "Trace of stress is not monotonically increasing with increasing lattice constant"
    assert np.allclose(stresses[-1], 0), "Stress should be zero beyond cutoff"
