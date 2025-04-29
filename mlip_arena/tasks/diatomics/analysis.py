from pathlib import Path

import numpy as np
import pandas as pd
from ase.data import chemical_symbols
from ase.io import read
from scipy import stats
from scipy.interpolate import UnivariateSpline
from tqdm.auto import tqdm

from mlip_arena.models import REGISTRY, MLIPEnum

for model in MLIPEnum:

    df = pd.DataFrame(
        columns=[
            "name",
            "method",
            "R",
            "E",
            "F",
            "S^2",
            "force-flip-times",
            "force-total-variation",
            "force-jump",
            "energy-diff-flip-times",
            "energy-grad-norm-max",
            "energy-jump",
            "energy-total-variation",
            "tortuosity",
            "conservation-deviation",
            "spearman-descending-force",
            "spearman-ascending-force",
            "spearman-repulsion-energy",
            "spearman-attraction-energy",
            "pbe-energy-mae",
            "pbe-force-mae",
        ]
    )

    for symbol in tqdm(chemical_symbols[1:]):
        da = symbol + symbol

        out_dir = Path(model.name)

        traj_fpath = out_dir / f"{str(da)}.extxyz"

        if traj_fpath.exists():
            traj = read(traj_fpath, index=":")
        else:
            continue

        Rs, Es, Fs, S2s = [], [], [], []
        for atoms in traj:
            vec = atoms.positions[1] - atoms.positions[0]
            r = np.linalg.norm(vec)
            e = atoms.get_potential_energy()
            f = np.inner(vec / r, atoms.get_forces()[1])
            # s2 = np.mean(np.power(atoms.get_magnetic_moments(), 2))

            Rs.append(r)
            Es.append(e)
            Fs.append(f)
            # S2s.append(s2)

        rs = np.array(Rs)
        es = np.array(Es)
        fs = np.array(Fs)

        # sort interatomic distances and align to zero at far field
        indices = np.argsort(rs)[::-1]
        rs = rs[indices]
        es = es[indices]
        eshift = es[0]
        es -= eshift
        fs = fs[indices]

        iminf = np.argmin(fs)
        imine = np.argmin(es)

        de_dr = np.gradient(es, rs)
        d2e_dr2 = np.gradient(de_dr, rs)

        # avoid numerical sensitity close to zero
        rounded_fs = np.copy(fs)
        rounded_fs[np.abs(rounded_fs) < 1e-2] = 0  # 10meV/A
        fs_sign = np.sign(rounded_fs)
        mask = fs_sign != 0
        rounded_fs = rounded_fs[mask]
        fs_sign = fs_sign[mask]
        f_flip = np.diff(fs_sign) != 0

        fdiff = np.diff(fs)
        fdiff_sign = np.sign(fdiff)
        mask = fdiff_sign != 0
        fdiff = fdiff[mask]
        fdiff_sign = fdiff_sign[mask]
        fdiff_flip = np.diff(fdiff_sign) != 0
        fjump = (
            np.abs(fdiff[:-1][fdiff_flip]).sum() + np.abs(fdiff[1:][fdiff_flip]).sum()
        )

        ediff = np.diff(es)
        ediff[np.abs(ediff) < 1e-3] = 0  # 1meV
        ediff_sign = np.sign(ediff)
        mask = ediff_sign != 0
        ediff = ediff[mask]
        ediff_sign = ediff_sign[mask]
        ediff_flip = np.diff(ediff_sign) != 0
        ejump = (
            np.abs(ediff[:-1][ediff_flip]).sum() + np.abs(ediff[1:][ediff_flip]).sum()
        )

        try:
            pbe_traj = read(f"./vasp/{da}/PBE.extxyz", index=":")

            pbe_rs, pbe_es, pbe_fs = [], [], []

            for atoms in pbe_traj:
                vec = atoms.positions[1] - atoms.positions[0]
                r = np.linalg.norm(vec)
                pbe_rs.append(r)
                pbe_es.append(atoms.get_potential_energy())
                pbe_fs.append(np.inner(vec / r, atoms.get_forces()[1]))

            pbe_rs = np.array(pbe_rs)
            pbe_es = np.array(pbe_es)
            pbe_fs = np.array(pbe_fs)

            indices = np.argsort(pbe_rs)
            pbe_rs = pbe_rs[indices]
            pbe_es = pbe_es[indices]
            pbe_fs = pbe_fs[indices]

            pbe_es -= pbe_es[-1]

            xs = np.linspace(pbe_rs.min(), pbe_rs.max(), int(1e3))

            cs = UnivariateSpline(pbe_rs, pbe_es, s=0)
            pbe_energy_mae = np.mean(np.abs(es - cs(rs)))

            cs = UnivariateSpline(pbe_rs, pbe_fs, s=0)
            pbe_force_mae = np.mean(np.abs(fs - cs(rs)))
        except Exception as e:
            print(e)
            pbe_energy_mae = None
            pbe_force_mae = None

        conservation_deviation = np.mean(np.abs(fs + de_dr))

        etv = np.sum(np.abs(np.diff(es)))

        data = {
            "name": da,
            "method": model.name,
            "R": rs,
            "E": es + eshift,
            "F": fs,
            "S^2": S2s,
            "force-flip-times": np.sum(f_flip),
            "force-total-variation": np.sum(np.abs(np.diff(fs))),
            "force-jump": fjump,
            "energy-diff-flip-times": np.sum(ediff_flip),
            "energy-grad-norm-max": np.max(np.abs(de_dr)),
            "energy-jump": ejump,
            # "energy-grad-norm-mean": np.mean(de_dr_abs),
            "energy-total-variation": etv,
            "tortuosity": etv / (abs(es[0] - es.min()) + (es[-1] - es.min())),
            "conservation-deviation": conservation_deviation,
            "spearman-descending-force": stats.spearmanr(
                rs[iminf:], fs[iminf:]
            ).statistic,
            "spearman-ascending-force": stats.spearmanr(
                rs[:iminf], fs[:iminf]
            ).statistic,
            "spearman-repulsion-energy": stats.spearmanr(
                rs[imine:], es[imine:]
            ).statistic,
            "spearman-attraction-energy": stats.spearmanr(
                rs[:imine], es[:imine]
            ).statistic,
            "pbe-energy-mae": pbe_energy_mae,
            "pbe-force-mae": pbe_force_mae,
        }

        df = pd.concat([df, pd.DataFrame([data])], ignore_index=True)

    json_fpath = Path(REGISTRY[model.name]["family"]) / "homonuclear-diatomics.json"

    if json_fpath.exists():
        df0 = pd.read_json(json_fpath)
        df = pd.concat([df0, df], ignore_index=True)
        df.drop_duplicates(inplace=True, subset=["name", "method"], keep="last")

    df.to_json(json_fpath, orient="records")

    json_fpath = Path(model.name) / "homonuclear-diatomics.json"
    df.to_json(json_fpath, orient="records")
