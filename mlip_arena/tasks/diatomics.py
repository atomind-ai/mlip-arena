import covalent as ct
import numpy as np
import pandas as pd
import torch
from ase import Atoms
from ase.calculators.calculator import Calculator
from ase.data import chemical_symbols
from matplotlib import pyplot as plt

from mlip_arena.models import MLIPCalculator

device = torch.device("cuda")


@ct.electron
def calculate_single_diatomic(
    calculator: MLIPCalculator | Calculator,
    atom1: str,
    atom2: str,
    rmin: float = 0.1,
    rmax: float = 6.5,
    npts: int = int(1e3),
):
    a = 2 * rmax

    rs = np.linspace(rmin, rmax, npts)
    e = np.zeros_like(rs)
    f = np.zeros_like(rs)

    da = atom1 + atom2

    for i, r in enumerate(rs):

        positions = [
            [0, 0, 0],
            [r, 0, 0],
        ]

        # Create the unit cell with two atoms
        atoms = Atoms(da, positions=positions, cell=[a, a, a])

        atoms.calc = calculator

        e[i] = atoms.get_potential_energy()
        f[i] = np.inner(np.array([1, 0, 0]), atoms.get_forces()[1])

    return rs, e, f, da


@ct.lattice
def calculate_homonuclear_diatomics(calculator: MLIPCalculator | Calculator):

    chemical_symbols.remove("X")

    results = {}

    for atom in chemical_symbols:
        rs, e, f, da = calculate_single_diatomic(calculator, atom, atom)
        results[da] = {"r": rs, "E": e, "F": f}

    return results


# with plt.style.context("default"):

#     SMALL_SIZE = 6
#     MEDIUM_SIZE = 8
#     LARGE_SIZE = 10

#     LINE_WIDTH = 1

#     plt.rcParams.update(
#         {
#             "pgf.texsystem": "pdflatex",
#             "font.family": "sans-serif",
#             "text.usetex": True,
#             "pgf.rcfonts": True,
#             "figure.constrained_layout.use": True,
#             "axes.labelsize": MEDIUM_SIZE,
#             "axes.titlesize": MEDIUM_SIZE,
#             "legend.frameon": False,
#             "legend.fontsize": MEDIUM_SIZE,
#             "legend.loc": "best",
#             "lines.linewidth": LINE_WIDTH,
#             "xtick.labelsize": SMALL_SIZE,
#             "ytick.labelsize": SMALL_SIZE,
#         }
#     )

#     fig, ax = plt.subplots(layout="constrained", figsize=(3, 2), dpi=300)

#     color = "tab:red"
#     ax.plot(rs, e, color=color, zorder=1)

#     ax.axhline(ls="--", color=color, alpha=0.5, lw=0.5 * LINE_WIDTH)

#     ylo, yhi = ax.get_ylim()
#     ax.set(xlabel=r"r [$\AA]$", ylim=(max(-7, ylo), min(5, yhi)))
#     ax.set_ylabel(ylabel="E [eV]", color=color)
#     ax.tick_params(axis="y", labelcolor=color)
#     ax.text(0.8, 0.85, da, fontsize=LARGE_SIZE, transform=ax.transAxes)

#     color = "tab:blue"

#     at = ax.twinx()
#     at.plot(rs, f, color=color, zorder=0, lw=0.5 * LINE_WIDTH)

#     at.axhline(ls="--", color=color, alpha=0.5, lw=0.5 * LINE_WIDTH)

#     ylo, yhi = at.get_ylim()
#     at.set(
#         xlabel=r"r [$\AA]$",
#         ylim=(max(-20, ylo), min(20, yhi)),
#     )
#     at.set_ylabel(ylabel="F [eV/$\AA$]", color=color)
#     at.tick_params(axis="y", labelcolor=color)

#     plt.show()


if __name__ == "__main__":

    local = ct.executor.LocalExecutor()
