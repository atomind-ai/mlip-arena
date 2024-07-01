from pathlib import Path

import numpy as np
import numpy.linalg as LA
import plotly.express as px
import streamlit as st
from ase.data import chemical_symbols
from ase.io import read
from scipy.interpolate import CubicSpline

st.markdown("# Homonuclear diatomics")

DATA_DIR = Path("mlip_arena/tasks/diatomics")


for i, symbol in enumerate(chemical_symbols[1:10]):

    if i % 3 == 0:
        cols = st.columns(3)

    fpath = DATA_DIR / "gpaw" / f"{symbol+symbol}_AFM" / "traj.extxyz"

    if not fpath.exists():
        continue

    trj = read(fpath, index=":")

    rs, es, s2s = [], [], []

    for atoms in trj:
        rs.append(LA.norm(atoms.positions[1] - atoms.positions[0]))
        es.append(atoms.get_potential_energy())
        s2s.append(np.power(atoms.get_magnetic_moments(), 2).mean())

    rs = np.array(rs)
    ind = np.argsort(rs)
    es = np.array(es)
    s2s = np.array(s2s)

    rs = rs[ind]
    es = es[ind]
    s2s = s2s[ind]

    es = es - es[-1]

    xs = np.linspace(rs.min()*0.99, rs.max()*1.01, int(5e2))

    cs = CubicSpline(rs, es)
    ys = cs(xs)

    cs = CubicSpline(rs, s2s)
    s2s = cs(xs)

    ylo = min(ys.min()*1.5, -1)

    fig = px.scatter(
        x=xs, y=ys,
        render_mode="webgl",
        color=s2s,
        range_color=[0, s2s.max()],
        width=500,
        range_y=[ylo, 1.2*(abs(ylo))],
        # title=f"{atoms.get_chemical_formula()}",
        labels={"x": "Bond length (Å)", "y": "Energy", "color": "Magnetic moment"},
    )

    cols[i % 3].title(f"{symbol+symbol}")
    cols[i % 3].plotly_chart(fig, use_container_width=False)

# st.latex(r"\frac{d^2E}{dr^2} = \frac{d^2E}{dr^2}")

# st.components.v1.html(fig.to_html(include_mathjax='cdn'),height=500)

