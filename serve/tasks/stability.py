from pathlib import Path

import numpy as np
import pandas as pd
import plotly.colors as pcolors
import plotly.graph_objects as go
import streamlit as st
from ase.data import chemical_symbols
from ase.io import read, write
from plotly.subplots import make_subplots
from scipy.interpolate import CubicSpline

from mlip_arena.models.utils import MLIPMap

st.markdown("# Stability")

st.markdown("### Methods")
container = st.container(border=True)
methods = container.multiselect("MLIPs", ["MACE-MP", "Equiformer", "CHGNet", "MACE-OFF", "eSCN", "ALIGNN"], ["MACE-MP", "Equiformer", "CHGNet", "eSCN", "ALIGNN"])


DATA_DIR = Path("mlip_arena/tasks/stability")






