from pathlib import Path

import numpy as np
import pandas as pd
import plotly.colors as pcolors
import plotly.graph_objects as go
import streamlit as st
from ase.db import connect
from scipy import stats

from mlip_arena.models import REGISTRY as MODELS

DATA_DIR = Path("benchmarks/wbm_ev")

st.markdown("""
# Energy-volume scans
""")

# Control panels at the top
st.markdown("### Methods")
methods_container = st.container(border=True)

# Get valid models that support wbm_ev
valid_models = [
    model
    for model, metadata in MODELS.items()
    if Path(__file__).stem in metadata.get("gpu-tasks", [])
]

# Model selection
selected_models = methods_container.multiselect(
    "Select Models",
    options=valid_models,
    default=valid_models
)

# Visualization settings
st.markdown("### Visualization Settings")
vis = st.container(border=True)

# Column settings
ncols = vis.select_slider("Number of columns", options=[1, 2, 3, 4], value=2)

# Color palette selection
all_attributes = dir(pcolors.qualitative)
color_palettes = {
    attr: getattr(pcolors.qualitative, attr)
    for attr in all_attributes
    if isinstance(getattr(pcolors.qualitative, attr), list)
}
color_palettes.pop("__all__", None)

palette_names = list(color_palettes.keys())
palette_name = vis.selectbox("Color sequence", options=palette_names, index=22)
color_sequence = color_palettes[palette_name]

# Stop execution if no models selected
if not selected_models:
    st.warning("Please select at least one model to visualize.")
    st.stop()


def load_wbm_structures():
    """
    Load the WBM structures from a ASE DB file.
    """
    with connect(DATA_DIR.parent / "wbm_structures.db") as db:
        for row in db.select():
            yield row.toatoms(add_additional_information=True)


@st.cache_data
def generate_dataframe(model_name):
    fpath = DATA_DIR / f"{model_name}.parquet"
    if not fpath.exists():
        return pd.DataFrame()  # Return empty dataframe instead of using continue

    df_raw_results = pd.read_parquet(fpath)

    df_analyzed = pd.DataFrame(
        columns=[
            "model",
            "structure",
            "formula",
            "volume-ratio",
            "energy-delta-per-atom",
            "energy-diff-flip-times",
            "tortuosity",
            "spearman-compression-energy",
            "spearman-compression-derivative",
            "spearman-tension-energy",
            "missing",
        ]
    )

    for wbm_struct in load_wbm_structures():
        structure_id = wbm_struct.info["key_value_pairs"]["wbm_id"]

        try:
            results = df_raw_results.loc[df_raw_results["id"] == structure_id]
            results = results["eos"].values[0]
            es = np.array(results["energies"])
            vols = np.array(results["volumes"])
            vol0 = wbm_struct.get_volume()

            indices = np.argsort(vols)
            vols = vols[indices]
            es = es[indices]

            imine = len(es) // 2
            # min_center_val = np.min(es[imid - 1 : imid + 2])
            # imine = np.where(es == min_center_val)[0][0]
            emin = es[imine]

            interpolated_volumes = [
                (vols[i] + vols[i + 1]) / 2 for i in range(len(vols) - 1)
            ]
            ediff = np.diff(es)
            ediff_sign = np.sign(ediff)
            mask = ediff_sign != 0
            ediff = ediff[mask]
            ediff_sign = ediff_sign[mask]
            ediff_flip = np.diff(ediff_sign) != 0

            etv = np.sum(np.abs(np.diff(es)))

            data = {
                "model": model_name,
                "structure": structure_id,
                "formula": wbm_struct.get_chemical_formula(),
                "missing": False,
                "volume-ratio": vols / vol0,
                "energy-delta-per-atom": (es - emin) / len(wbm_struct),
                "energy-diff-flip-times": np.sum(ediff_flip).astype(int),
                "tortuosity": etv / (abs(es[0] - emin) + abs(es[-1] - emin)),
                "spearman-compression-energy": stats.spearmanr(
                    vols[:imine], es[:imine]
                ).statistic,
                "spearman-compression-derivative": stats.spearmanr(
                    interpolated_volumes[:imine], ediff[:imine]
                ).statistic,
                "spearman-tension-energy": stats.spearmanr(
                    vols[imine:], es[imine:]
                ).statistic,
            }

        except Exception:
            data = {
                "model": model_name,
                "structure": structure_id,
                "formula": wbm_struct.get_chemical_formula(),
                "missing": True,
                "volume-ratio": None,
                "energy-delta-per-atom": None,
                "energy-diff-flip-times": None,
                "tortuosity": None,
                "spearman-compression-energy": None,
                "spearman-compression-derivative": None,
                "spearman-tension-energy": None,
            }

        df_analyzed = pd.concat([df_analyzed, pd.DataFrame([data])], ignore_index=True)

    return df_analyzed


@st.cache_data
def get_plots(selected_models):
    """Generate one plot per model with all structures (legend disabled for each structure)."""
    figs = []

    for model_name in selected_models:

        fpath = DATA_DIR / f"{model_name}_processed.parquet"
        if not fpath.exists():
            df = generate_dataframe(model_name)
        else:
            df = pd.read_parquet(fpath)

        if len(df) == 0:
            continue

        fig = go.Figure()
        valid_structures = []
        for i, (_, row) in enumerate(df.iterrows()):
            structure_id = row["structure"]
            formula = row.get("formula", "")
            if isinstance(row["volume-ratio"], list | np.ndarray) and isinstance(
                row["energy-delta-per-atom"], list | np.ndarray
            ):
                vol_strain = row["volume-ratio"]
                energy_delta = row["energy-delta-per-atom"]
                color = color_sequence[i % len(color_sequence)]
                fig.add_trace(
                    go.Scatter(
                        x=vol_strain,
                        y=energy_delta,
                        mode="lines",
                        name=f"{structure_id}",
                        showlegend=False,
                        line=dict(color=color),
                        hoverlabel=dict(bgcolor=color, font=dict(color="black")),
                        hovertemplate=(
                            structure_id + "<br>"
                            "Formula: " + str(formula) + "<br>"
                            "Volume ratio V/V₀: %{x:.3f}<br>"
                            "ΔEnergy: %{y:.3f} eV/atom<br>"
                            "<extra></extra>"
                        ),

                    )
                )
                valid_structures.append(structure_id)

        # if valid_structures:
        fig.update_layout(
            title=f"{model_name} ({len(valid_structures)} / {len(df)} structures)",
            xaxis_title="Volume ratio V/V₀",
            yaxis_title="Relative energy E - E₀ (eV/atom)",
            height=500,
            showlegend=False,  # Disable legend for the whole plot
            yaxis=dict(range=[-1, 15]),  # Set y-axis limits
        )
        fig.add_vline(x=1, line_dash="dash", line_color="gray", opacity=0.7)
        figs.append((model_name, fig, valid_structures))

    return figs


# Generate all plots
all_plots = get_plots(selected_models)

# Display plots in the specified column layout
if all_plots:
    for i, (model_name, fig, structures) in enumerate(all_plots):
        if i % ncols == 0:
            cols = st.columns(ncols)
        cols[i % ncols].plotly_chart(fig, use_container_width=True)

        # Display number of structures in this plot
        # cols[i % ncols].caption(f"{len(structures)} / 1000 structures")
else:
    st.warning("No data available for the selected models.")
