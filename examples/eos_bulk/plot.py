from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from ase.db import connect

from mlip_arena.models import REGISTRY as MODELS

DATA_DIR = Path(__file__).parent.absolute()

# Use a qualitative color palette from matplotlib
palette_name = "tab10"  # Better for distinguishing multiple lines
color_sequence = plt.get_cmap(palette_name).colors

valid_models = [
    model
    for model, metadata in MODELS.items()
    if "eos_bulk" in metadata.get("gpu-tasks", [])
]

def load_wbm_structures():
    """
    Load the WBM structures from a ASE DB file.
    """
    with connect(DATA_DIR.parent / "wbm_structures.db") as db:
        for row in db.select():
            yield row.toatoms(add_additional_information=True)

# # Collect valid models first
# valid_models = []
# for model_name in valid_models:
#     fpath = DATA_DIR / f"{model_name}_processed.parquet"
#     if fpath.exists():
#         df = pd.read_parquet(fpath)
#         if len(df) > 0:
#             valid_models.append(model)

# # Ensure we're showing all 8 models
# if len(valid_models) < 8:
#     print(f"Warning: Only found {len(valid_models)} valid models instead of 8")

# Set up the grid layout
n_models = len(valid_models)
n_cols = 4  # Use 4 columns
n_rows = (n_models + n_cols - 1) // n_cols  # Ceiling division to get required rows

# Create figure with enough space for all subplots
fig = plt.figure(
    figsize=(6, 1.25 * n_rows),  # Wider for better readability
    constrained_layout=True,  # Better than tight_layout for this case
)

# Create grid of subplots
axes = []
for i in range(n_models):
    ax = plt.subplot(n_rows, n_cols, i+1)
    axes.append(ax)

SMALL_SIZE = 6
MEDIUM_SIZE = 8
LARGE_SIZE = 10

# Fill in the subplots with data
for i, model_name in enumerate(valid_models):
    fpath = DATA_DIR / f"{model_name}_processed.parquet"
    df = pd.read_parquet(fpath)

    ax = axes[i]
    valid_structures = []

    for j, (_, row) in enumerate(df.iterrows()):
        structure_id = row["structure"]
        formula = row.get("formula", "")
        if isinstance(row["volume-ratio"], (list, np.ndarray)) and isinstance(
            row["energy-delta-per-volume-b0"], (list, np.ndarray)
        ):
            vol_strain = row["volume-ratio"]
            energy_delta = row["energy-delta-per-volume-b0"]
            color = color_sequence[j % len(color_sequence)]
            ax.plot(
                vol_strain,
                energy_delta,
                color=color,
                linewidth=1,
                alpha=0.9,
            )
            valid_structures.append(structure_id)

    # Set subplot title
    ax.set_title(f"{model_name} ({len(valid_structures)})", fontsize=MEDIUM_SIZE)
    
    # Only add y-label to leftmost plots (those with index divisible by n_cols)
    if i % n_cols == 0:
        ax.set_ylabel("$\\frac{\\Delta E}{B V_0}$", fontsize=MEDIUM_SIZE)
    else:
        ax.set_ylabel("")
    
    # Only add x-label to bottom row plots
    # Check if this plot is in the bottom row
    is_bottom_row = (i // n_cols) == (n_rows - 1) or (i >= n_models - n_cols)
    if is_bottom_row:
        ax.set_xlabel("$V/V_0$", fontsize=MEDIUM_SIZE)
    else:
        ax.set_xlabel("")
    
    ax.set_ylim(-0.02, 0.1)  # Consistent y-limits
    ax.axvline(x=1, linestyle="--", color="gray", alpha=0.7)
    ax.tick_params(axis="both", which="major", labelsize=MEDIUM_SIZE)

# Make sure all subplots share the x and y limits
for ax in axes:
    ax.set_xlim(0.8, 1.2)  # Adjust these as needed
    ax.set_ylim(-0.02, 0.1)

# Save the figure with all plots
plt.savefig(DATA_DIR / "eos-bulk-grid.png", dpi=300, bbox_inches="tight")
plt.savefig(DATA_DIR / "eos-bulk-grid.pdf", bbox_inches="tight")
# plt.show()