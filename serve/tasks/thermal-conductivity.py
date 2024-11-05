
import pandas as pd
import streamlit as st

from mlip_arena import PKG_DIR


st.markdown(
"""
# Thermal Conductivity

Compared to Póta, Ahlawat, Csányi, and Simoncelli, [arXiv:2408.00755v4](https://arxiv.org/abs/2408.00755), the relaxation protocol has been updated and unified for all the models. The relaxation is a combination of sequential vc-relax (changes cell and atom positions) and relax (changes atom positions only). Each relaxation stage has a maximum number of 300 steps, and consist of a single FrechetCellFilter relaxation with force threshold =1e-4 eV/Ang. To preserve crystal symmetry, unit-cell angles are not allowed to change. This unified protocol gives the same SRME reported in [arXiv:2408.00755v4](https://arxiv.org/abs/2408.00755) for all the models but M3GNet. In M3GNet this updated relaxation protocol gives SRME = 1.412, slightly smaller than the value 1.469 that was obtained with the non-unified relaxation protocol in [arXiv:2408.00755v4](https://arxiv.org/abs/2408.00755). 

**SRME** is the Symmetric Relative Mean Error, defined as the mean of the absolute values of the relative errors of the predictions. Here, it is used to quantify the error on microscopic single-phonon conductivity: 

$$
\\text{SRME}[\\left\lbrace\\mathcal{K}({\\mathbf{q},s)}\\right\\rbrace] = \\frac{2}{N_qV}\\frac{\\sum_{\\mathbf{q}s}|\\mathcal{K}_{\\text{MLIP}}(\\mathbf{q},s) - \\mathcal{K}_{\\text{DFT}}(\\mathbf{q},s)|}{\\kappa_{\\text{MLIP}} + \\kappa_{\\text{DFT}}}
$$
"""
)

DATA_DIR = PKG_DIR / "tasks" / "thermal-conductivity"

table = pd.read_csv(DATA_DIR / "wte.csv")

table.rename(
    columns={
        "method": "Model",
        "srme": "SRME[𝜅]",
    },
    inplace=True,
)

table.set_index("Model", inplace=True)

table.sort_values(["SRME[𝜅]"], ascending=True, inplace=True)

s = (
    table.style.background_gradient(
        cmap="Reds", subset=["SRME[𝜅]"]
    )
    .format("{:.3f}", subset=["SRME[𝜅]"])
)

st.dataframe(
    s,
    use_container_width=True,
)