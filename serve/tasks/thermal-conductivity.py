from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from mlip_arena.models import REGISTRY as MODELS



st.markdown(
"""
# Thermal Conductivity

Compared to Póta, Ahlawat, Csányi, and Simoncelli, [arXiv:2408.00755v4](https://arxiv.org/abs/2408.00755), the relaxation protocol has been updated and unified for all the models. The relaxation is a combination of sequential vc-relax (changes cell and atom positions) and relax (changes atom positions only). Each relaxation stage has a maximum number of 300 steps, and consist of a single FrechetCellFilter relaxation with force threshold =1e-4 eV/Ang. To preserve crystal symmetry, unit-cell angles are not allowed to change. This unified protocol gives the same SRME reported in [arXiv:2408.00755v4](https://arxiv.org/abs/2408.00755) for all the models but M3GNet. In M3GNet this updated relaxation protocol gives SRME = 1.412, slightly smaller than the value 1.469 that was obtained with the non-unified relaxation protocol in [arXiv:2408.00755v4](https://arxiv.org/abs/2408.00755). 
"""
)