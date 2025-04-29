TORCH=2.4
CUDA=cu124
uv pip install torch==${TORCH}.0
uv pip install torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-${TORCH}.0+${CUDA}.html
uv pip install dgl -f https://data.dgl.ai/wheels/torch-${TORCH}/${CUDA}/repo.html
uv pip install -e .[fairchem]
uv pip install -e .[orb]
uv pip install -e .[matgl]
uv pip install -e .[test]
uv pip install -e .[mace]