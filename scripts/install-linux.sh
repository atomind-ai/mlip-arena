TORCH=2.6
CUDA=cu124
uv pip install torch==${TORCH}.0
uv pip install torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-${TORCH}.0+${CUDA}.html
uv pip install dgl -f https://data.dgl.ai/wheels/torch-${TORCH}/${CUDA}/repo.html
uv pip install -e .[fairchem] --no-cache
uv pip install -e .[orb] --no-cache
uv pip install -e .[matgl] --no-cache
uv pip install -e .[test,extra] --no-cache
uv pip install -e .[mace] --no-cache