TORCH=2.2
CUDA=cu121
uv pip install torch==${TORCH}.0
uv pip install torch-scatter -f https://data.pyg.org/whl/torch-${TORCH}.0+${CUDA}.html
uv pip install torch-sparse -f https://data.pyg.org/whl/torch-${TORCH}.0+${CUDA}.html
uv pip install dgl -f https://data.dgl.ai/wheels/torch-${TORCH}/${CUDA}/repo.html
uv pip install -e .[test]
uv pip install -e .[mace]