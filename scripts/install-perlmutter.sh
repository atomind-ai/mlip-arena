TORCH=2.9
CUDA=cu128
uv pip install torch==${TORCH}.0
uv pip install dgl -f https://data.dgl.ai/wheels/torch-${TORCH}/${CUDA}/repo.html
uv pip install -e  .[fairchem] --no-cache
uv pip install -e  .[orb] --no-cache
uv pip install -e  .[matgl] --no-cache
uv pip install -e  .[test,extra,all] --no-cache
uv pip install -e  .[mace] --no-cache
