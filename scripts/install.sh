TORCH=2.4
CUDA=cu124
uv pip install torch==${TORCH}.0
uv pip install torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-${TORCH}.0+${CUDA}.html
uv pip install dgl -f https://data.dgl.ai/wheels/torch-${TORCH}/${CUDA}/repo.html
uv pip install mlip-arena[fairchem]
uv pip install mlip-arena[orb]
uv pip install mlip-arena[matgl]
uv pip install mlip-arena[test]
uv pip install mlip-arena[mace]