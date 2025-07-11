

# (Optional) Install uv
# curl -LsSf https://astral.sh/uv/install.sh | sh
# source $HOME/.local/bin/env

TORCH=2.2.0

uv pip install torch==${TORCH}
uv pip install torch-scatter --no-build-isolation
uv pip install torch-sparse --no-build-isolation

uv pip install dgl -f https://data.dgl.ai/wheels/torch-${TORCH}/cpu/repo.html

uv pip install mlip-arena[fairchem]
uv pip install mlip-arena[orb]
uv pip install mlip-arena[matgl]
uv pip install mlip-arena[test]
uv pip install mlip-arena[mace]


