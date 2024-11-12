# PyTorch Geometric (OCP)
TORCH=2.2.0
CUDA=cu121

uv pip install --verbose --no-cache torch-scatter -f https://data.pyg.org/whl/torch-${TORCH}+${CUDA}.html
uv pip install --verbose --no-cache torch-sparse -f https://data.pyg.org/whl/torch-${TORCH}+${CUDA}.html
