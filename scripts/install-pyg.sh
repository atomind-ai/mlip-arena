

# PyTorch Geometric (OCP)
TORCH=2.3.1
CUDA=cu121

pip install --verbose --no-cache torch-scatter -f https://data.pyg.org/whl/torch-${TORCH}+${CUDA}.html
pip install --verbose --no-cache torch-sparse -f https://data.pyg.org/whl/torch-${TORCH}+${CUDA}.html
