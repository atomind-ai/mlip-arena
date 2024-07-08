

# PyTorch Geometric (OCP)
TORCH=2.2.0
CUDA=cu121

pip install --verbose --no-cache torch-scatter -f https://data.pyg.org/whl/torch-${TORCH}+${CUDA}.html
pip install --verbose --no-cache torch-sparse -f https://data.pyg.org/whl/torch-${TORCH}+${CUDA}.html

# DGL (M3GNet)
pip install --verbose --no-cache dgl -f https://data.dgl.ai/wheels/{CUDA}/repo.html


# DGL (ALIGNN)
# pip install --verbose --no-cache dgl -f https://data.dgl.ai/wheels/torch-2.2/cu122/repo.html
