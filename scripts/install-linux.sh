GROUP=${1:-default}

install_pyg_extensions() {
    local torch_ver=$1
    local cuda_ver=$2
    local python_ver
    python_ver=$(python -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
    if [ "$python_ver" != "3.13" ]; then
        uv pip install torch-scatter torch-sparse -f "https://data.pyg.org/whl/torch-${torch_ver}.0+${cuda_ver}.html"
    else
        echo "Skipping torch-scatter and torch-sparse on Python 3.13 due to lack of prebuilt wheels."
    fi
}

if [ "$GROUP" == "nequip" ]; then
    TORCH=2.5
    CUDA=cu124
    uv pip install torch==${TORCH}.0
    install_pyg_extensions ${TORCH} ${CUDA}
    uv pip install dgl -f https://data.dgl.ai/wheels/torch-${TORCH}/${CUDA}/repo.html
    uv pip install -e .[test,extra,nequip] --no-cache
elif [ "$GROUP" == "sevennet" ]; then
    TORCH=2.8
    CUDA=cu128
    uv pip install torch==${TORCH}.0
    install_pyg_extensions ${TORCH} ${CUDA}
    uv pip install dgl -f https://data.dgl.ai/wheels/torch-${TORCH}/${CUDA}/repo.html
    uv pip install -e .[test,extra,sevennet] --no-cache
elif [ "$GROUP" == "mace" ]; then
    TORCH=2.8
    CUDA=cu128
    uv pip install torch==${TORCH}.0
    install_pyg_extensions ${TORCH} ${CUDA}
    uv pip install dgl -f https://data.dgl.ai/wheels/torch-${TORCH}/${CUDA}/repo.html
    uv pip install -e .[test,extra,mace] --no-cache
elif [ "$GROUP" == "fairchem" ]; then
    TORCH=2.8
    CUDA=cu128
    uv pip install torch==${TORCH}.0
    install_pyg_extensions ${TORCH} ${CUDA}
    uv pip install dgl -f https://data.dgl.ai/wheels/torch-${TORCH}/${CUDA}/repo.html
    uv pip install -e .[test,extra,fairchem] --no-cache
else
    TORCH=2.8
    CUDA=cu128
    uv pip install torch==${TORCH}.0
    install_pyg_extensions ${TORCH} ${CUDA}
    uv pip install dgl -f https://data.dgl.ai/wheels/torch-${TORCH}/${CUDA}/repo.html
    uv pip install -e .[orb] --no-cache
    uv pip install -e .[matgl] --no-cache
    uv pip install -e .[test,extra,all] --no-cache
fi
