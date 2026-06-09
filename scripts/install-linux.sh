GROUP=${1:-default}

if [ "$GROUP" == "nequip" ]; then
    TORCH=2.5
    CUDA=cu124
    uv pip install torch==${TORCH}.0
    uv pip install -e .[test,extra,nequip] --no-cache
elif [ "$GROUP" == "sevennet" ]; then
    TORCH=2.8
    CUDA=cu129
    uv pip install torch==${TORCH}.0
    uv pip install -e .[test,extra,sevennet] --no-cache
elif [ "$GROUP" == "mace" ]; then
    TORCH=2.8
    CUDA=cu129
    uv pip install torch==${TORCH}.0
    uv pip install -e .[test,extra,mace] --no-cache
elif [ "$GROUP" == "fairchem" ]; then
    TORCH=2.8
    CUDA=cu129
    uv pip install torch==${TORCH}.0
    uv pip install -e .[test,extra,fairchem] --no-cache
else
    TORCH=2.8
    CUDA=cu129
    uv pip install torch==${TORCH}.0
    uv pip install dgl -f https://data.dgl.ai/wheels/torch-${TORCH}/${CUDA}/repo.html
    uv pip install -e .[orb] --no-cache
    uv pip install -e .[matgl] --no-cache
    uv pip install -e .[test,extra,all] --no-cache
fi
