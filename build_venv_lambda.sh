#!/bin/bash

# create env
# ---------
uv venv --python 3.11

uv pip install --upgrade pip
uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
uv pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.8.0+cu128.html
uv pip install -r requirements.txt

# test env
# --------

echo 'which python -> venv'
which python

echo 'test_pytorch.py -> random tensor'
uv run test/test_pytorch.py 

echo 'test_pytorch_cuda_gpu.py -> True if GPU'
uv run test/test_pytorch_cuda_gpu.py

echo 'test_torch_geometric.py -> no retun if import sucessful'
uv run test/test_torch_geometric.py

# Clean up
# --------
#deactivate
#rm -r venv
