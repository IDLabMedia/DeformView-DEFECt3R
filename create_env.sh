#!/bin/bash
# Creates the conda environment for DEFECt3R, MEt3R and TSED (and the evaluation).
#
#   bash create_env.sh
#   conda activate ./env/DEFECt3R
#
# Requires conda and an NVIDIA GPU driver supporting CUDA 12.1. PyTorch3D and FeatUp
# are compiled from source against the CUDA toolkit installed in the environment.

set -e

cd "$(dirname "$0")"
ROOT=$(pwd)
ENV=$ROOT/env/DEFECt3R
PYTHON=$ENV/bin/python

mkdir -p env
mkdir -p env/conda/envs
mkdir -p env/conda/pkgs
mkdir -p env/pip_cache

conda config --add envs_dirs $ROOT/env/conda/envs
conda config --add pkgs_dirs $ROOT/env/conda/pkgs
export PIP_CACHE_DIR=$ROOT/env/pip_cache
# Install into the environment only: fail instead of falling back to ~/.local,
# and ignore packages that are already in ~/.local
export PIP_USER=0
export PYTHONNOUSERSITE=1

conda config --show envs_dirs
conda config --show pkgs_dirs

eval "$(conda shell.bash hook)"

if [ ! -f "$ENV/bin/python" ]; then
    echo "Creating environment..."
    conda create -y --prefix $ENV python=3.10
else
    echo "Environment already exists."
fi
conda activate $ENV

echo "Updating packages..."

# PyTorch (CUDA 12.1 wheels; they ship their own CUDA runtime libraries)
$PYTHON -m pip install \
    torch==2.5.1 torchvision==0.20.1 \
    --index-url https://download.pytorch.org/whl/cu121

# CUDA 12.1 compiler + headers, only needed to compile PyTorch3D, FeatUp and curope.
# The version label and pins keep every component at 12.1 (the plain nvidia /
# conda-forge channels would pull the newest 13.x packages).
conda install -y \
    -c nvidia/label/cuda-12.1.1 -c nvidia \
    cuda-version=12.1 cuda-nvcc=12.1 cuda-cudart-dev=12.1 cuda-cccl=12.1 cuda-libraries-dev=12.1
# Host compiler supported by nvcc 12.1 (gcc <= 12)
conda install -y -c conda-forge gcc_linux-64=11 gxx_linux-64=11
conda deactivate
conda activate $ENV   # re-activate so CC / CXX point to the conda compilers

export CUDA_HOME=$CONDA_PREFIX
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

# Check the toolchain before compiling anything
NVCC_CUDA=$(nvcc --version | sed -n 's/.*release \([0-9]*\.[0-9]*\).*/\1/p')
TORCH_CUDA=$($PYTHON -c "import torch; print(torch.version.cuda)")
echo "nvcc: CUDA $NVCC_CUDA   PyTorch: CUDA $TORCH_CUDA   CC: ${CC:-system}"
if [ "$NVCC_CUDA" != "$TORCH_CUDA" ]; then
    echo "ERROR: nvcc (CUDA $NVCC_CUDA) does not match PyTorch (CUDA $TORCH_CUDA)."
    exit 1
fi

# DEFECt3R / MEt3R / MASt3R / TSED / evaluation
$PYTHON -m pip install \
    jaxtyping \
    einops \
    tqdm \
    numpy \
    scipy \
    scikit-learn \
    pillow \
    matplotlib \
    "opencv-python-headless<5" \
    roma \
    trimesh \
    "huggingface-hub[torch]>=0.22"
# headless OpenCV: no libGL needed (GPU servers); install only one OpenCV package

# PyTorch3D (point rendering for MASt3R correspondences) and FeatUp (DINO upsampling)
$PYTHON -m pip install --no-build-isolation \
    git+https://github.com/facebookresearch/pytorch3d.git
$PYTHON -m pip install --no-build-isolation \
    git+https://github.com/mhamilton723/FeatUp

# # MASt3R submodule
# git submodule update --init --recursive

# # Optional: CUDA kernels for the RoPE embeddings of MASt3R (faster; falls back to PyTorch otherwise)
# (cd DEFECt3R/third_party/mast3r/dust3r/croco/models/curope && $PYTHON setup.py build_ext --inplace) \
#     || echo "curope build failed; MASt3R will use the slower PyTorch RoPE implementation."

echo "Done. Activate with:  conda activate $ENV"
