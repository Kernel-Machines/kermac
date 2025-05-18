#!/usr/bin/env bash
set -euo pipefail

###############################################################################
# CUDA 11.8 Conda toolchain – emits .version 7.8 / .target sm_80              #
# Works on any modern Ubuntu / WSL / container that already has Conda/Mamba.  #
# Sources:  nvidia/label/cuda-11.8.0 channel docs.                           #
###############################################################################

ENV_NAME=cuda118
CUDA_LABEL="cuda-11.8.0"                         # PTX 7.8 lives here
NV_CHANNEL="nvidia/label/${CUDA_LABEL}"

eval "$($HOME/miniconda3/bin/conda shell.bash hook)"

echo ">> Cleaning any previous attempt"
conda env remove -n "$ENV_NAME" -y || true
conda clean --packages --tarballs -y

echo ">> Enforcing strict channel priority"
conda config --set channel_priority strict

echo ">> Creating toolchain environment: $ENV_NAME"
conda create -n "$ENV_NAME" -y \
      -c "$NV_CHANNEL" \
      -c conda-forge \
      cuda-nvcc cuda-cudart-dev cuda-cccl \
      gcc=11 gxx=11                      # GCC ≤11 is supported by nvcc 11.8
