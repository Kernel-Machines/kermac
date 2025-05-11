from setuptools import setup, Extension, find_packages
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

NAME="kermac"
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
PY_SRC_DIR = os.path.join(BASE_DIR, 'kermac','csrc')
INCLUDE_DIRS = [
    os.path.join(BASE_DIR, 'kermac','csrc'),
    os.path.join(BASE_DIR, 'thirdparty','cutlass-stripped','include')
]
SOURCES = [
    os.path.join(PY_SRC_DIR, 'bindings.cpp'),
    os.path.join(PY_SRC_DIR, 'p_norm.cu'),
    os.path.join(PY_SRC_DIR, 'utils.cu'),
]
EXTRA_LINK_ARGS = []

USE_DEBUG = os.getenv('USE_KERMAC_DEBUG', '0') == '1'

def get_cxx_compile_args():
    if USE_DEBUG:
        return ['-g']
    return [
        '-O3',
    ]

def get_nvcc_compile_args():
    if USE_DEBUG:
        return ['-g']
    return [
        '-Xptxas=-v',
        '--expt-relaxed-constexpr',
        '-Xcompiler=-fno-strict-aliasing',
        '--expt-extended-lambda',
        '-O3',
    ]

EXTRA_COMPILE_ARGS = {
    'cxx': get_cxx_compile_args(),
    'nvcc': get_nvcc_compile_args()
}

def get_ext_modules():
    return [
        CUDAExtension(
            name= NAME,
            sources=SOURCES,
            include_dirs=INCLUDE_DIRS,
            extra_compile_args=EXTRA_COMPILE_ARGS,
            extra_link_args=EXTRA_LINK_ARGS,
        ),
    ]

with open('requirements.txt') as f:
    requirements = f.read().splitlines()

setup(
    name='kermac',
    version='0.1',
    packages=find_packages(),
    description='PyTorch extension with C++ and CUDA',
    ext_modules=get_ext_modules(),
    cmdclass={
        'build_ext': BuildExtension,
    },
    install_requires=requirements,
)