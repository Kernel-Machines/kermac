from pathlib import Path
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import os

ROOT = Path(__file__).parent.resolve()        # directory that contains setup.py

def rel(*parts):
    """Convert project-relative paths to absolute ones."""
    return str(ROOT.joinpath(*parts))

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
            name="kermac._cuda_extension",
            sources=[
                os.path.join('csrc', 'bindings.cpp'),
                os.path.join('csrc', 'p_norm_pytorch.cu'),
            ],
            include_dirs=[
                rel("csrc"),
                rel("csrc", "include"),
                rel("thirdparty", "cutlass-stripped", "include"),
            ],
            extra_compile_args=EXTRA_COMPILE_ARGS,
            extra_link_args=[],
        ),
    ]

with open('requirements.txt') as f:
    requirements = f.read().splitlines()

setup(
    name='kermac',
    version='0.1',
    package_dir={"": "src"},
    packages=["kermac"],
    description='(Ker)nel (Mac)hines. CUDA routines for Nvidia cards.',
    ext_modules=get_ext_modules(),
    cmdclass={
        'build_ext': BuildExtension,
    },
    install_requires=open("requirements.txt").read().splitlines(),
    include_package_data=True
)