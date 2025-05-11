from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import os

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

BASE_DIR = os.path.abspath(os.path.dirname(__file__))

def get_ext_modules():
    return [
        CUDAExtension(
            name="kermac._cuda_extension",
            sources=[
                os.path.join('csrc', 'bindings.cpp'),
                os.path.join('csrc', 'p_norm_pytorch.cu'),
            ],
            include_dirs=[
                os.path.join(BASE_DIR, 'csrc'),
                os.path.join(BASE_DIR, 'csrc', 'include'),
                os.path.join(BASE_DIR, 'thirdparty','cutlass-stripped','include')
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
    description='PyTorch extension with C++ and CUDA',
    ext_modules=get_ext_modules(),
    cmdclass={
        'build_ext': BuildExtension,
    },
    install_requires=requirements,
)