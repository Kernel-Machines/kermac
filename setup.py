# setup.py
from pathlib import Path
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import os

ROOT = Path(__file__).parent.resolve()

def abspath(*parts):           # helper for absolute -I flags
    return str(ROOT.joinpath(*parts))


USE_DEBUG = os.getenv("USE_KERMAC_DEBUG") == "1"

EXTRA_COMPILE_ARGS = {
    "cxx": ["-g"] if USE_DEBUG else ["-O3"],
    "nvcc": (["-g"] if USE_DEBUG else [
        "-Xptxas=-v",
        "--expt-relaxed-constexpr",
        "--expt-extended-lambda",
        "-Xcompiler=-fno-strict-aliasing",
        "-O3",
    ]),
}

setup(
    ext_modules=[
        CUDAExtension(
            name="kermac._cuda_extension",
            sources=[
                "csrc/bindings.cpp",
                "csrc/p_norm_pytorch.cu",
            ],
            include_dirs=[
                abspath("csrc"),
                abspath("csrc", "include"),
                abspath("thirdparty", "cutlass-stripped", "include"),
            ],
            extra_compile_args=EXTRA_COMPILE_ARGS,
        )
    ],
    cmdclass={"build_ext": BuildExtension},
    include_package_data=True,   # tell wheels to keep the headers
)
