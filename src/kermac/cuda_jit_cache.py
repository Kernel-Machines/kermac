import threading
from typing import Dict, Any, Tuple
import sys
import gzip
import io

from cuda.core.experimental import Device, Program, ProgramOptions, ObjectCode

from .paths import *

class Singleton(type):
    """Metaclass for creating singleton classes."""
    _instances = {}
    _lock = threading.Lock()

    def __call__(cls, *args, **kwargs):
        with cls._lock:
            if cls not in cls._instances:
                cls._instances[cls] = super().__call__(*args, **kwargs)
        return cls._instances[cls]
    
class DeviceModuleMap(metaclass=Singleton):
    """Singleton class mapping device IDs to lazily loaded modules."""
    
    def __init__(self):
        self._modules: Dict[Tuple[int, str], Any] = {}  # device_id -> module
        self._lock = threading.Lock()

    def get_module(self, device: Device, module_name : str, debug = False) -> Any:
        device_id = device.device_id
        if device_id < 0:
            raise ValueError(f"Invalid device ID: {device_id}")

        key = (device_id, module_name)
        with self._lock:
            if key not in self._modules:
                arch = "".join(f"{i}" for i in device.compute_capability)
                package_name = get_package_name()
                package_version = get_package_version()
                cubin_path = get_cache_cubin_dir() / f'{package_name}.{package_version}.{arch}.{module_name}.cubin'
                if debug:
                    print(f'(Kermac Debug) Loaded module not found for (device:{device_id}, module:{module_name}), looking for pre-built cubin')
                if cubin_path.is_file():
                    if debug:
                        print(f'(Kermac Debug) Found matching cubin at {cubin_path}')
                    module_cubin = ObjectCode.from_cubin(str(cubin_path))
                else:
                    if debug:
                        print(f'(Kermac Debug) Cubin not found: {cubin_path}')
                    if True: # Generate from cuda src
                        cuda_code_path = get_local_cuda_src_dir() / 'p_norm.cu'
                        with open(cuda_code_path, "r", encoding="utf-8") as f:
                            code = f.read()  # Read as text
                            module_cubin = Program(
                                code, 
                                code_type="c++", 
                                options= \
                                    ProgramOptions(
                                        std="c++17",

                                        arch=f"sm_{arch}",
                                        device_as_default_execution_space=True,
                                        
                                        include_path=[
                                            get_include_local_cuda_dir(),       # *.cuh
                                            get_include_dir_cutlass(),          # main cutlass include
                                            get_include_dir_cutlass_tools(),    # cutlass tools include
                                            get_include_dir_cuda()              # cuda toolkit for <cuda/src/assert>, etc.. (dependency of cutlass)
                                        ],
                                    )
                            ).compile(
                                "cubin", 
                                logs=sys.stdout,
                            )
                            with open(cubin_path, 'wb') as file:
                                file.write(module_cubin.code)
                    else: # Generate from ptx instead
                        ptx_code_gz_path = get_local_ptx_src_dir() / f'{module_name}.ptx.gz'
                        if debug:
                            print(f'(Kermac Debug) Looking for compressed pre-built ptx at {ptx_code_gz_path}')
                        with open(ptx_code_gz_path, "rb") as file:
                            compressed_data = file.read()
                        with gzip.GzipFile(fileobj=io.BytesIO(compressed_data), mode='rb') as gz:
                            ptx_bytes = gz.read()
                            module_ptx_deserialized = ObjectCode.from_ptx(ptx_bytes)
                        if debug:
                            print(f'(Kermac Debug) Found compressed pre-built ptx, compiling..')
                        # Compile ptx to cubin
                        module_cubin = \
                            Program(
                                module_ptx_deserialized.code.decode(), 
                                code_type="ptx", 
                                options= \
                                    ProgramOptions(
                                        arch=f"sm_{arch}"
                                    )
                            ).compile("cubin", logs=sys.stdout)
                        with open(cubin_path, 'wb') as file:
                            file.write(module_cubin.code)
                        if debug:
                            print(f'(Kermac Debug) Done compiling, storing cubin at {cubin_path}')
                            
                self._modules[key] = module_cubin
                return module_cubin
            else:
                if debug:
                    print(f'(Kermac Debug) Loaded module found for (device:{device_id}, module:{module_name})')
            return self._modules[key]