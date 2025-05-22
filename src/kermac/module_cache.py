import threading
from typing import Dict, Any, Tuple
import sys
import torch

from cuda.core.experimental import Device, Program, ProgramOptions, ObjectCode

from .paths import *
from .disk_cache import *
from .common import hash_text_files

def get_compute_capability(device : Device) -> str:
    arch = "".join(f"{i}" for i in device.compute_capability)
    return arch

class Singleton(type):
    """Metaclass for creating singleton classes."""
    _instances = {}
    _lock = threading.Lock()

    def __call__(cls, *args, **kwargs):
        with cls._lock:
            if cls not in cls._instances:
                cls._instances[cls] = super().__call__(*args, **kwargs)
        return cls._instances[cls]
    
class DeviceLoadedFunctionMap(metaclass=Singleton):
    """Singleton class mapping device IDs to lazily loaded modules/functions."""
    
    def __init__(self, debug = False):
        self._functions: Dict[Tuple[int, str], Any] = {}  # device_id -> module
        self._lock = threading.Lock()
        directory = get_include_local_cuda_dir()
        hash_result = hash_text_files(directory)
        print(f"Combined hash of text files: {hash_result}")
        self._db = DiskCache(
            cache_dir=str(cache_root().resolve()),
            max_size_mb=1024,
            db_name='cubin_cache',
            current_file_src_hash=hash_result,
            debug=debug
        )
        self._cuda_version = str(torch.version.cuda)
        if debug:
            print(f'(Kermac Debug) Using database at: {cache_root().resolve()}')

        # Example usage
        

    def get_function(self, device: Device, function_name : str, debug = False) -> Any:
        device_id = device.device_id
        if device.compute_capability.major < 8:
            raise ValueError(f"Invalid device compute capability, (device:{device.compute_capability}, requrires at least:8.0")

        key = (device_id, function_name)
        with self._lock:
            if key in self._functions:
                if debug: 
                    print(f'(Kermac Debug) Loaded function found for (device:{device_id}, function:{function_name})')
                return self._functions[key]

            if debug: 
                print(f'(Kermac Debug) Loaded module not found for (device:{device_id}, function:{function_name})')
            arch = get_compute_capability(device)
            cubin_db_key = {
                'package_name':     get_package_name(),
                'package_version':  get_package_version(),
                'cuda_version':     self._cuda_version,
                'arch':             arch,
                'function_name':    function_name
            }

            result = self._db.lookup(cubin_db_key)
            if result:
                if debug: 
                    print(f'(Kermac Debug) Found pre-built cubin: {cubin_db_key}')
                lowered_symbol, cubin_code = result
                symbol_map = {function_name: lowered_symbol}
                module_cubin = ObjectCode.from_cubin(cubin_code, symbol_mapping=symbol_map)
            else:
                if debug: 
                    print(f'(Kermac Debug) No pre-built cubin, building: {cubin_db_key}')
                module_cubin = Program(
                    '#include <kermac.cuh>',
                    code_type="c++", 
                    options= \
                        ProgramOptions(
                            std="c++17",
                            arch=f"sm_{arch}",
                            device_as_default_execution_space=True,
                            # diag_suppress cutlass: 64-D: declaration does not declare anything
                            # diag_suppress cutlass: 1055-D: declaration does not declare anything
                            diag_suppress=[64,1055],
                            ptxas_options=['-v'] if debug else None,
                            # some good ones
                            # device_code_optimize=True,
                            # extensible_whole_program=True,
                            # ftz=True,
                            # extra_device_vectorization=True,
                            # restrict=True,
                            # use_fast_math=True,
                            # prec_sqrt=False,
                            # prec_div=False,
                            # split_compile=8,
                            include_path=[
                                get_include_local_cuda_dir(),   # include/*.cuh
                                get_include_dir_cutlass(),      # thirdparty/cutlass/include
                                get_include_dir_cuda()          # cuda toolkit for <cuda/src/assert>, etc.. (dependency of cutlass)
                            ],
                        )
                ).compile(
                    "cubin", 
                    logs=sys.stdout,
                    name_expressions=[function_name]
                )
                cubin_sym_map_lowered_name = module_cubin._sym_map[function_name]
                cubin_db_value = (cubin_sym_map_lowered_name, module_cubin.code)
                self._db.store(cubin_db_key, cubin_db_value)
                if debug: 
                    print(f'(Kermac Debug) Built and Saved: {cubin_db_key}')
            function = module_cubin.get_kernel(function_name)
            self._functions[key] = function
            return function
            
