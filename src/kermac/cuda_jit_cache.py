import threading
from typing import Dict, Any, Tuple
import sys
import gzip
import io
import torch

from cuda.core.experimental import Device, Program, ProgramOptions, ObjectCode

from .paths import *
from .disk_cache import *

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
        self._db = DiskCache(
            cache_dir=str(cache_root().resolve()),
            max_size_mb=1024,
            db_name='cubin_cache',
        )
        self._cuda_version = str(torch.version.cuda)

    def get_module(self, device: Device, function_name : str, storage_name : str, debug = False) -> Any:
        device_id = device.device_id
        if device_id < 0:
            raise ValueError(f"Invalid device ID: {device_id}")

        key = (device_id, function_name)
        with self._lock:
            if key not in self._modules:
                arch = "".join(f"{i}" for i in device.compute_capability)
                package_name = get_package_name()
                package_version = get_package_version()
                
                cubin_key = {
                    'package_name':     package_name,
                    'package_version':  package_version,
                    'cuda_version':     self._cuda_version,
                    'arch':             arch,
                    'function_name':    function_name
                }
                if debug:
                    print(f'(Kermac Debug) Loaded module not found for (device:{device_id}, function:{function_name})')
                result = self._db.lookup(cubin_key)
                if result:
                    lowered_symbol, cubin_code = result
                    symbol_map = {function_name: lowered_symbol}
                    if debug:
                        print(f'(Kermac Debug) Found pre-built cubin: {cubin_key}')
                    module_cubin = ObjectCode.from_cubin(cubin_code, symbol_mapping=symbol_map)
                else:
                    if debug:
                        print(f'(Kermac Debug) No pre-built cubin, building: {cubin_key}')
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
                    self._db.store(
                        cubin_key,
                        (
                            module_cubin._sym_map[function_name],
                            module_cubin.code
                        )
                    )
                    if debug:
                        print(f'(Kermac Debug) Built and Saved: {cubin_key}')
                self._modules[key] = module_cubin
                return module_cubin
            else:
                if debug:
                    print(f'(Kermac Debug) Loaded module found for (device:{device_id}, function:{function_name})')
            return self._modules[key]
